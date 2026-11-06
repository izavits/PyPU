"""SAR-EM: a classifier and a propensity model learned together by expectation-maximization."""

# Ported from the SAR-PU code of Jessa Bekker and colleagues (MIT License, KU Leuven DTAI),
# keeping its sequence of fits and its arithmetic so that the results match it. See
# THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from pypu.base import BasePUClassifier

# Probabilities are clipped to [_EPS, 1 - _EPS] in the expectation step, as in SAR-PU.
_EPS = 1e-4


class SAREM(BasePUClassifier):
    r"""SAR-EM: learns the classifier and the labeling mechanism together, by EM.

    When labels are selected at random (SAR), the probability that a positive gets labeled,
    the propensity :math:`e(x) = P(s = 1 \mid y = 1, x)`, depends on the sample. SAR-EM [1]_
    fits a classifier :math:`f(x) = P(y = 1 \mid x)` and a propensity model :math:`e(x)` by
    expectation-maximization:

    - **E-step:** the probability that each unlabeled sample is positive,
      :math:`P(y = 1 \mid x, s = 0) = \frac{f(x) (1 - e(x))}{1 - f(x) e(x)}`; labeled samples
      are positive.
    - **M-step:** the propensity model is trained on the labels :math:`s`, each sample
      weighted by its probability of being positive; the classifier is trained on two copies
      of the data, positives weighted by that probability and negatives by its complement.

    EM starts from a classifier trained with the unlabeled samples as negatives, reweighted to
    a class prior of 0.5, and stops when both the log-likelihood and the propensities of the
    unlabeled samples stop changing. With ``refit=True`` the classifier is finally retrained
    with propensity weights: each labeled sample counts as :math:`1 / e(x)` positives and
    :math:`1 - 1 / e(x)` negatives, each unlabeled sample as one negative.

    Classifier and propensity cannot both be learned from the same features without further
    assumptions; restricting the propensity to the features that drive labeling
    (``propensity_features``) helps.

    Parameters
    ----------
    estimator : classifier, default=None
        The classifier :math:`f`, implementing ``predict_proba`` and accepting
        ``sample_weight`` in ``fit`` (negative weights too, with ``refit=True``). It is cloned
        before fitting. ``None`` uses ``LogisticRegression(solver="liblinear")``, as SAR-PU
        does.
    propensity_estimator : classifier, default=None
        The propensity model :math:`e`, with the same requirements except for negative
        weights. ``None`` uses ``LogisticRegression(solver="liblinear")``.
    propensity_features : array-like of int or bool, default=None
        Columns of ``X`` the propensity depends on, as indices or as a boolean mask. ``None``
        uses all columns; an empty list makes the propensity a constant (SCAR).
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-4
        EM stops when, over the last ``convergence_window`` iterations, the log-likelihood
        improved by less than ``tol`` and the propensities of the unlabeled samples changed
        by less than ``tol`` per iteration on average.
    convergence_window : int, default=10
        Number of iterations over which convergence is measured, ``>= 2``.
    refit : bool, default=True
        Whether to retrain the classifier with propensity weights after EM.

    Attributes
    ----------
    estimator_ : classifier
        The fitted classifier.
    propensity_estimator_ : classifier
        The fitted propensity model.
    propensity_features_ : ndarray of int
        The columns used by the propensity model.
    n_iter_ : int
        Number of EM iterations run.
    log_likelihoods_ : ndarray of shape (n_iter_ + 1,)
        Expected log-likelihood per sample after the initialization and after each iteration.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    Notes
    -----
    With the defaults SAREM reproduces ``pu_learn_sar_em`` of SAR-PU, as used in the iFPNU
    experiments.

    A constant propensity is the label frequency :math:`c`. Its EM estimate is accurate when
    the classes barely overlap but can be far too high when they overlap; to estimate
    :math:`c`, prefer :mod:`pypu.prior`. (SAR-PU's constant model estimated
    :math:`P(s = 1)` instead of :math:`c`; PyPU uses the weighted M-step estimate.)

    References
    ----------
    .. [1] J. Bekker, P. Robberechts and J. Davis. "Beyond the selected completely at random
       assumption for learning from positive and unlabeled data." ECML PKDD 2019.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_sar_propensity
    >>> from pypu.models import SAREM
    >>> X, y = make_classification(n_samples=1000, n_features=5, weights=[0.7], random_state=0)
    >>> s, propensity = label_sar_propensity(
    ...     X, y, features=[0], label_frequency=0.4, return_propensity=True, random_state=0
    ... )
    >>> clf = SAREM(propensity_features=[0]).fit(X, s)
    >>> clf.predict_proba(X).shape
    (1000, 2)
    >>> estimated = clf.predict_propensity(X[y == 1])
    >>> float(np.corrcoef(estimated, propensity[y == 1])[0, 1]) > 0.9
    True
    """

    def __init__(
        self,
        estimator=None,
        propensity_estimator=None,
        propensity_features=None,
        max_iter=500,
        tol=1e-4,
        convergence_window=10,
        refit=True,
    ):
        self.estimator = estimator
        self.propensity_estimator = propensity_estimator
        self.propensity_features = propensity_features
        self.max_iter = max_iter
        self.tol = tol
        self.convergence_window = convergence_window
        self.refit = refit

    def predict_propensity(self, X):
        r"""Estimate the propensity :math:`e(x) = P(s = 1 \mid y = 1, x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        propensity : ndarray of shape (n_samples,)
            Probability that each sample would be labeled if it were positive.
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return _positive_proba(self.propensity_estimator_, X[:, self.propensity_features_])

    def _fit(self, X, s):
        max_iter = _check_int(self.max_iter, "max_iter", 1)
        window = _check_int(self.convergence_window, "convergence_window", 2)
        tol = self.tol
        if isinstance(tol, bool) or not isinstance(tol, numbers.Real) or not 0 <= tol < math.inf:
            raise ValueError(f"tol must be a finite number >= 0, got {tol!r}.")
        features = _resolve_features(self.propensity_features, X.shape[1])
        classifier = _check_estimator(self.estimator, "estimator")
        if features.size:
            propensity = _check_estimator(self.propensity_estimator, "propensity_estimator")
        else:
            propensity = _ConstantPropensity()
        X_e = X[:, features]

        # Initialization: unlabeled samples as negatives, reweighted to a class prior of 0.5.
        proportion_labeled = s.sum() / s.size
        classifier.fit(
            X, s, sample_weight=s * (1 - proportion_labeled) + (1 - s) * proportion_labeled
        )
        expectation = _positive_proba(classifier, X)
        propensity.fit(X_e, s, sample_weight=s + (1 - s) * expectation)

        f_x, e_x = self._expectations(classifier, propensity, X, X_e)
        posterior = _posterior(f_x, e_x, s)
        log_likelihoods = [_log_likelihood(f_x, e_x, s)]

        unlabeled = s == 0
        past_propensities = np.zeros([int(unlabeled.sum()), window])
        X_twice = np.concatenate([X, X], axis=0)
        y_twice = np.concatenate(
            [np.ones(s.size, dtype=np.int64), np.zeros(s.size, dtype=np.int64)]
        )
        converged = False
        for i in range(max_iter):
            # Maximization
            propensity.fit(X_e, s, sample_weight=posterior)
            classifier.fit(
                X_twice, y_twice, sample_weight=np.concatenate([posterior, 1 - posterior])
            )
            # Expectation
            f_x, e_x = self._expectations(classifier, propensity, X, X_e)
            posterior = _posterior(f_x, e_x, s)
            log_likelihoods.append(_log_likelihood(f_x, e_x, s))
            # Convergence
            _push(past_propensities, e_x[unlabeled])
            if i > window:
                max_ll_improvement = max(log_likelihoods[-window:]) - log_likelihoods[-window]
                average_abs_slope = np.average(np.abs(_slope(past_propensities, axis=1)))
                if average_abs_slope < tol and max_ll_improvement < tol:
                    converged = True
                    break
        if not converged:
            warnings.warn(
                f"SAREM did not converge in {max_iter} iterations; increase max_iter or tol.",
                ConvergenceWarning,
                stacklevel=3,
            )

        if self.refit:
            weights_positive = s / e_x
            weights_negative = (1 - s) + s * (1 - 1 / e_x)
            classifier.fit(
                X_twice, y_twice, sample_weight=np.concatenate([weights_positive, weights_negative])
            )

        self.estimator_ = classifier
        self.propensity_estimator_ = propensity
        self.propensity_features_ = features
        self.n_iter_ = i + 1
        self.log_likelihoods_ = np.asarray(log_likelihoods)

    @staticmethod
    def _expectations(classifier, propensity, X, X_e):
        f_x = np.clip(_positive_proba(classifier, X), _EPS, 1 - _EPS)
        e_x = np.clip(_positive_proba(propensity, X_e), _EPS, 1 - _EPS)
        return f_x, e_x

    def _predict_proba(self, X):
        return _positive_proba(self.estimator_, X)


class _ConstantPropensity:
    """Propensity model without features: the weighted fraction of labeled samples."""

    classes_ = np.array([0, 1])

    def fit(self, X, s, sample_weight):
        self.propensity_ = float(np.sum(s * sample_weight) / np.sum(sample_weight))
        return self

    def predict_proba(self, X):
        return np.column_stack(
            [np.full(X.shape[0], 1 - self.propensity_), np.full(X.shape[0], self.propensity_)]
        )


def _positive_proba(model, X):
    positive_column = np.flatnonzero(model.classes_ == 1)[0]
    return model.predict_proba(X)[:, positive_column]


def _posterior(f_x, e_x, s):
    """Return P(y = 1 | x, s): 1 for labeled samples, by Bayes' rule for unlabeled ones."""
    return s + (1 - s) * (f_x * (1 - e_x)) / (1 - f_x * e_x)


def _log_likelihood(f_x, e_x, s):
    """Return the expected log-likelihood per sample, as computed by SAR-PU."""
    prob_labeled = f_x * e_x
    prob_unlabeled_pos = np.clip(f_x * (1 - e_x), _EPS, 1 - _EPS)
    prob_unlabeled_neg = np.clip(1 - f_x, _EPS, 1 - _EPS)
    prob_pos_given_unl = prob_unlabeled_pos / (prob_unlabeled_pos + prob_unlabeled_neg)
    prob_neg_given_unl = 1 - prob_pos_given_unl
    return (
        s * np.log(prob_labeled)
        + (1 - s)
        * (
            prob_pos_given_unl * np.log(prob_unlabeled_pos)
            + prob_neg_given_unl * np.log(prob_unlabeled_neg)
        )
    ).mean()


def _slope(array, axis=0):
    """Return the least-squares slope of equidistant values along ``axis``."""
    if axis == 1:
        array = array.transpose()
    n = array.shape[0]
    norm_x = np.asarray(range(n)) - (n - 1) / 2
    auto_cor_x = np.square(norm_x).mean(0)
    avg_y = array.mean(axis=0)
    norm_y = array - avg_y
    cov_x_y = np.matmul(norm_y.transpose(), norm_x) / n
    result = cov_x_y / auto_cor_x
    if axis == 1:
        result = result.transpose()
    return result


def _push(array_queue, new_array):
    """Shift the columns left and put ``new_array`` in the last one."""
    array_queue[:, :-1] = array_queue[:, 1:]
    array_queue[:, -1] = new_array


def _check_int(value, name, lowest):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < lowest:
        raise ValueError(f"{name} must be an int >= {lowest}, got {value!r}.")
    return int(value)


def _check_estimator(estimator, name):
    if estimator is None:
        return LogisticRegression(solver="liblinear")
    estimator = clone(estimator)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"{name} must implement predict_proba; {type(estimator).__name__} does not."
        )
    if not has_fit_parameter(estimator, "sample_weight"):
        raise TypeError(
            f"{name} must accept sample_weight in fit; {type(estimator).__name__} does not."
        )
    return estimator


def _resolve_features(features, n_features):
    """Return the propensity features as an array of column indices."""
    if features is None:
        return np.arange(n_features)
    features = np.asarray(features)
    if features.ndim != 1:
        raise ValueError("propensity_features must be a 1-D list of indices or a boolean mask.")
    if features.dtype == bool:
        if features.size != n_features:
            raise ValueError(
                f"The boolean mask propensity_features has {features.size} entries for "
                f"{n_features} features."
            )
        return np.flatnonzero(features)
    if features.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(features.dtype, np.integer):
        raise ValueError("propensity_features must hold column indices or be a boolean mask.")
    if features.min() < 0 or features.max() >= n_features:
        raise ValueError(
            f"propensity_features must be column indices in [0, {n_features - 1}], got "
            f"{features.tolist()}."
        )
    return features.astype(np.int64)
