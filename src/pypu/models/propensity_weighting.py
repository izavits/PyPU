"""Risk minimization on SAR PU data with a known propensity."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import has_fit_parameter

from pypu.base import BasePUClassifier


class PropensityWeightedPU(BasePUClassifier):
    r"""A classifier trained on PU data weighted by the inverse of a known propensity.

    When labels are selected at random (SAR), each positive :math:`x` gets labeled with its
    own probability, the propensity :math:`e(x) = P(s = 1 \mid y = 1, x)`. Given the
    propensity, the risk of a classifier on the true labels can be estimated without bias
    from PU data [1]_ [2]_: each labeled sample counts as a positive with weight
    :math:`1 / e(x)` and as a negative with weight :math:`1 - 1 / e(x)` (which is not
    positive), and each unlabeled sample as a negative with weight 1. A classifier is trained
    on that weighted data. With a constant propensity, the label frequency :math:`c`, this is
    the unbiased PU risk of SCAR data.

    Parameters
    ----------
    propensity : float or callable
        The propensity :math:`e(x)`, in ``(0, 1]``: a number (a constant propensity, the
        label frequency) or a function that maps an array of samples to their propensities,
        such as the ``predict_propensity`` method of a fitted :class:`SAREM`. Only the labeled
        samples' propensities are used. Required; the default ``None`` raises an error in
        ``fit``.
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba`` and accepting
        ``sample_weight`` in ``fit``, negative weights included. It is cloned before fitting.
        ``None`` uses ``LogisticRegression(solver="liblinear")``, as SAR-PU does.

    Attributes
    ----------
    estimator_ : classifier
        The classifier fitted on the weighted samples.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    SAREM : Estimates the propensity and the classifier together, and refits the classifier
        with these weights.
    pypu.datasets.label_sar_propensity : Makes SAR data whose propensity is known.

    Notes
    -----
    Small propensities give large weights to their labeled samples, and the negative weights
    make the objective non-convex (and unbounded below without regularization): flexible
    classifiers can overfit, and results can depend on the solver and its tolerance. A
    regularized linear model is the safe choice.

    References
    ----------
    .. [1] J. Bekker, P. Robberechts and J. Davis. "Beyond the selected completely at random
       assumption for learning from positive and unlabeled data." ECML PKDD 2019.
    .. [2] J. Bekker and J. Davis. "Learning from positive and unlabeled data: a survey."
       Machine Learning, 109, 719-760 (2020).

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_sar_propensity, label_scar
    >>> from pypu.models import SAREM, PropensityWeightedPU
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)

    A constant propensity, the label frequency of SCAR data:

    >>> s = label_scar(y, label_frequency=0.4, random_state=0)
    >>> clf = PropensityWeightedPU(propensity=0.4).fit(X, s)

    A propensity estimated by SAR-EM, which depends on the first feature:

    >>> s = label_sar_propensity(X, y, features=[0], random_state=0)
    >>> sarem = SAREM(propensity_features=[0]).fit(X, s)
    >>> clf = PropensityWeightedPU(propensity=sarem.predict_propensity).fit(X, s)
    >>> clf.predict_proba(X).shape
    (1000, 2)
    """

    def __init__(self, propensity=None, estimator=None):
        self.propensity = propensity
        self.estimator = estimator

    def _fit(self, X, s):
        estimator = (
            LogisticRegression(solver="liblinear")
            if self.estimator is None
            else clone(self.estimator)
        )
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"estimator must implement predict_proba; {type(estimator).__name__} does not."
            )
        if not has_fit_parameter(estimator, "sample_weight"):
            raise TypeError(
                f"estimator must accept sample_weight in fit; {type(estimator).__name__} does not."
            )
        X_labeled = X[s == 1]
        propensity = self._labeled_propensity(X_labeled)
        n_labeled, n_unlabeled = len(X_labeled), int((s == 0).sum())

        # Labeled samples as positives (1 / e) and as negatives (1 - 1 / e); unlabeled as
        # negatives (1). Copies with a zero weight (e = 1) are dropped.
        X_fit = np.concatenate([X_labeled, X_labeled, X[s == 0]])
        y_fit = np.concatenate([np.ones(n_labeled), np.zeros(n_labeled + n_unlabeled)])
        weight = np.concatenate([1 / propensity, 1 - 1 / propensity, np.ones(n_unlabeled)])
        keep = weight != 0
        self.estimator_ = estimator.fit(X_fit[keep], y_fit[keep], sample_weight=weight[keep])

    def _labeled_propensity(self, X_labeled):
        propensity = self.propensity
        if propensity is None:
            raise ValueError(
                "PropensityWeightedPU needs the propensity: a number in (0, 1] or a function "
                "of the samples."
            )
        if isinstance(propensity, numbers.Real) and not isinstance(propensity, bool):
            values = np.full(len(X_labeled), float(propensity))
        elif callable(propensity):
            values = np.asarray(propensity(X_labeled), dtype=np.float64).reshape(-1)
            if values.shape != (len(X_labeled),):
                raise ValueError(
                    f"propensity returned {values.size} values for {len(X_labeled)} samples."
                )
        else:
            raise TypeError(
                f"propensity must be a number or a callable, got {type(propensity).__name__}."
            )
        if not np.all((values > 0) & (values <= 1)):
            raise ValueError("The propensities of the labeled samples must be in (0, 1].")
        return values

    def _predict_proba(self, X):
        positive_column = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        return self.estimator_.predict_proba(X)[:, positive_column]
