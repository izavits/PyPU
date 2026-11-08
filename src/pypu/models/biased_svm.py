"""Biased SVM: an SVM with class weights chosen by Lee and Liu's criterion."""

from __future__ import annotations

import numbers

import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.svm import SVC

from pypu.base import BasePUClassifier
from pypu.metrics import lee_liu_score

_DEFAULT_POSITIVE_WEIGHTS = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


class BiasedSVM(BasePUClassifier):
    r"""Biased SVM: an SVM on PU labels that penalizes errors on the two classes differently.

    The biased SVM [1]_ is trained to separate labeled from unlabeled samples, with a larger
    penalty for misclassifying a labeled positive (weight :math:`w`) than an unlabeled sample
    (weight :math:`1 - w`), since some unlabeled samples are positives. The weight is chosen
    among ``positive_weights`` by cross-validation, maximizing Lee and Liu's criterion
    :math:`r^2 / P(\hat y = 1)` (:func:`~pypu.metrics.lee_liu_score`), which needs no true
    labels; the SVM is then refitted on all the data with the best weight.

    Parameters
    ----------
    estimator : classifier, default=None
        A scikit-learn classifier with a ``class_weight`` parameter, such as
        :class:`~sklearn.svm.SVC` or :class:`~sklearn.svm.LinearSVC`. It is cloned for each
        fit. ``None`` uses ``SVC()`` (RBF kernel).
    positive_weights : sequence of float, default=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        Candidate weights :math:`w` of the labeled samples, in ``(0, 1)``; unlabeled samples
        get :math:`1 - w`.
    cv : int or cross-validation generator, default=5
        Splitting for the choice of the weight. An int is the number of shuffled, stratified
        folds.
    random_state : int, RandomState instance or None, default=None
        Seed for shuffling the folds when ``cv`` is an int.

    Attributes
    ----------
    estimator_ : classifier
        The SVM refitted on all the data with the best weight.
    positive_weight_ : float
        The chosen weight :math:`w` of the labeled samples.
    cv_scores_ : ndarray of shape (n_weights,)
        Mean Lee and Liu criterion over the folds for each candidate weight.
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
    NaivePU : The same idea without the weights: unlabeled samples as negatives.

    Notes
    -----
    ``decision_function`` returns the SVM's decision value and ``predict_proba`` its
    sigmoid, which is not calibrated; ``predict`` gives the SVM's classification.

    References
    ----------
    .. [1] B. Liu, Y. Dai, X. Li, W. S. Lee and P. S. Yu. "Building text classifiers using
       positive and unlabeled examples." ICDM 2003.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import BiasedSVM
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = BiasedSVM(random_state=0).fit(X_train, s_train)
    >>> 0.7 <= clf.positive_weight_ <= 0.95
    True
    """

    def __init__(
        self, estimator=None, positive_weights=_DEFAULT_POSITIVE_WEIGHTS, cv=5, random_state=None
    ):
        self.estimator = estimator
        self.positive_weights = positive_weights
        self.cv = cv
        self.random_state = random_state

    def _fit(self, X, s):
        weights = np.asarray(self.positive_weights, dtype=np.float64).reshape(-1)
        if weights.size == 0 or not np.all((weights > 0) & (weights < 1)):
            raise ValueError("positive_weights must be a non-empty sequence of numbers in (0, 1).")
        base = SVC() if self.estimator is None else self.estimator
        if "class_weight" not in base.get_params():
            raise TypeError(
                f"estimator must have a class_weight parameter; {type(base).__name__} does not."
            )
        cv = self.cv
        if isinstance(cv, numbers.Integral) and not isinstance(cv, bool):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv = check_cv(cv, s, classifier=True)
        splits = list(cv.split(X, s))

        scores = np.empty(weights.size)
        for i, weight in enumerate(weights):
            fold_scores = []
            for train, test in splits:
                model = _weighted(base, weight).fit(X[train], s[train])
                fold_scores.append(lee_liu_score(s[test], model.predict(X[test])))
            scores[i] = np.mean(fold_scores)
        best = int(np.argmax(scores))
        self.cv_scores_ = scores
        self.positive_weight_ = float(weights[best])
        self.estimator_ = _weighted(base, self.positive_weight_).fit(X, s)

    def _decision_function(self, X):
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        positive_column = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        proba = np.clip(self.estimator_.predict_proba(X)[:, positive_column], 1e-12, 1 - 1e-12)
        return np.log(proba / (1 - proba))

    def _predict_proba(self, X):
        return expit(self._decision_function(X))


def _weighted(estimator, positive_weight):
    """Return an unfitted copy of ``estimator`` with the class weights of ``positive_weight``."""
    return clone(estimator).set_params(class_weight={0: 1.0 - positive_weight, 1: positive_weight})
