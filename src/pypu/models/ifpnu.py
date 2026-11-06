"""IFPNU: reliable negatives from an anomaly detector, then focal-loss gradient boosting."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

from pypu.models.ifpu import _focal_grad, _focal_hess, _FocalBoostingClassifier


class _FocalPNUObjective:
    """LightGBM objective of IFPNU.

    Labeled positives contribute ``prior`` times the focal loss of the positive class,
    reliable negatives ``eta * (1 - prior)`` times that of the negative class, and the other
    unlabeled samples ``1 - eta`` times that of the negative class. The original
    implementation (``pnu_focal_loss`` in the iFPNU code) sums a PN part and a PU part as
    ``eta * pn + (1 - eta) * pu``; the arithmetic below keeps its order of operations, so that
    the results match it exactly.
    """

    def __init__(self, pnu_labels, prior: float, eta: float, gamma: float, clip: float = 1e-7):
        self.positive = pnu_labels == 1
        self.negative = pnu_labels == 0
        self.unlabeled = pnu_labels == -1
        self.prior = prior
        self.eta = eta
        self.gamma = gamma
        self.clip = clip

    def __call__(self, y_true, raw_score):
        if len(raw_score) != self.positive.size:
            raise ValueError("The objective was built for another training set.")
        prior, eta, gamma = self.prior, self.eta, self.gamma
        pos, neg, unl = self.positive, self.negative, self.unlabeled
        prob = 1 / (1 + np.exp(-raw_score))
        prob = np.clip(prob, self.clip, 1 - self.clip)
        grad_pos = _focal_grad(prob, gamma)
        grad_neg = _focal_grad(1 - prob, gamma)
        hess_pos = _focal_hess(prob, gamma)
        hess_neg = _focal_hess(1 - prob, gamma)

        grad = np.zeros_like(prob)
        hess = np.zeros_like(prob)
        # Positives are in both parts, with weight prior.
        grad[pos] = eta * (prior * grad_pos[pos]) + (1 - eta) * (prior * grad_pos[pos])
        hess[pos] = eta * (prior * hess_pos[pos]) + (1 - eta) * (prior * hess_pos[pos])
        # Reliable negatives are in the PN part only.
        grad[neg] = eta * ((1 - prior) * -grad_neg[neg])
        hess[neg] = eta * ((1 - prior) * hess_neg[neg])
        # Unlabeled samples are negatives of the PU part. The original applied this term only
        # while the mean negative-class focal loss over them was positive, which it always is.
        grad[unl] = (1 - eta) * -grad_neg[unl]
        hess[unl] = (1 - eta) * hess_neg[unl]
        return grad, np.maximum(hess, 1e-6)


class IFPNU(_FocalBoostingClassifier):
    r"""Reliable negatives from an anomaly detector, then focal-loss gradient boosting.

    IFPNU turns PU data into PNU data (positive, negative and unlabeled) in two steps:

    1. An unsupervised anomaly detector scores every training sample. The
       :math:`\lfloor \pi n \rfloor` most anomalous samples are taken as possible positives:
       the unlabeled ones among them stay unlabeled, and every other unlabeled sample becomes
       a reliable negative. The labels are kept in :attr:`pnu_labels_`.
    2. LightGBM trees are trained on a focal-loss objective, with the focal losses
       :math:`\ell_\gamma^+` and :math:`\ell_\gamma^-` of the positive and the negative class.
       A labeled positive contributes :math:`\pi \, \ell_\gamma^+(x)`, a reliable negative
       :math:`\eta (1 - \pi) \, \ell_\gamma^-(x)`, and an unlabeled sample
       :math:`(1 - \eta) \, \ell_\gamma^-(x)`.

    This is the objective of the original iFPNU experiments, whose saved results it
    reproduces from the same PNU labels. ``eta`` sets how much the reliable negatives count
    against the unlabeled samples: with the default ``0.5`` their weights differ only by the
    factor :math:`1 - \pi`, so the first step changes the model little, while a larger ``eta``
    leans on the reliable negatives.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    detector : estimator, default=None
        Unsupervised anomaly detector, cloned and fitted on all the training samples. A PyOD
        detector gives its training scores ``decision_scores_``; a scikit-learn one gives
        ``negative_outlier_factor_`` (:class:`~sklearn.neighbors.LocalOutlierFactor`) or
        ``score_samples`` (such as :class:`~sklearn.ensemble.IsolationForest` or
        :class:`~sklearn.svm.OneClassSVM`), negated so that higher means more anomalous.
        ``None`` uses :class:`~sklearn.ensemble.IsolationForest` seeded with
        ``random_state``.
    eta : float, default=0.5
        Weight of the reliable negatives against the unlabeled samples, in ``[0, 1]``.
    gamma : float, default=3.0
        Focusing parameter of the focal loss, ``>= 0``.
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage applied to each tree.
    num_leaves : int, default=31
        Maximum number of leaves per tree.
    max_depth : int, default=-1
        Maximum tree depth; ``-1`` means no limit.
    min_child_samples : int, default=20
        Minimum number of samples in a leaf.
    colsample_bytree : float, default=1.0
        Fraction of the features sampled for each tree.
    reg_lambda : float, default=0.0
        L2 regularization of the leaf values.
    random_state : int or None, default=None
        Seed for the default detector and for LightGBM's random choices (only used when
        sampling features or rows).
    n_jobs : int or None, default=None
        Number of LightGBM threads; ``None`` lets LightGBM decide.
    lgbm_params : dict, default=None
        Further LightGBM parameters, such as ``{"subsample": 0.8, "subsample_freq": 1}``.
        The objective and the number of rounds cannot be set here.

    Attributes
    ----------
    booster_ : lightgbm.Booster
        The fitted trees.
    detector_ : estimator
        The fitted anomaly detector.
    pnu_labels_ : ndarray of shape (n_samples,)
        Labels of the training samples after the first step: ``1`` for labeled positives,
        ``0`` for reliable negatives and ``-1`` for samples left unlabeled.
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
    IFPU : The same boosting with a focal PU objective, without the first step.

    Notes
    -----
    The defaults are the settings of the original experiments: 100 trees, learning rate 0.1,
    ``gamma=3`` and ``eta=0.5``. Those used DIF, EIF and OptIForest as detectors, which
    PyPU does not include; PyOD provides DIF and many others.

    ``decision_function`` returns the raw boosting score, whose sigmoid is
    ``predict_proba(X)[:, 1]``.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import IFPNU
    >>> X, y = make_classification(n_samples=1000, weights=[0.8], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = IFPNU(prior=y_train.mean(), random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    >>> sorted(set(clf.pnu_labels_.tolist()))  # positive, reliable negative, unlabeled
    [-1, 0, 1]
    """

    def __init__(
        self,
        prior=None,
        detector=None,
        eta=0.5,
        gamma=3.0,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        lgbm_params=None,
    ):
        self.prior = prior
        self.detector = detector
        self.eta = eta
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lgbm_params = lgbm_params

    def _fit(self, X, s):
        prior, gamma = self._check_prior_and_gamma(s)
        eta = self.eta
        if isinstance(eta, bool) or not isinstance(eta, numbers.Real) or not 0 <= eta <= 1:
            raise ValueError(f"eta must be a number in [0, 1], got {eta!r}.")
        if self.detector is None:
            detector = IsolationForest(random_state=self.random_state)
        else:
            detector = clone(self.detector)
        detector.fit(X)
        self.detector_ = detector
        self.pnu_labels_ = _pnu_labels(s, _anomaly_scores(detector, X), prior)
        objective = _FocalPNUObjective(self.pnu_labels_, prior=prior, eta=float(eta), gamma=gamma)
        self._fit_booster(X, s, objective)


def _anomaly_scores(detector, X):
    """Return the anomaly score of each training sample; higher means more anomalous."""
    if hasattr(detector, "decision_scores_"):  # PyOD
        scores = detector.decision_scores_
    elif hasattr(detector, "negative_outlier_factor_"):  # scikit-learn's LocalOutlierFactor
        scores = -detector.negative_outlier_factor_
    elif hasattr(detector, "score_samples"):  # scikit-learn: higher means more normal
        scores = -detector.score_samples(X)
    else:
        raise TypeError(
            f"detector must be a PyOD detector or a scikit-learn outlier detector with "
            f"score_samples; {type(detector).__name__} is neither."
        )
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape != (X.shape[0],) or not np.all(np.isfinite(scores)):
        raise ValueError("The detector must give one finite anomaly score per training sample.")
    return scores


def _pnu_labels(s, anomaly_scores, prior):
    """Return the PNU labels: 1 (labeled), -1 (unlabeled) or 0 (reliable negative).

    The unlabeled samples among the floor(prior * n) most anomalous stay unlabeled; the other
    unlabeled samples become reliable negatives.
    """
    n_flagged = int(s.size * prior)
    # Descending scores; ties put the later sample first, as the original's reversed sort.
    order = np.argsort(anomaly_scores, kind="stable")[::-1]
    flagged = np.zeros(s.size, dtype=bool)
    flagged[order[:n_flagged]] = True
    return np.where(s == 1, 1, np.where(flagged, -1, 0))
