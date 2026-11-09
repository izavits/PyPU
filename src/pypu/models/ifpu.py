"""IFPU: gradient boosting with a focal-loss objective for PU data."""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.special import expit

from pypu.base import BasePUClassifier
from pypu.utils.validation import check_prior

try:
    import lightgbm as lgb
except OSError as exc:  # pragma: no cover - depends on the system libraries
    raise ImportError(
        "IFPU and IFPNU use LightGBM, which could not load its OpenMP runtime. On macOS, "
        "install it with `brew install libomp`."
    ) from exc

# LightGBM aliases of the number of boosting rounds. Any of them in the parameters overrides
# n_estimators without an error, which is how the original experiments ended up with 10 trees.
_N_ESTIMATORS_ALIASES = frozenset(
    {
        "n_estimators",
        "num_iterations",
        "num_iteration",
        "n_iter",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "nrounds",
        "num_boost_round",
        "max_iter",
    }
)


def _focal_grad(p, gamma):
    """Return the derivative of the positive-class focal loss with respect to the raw score."""
    return ((1 - p) ** gamma) * (gamma * p * np.log(p) + p - 1)


def _focal_hess(p, gamma):
    """Return the second derivative of the positive-class focal loss in the raw score."""
    hess1 = ((1 - p) ** gamma) * (
        -gamma * (gamma - 1) * p**2 * np.log(p) - 2 * gamma * p * (p - 1) + (p - 1) ** 2
    )
    hess2 = ((1 - p) ** gamma) * (1 - 2 * p) * (gamma * p * np.log(p) + p - 1)
    return hess1 + hess2


class _FocalPUObjective:
    """LightGBM objective of IFPU.

    Each labeled positive contributes the focal loss of the positive class. Each unlabeled
    sample contributes ``prior`` times the focal loss of the positive class plus
    ``1 - prior`` times that of the negative class. The arithmetic follows the original
    implementation (``PUFLLossNonNegative`` in the iFPNU code) step by step, so that the
    results match it.
    """

    def __init__(self, gamma: float, prior: float, clip: float = 1e-7):
        self.gamma = gamma
        self.prior = prior
        self.clip = clip

    def __call__(self, y_true, raw_score):
        gamma, prior = self.gamma, self.prior
        prob = 1 / (1 + np.exp(-raw_score))
        prob = np.clip(prob, self.clip, 1 - self.clip)
        grad_p = _focal_grad(prob, gamma)
        grad_n = _focal_grad(1 - prob, gamma)
        hess_p = _focal_hess(prob, gamma)
        hess_n = _focal_hess(1 - prob, gamma)
        neg_term = (1 - prior) * grad_n
        grad = y_true * grad_p + (1 - y_true) * (prior * grad_p - neg_term)
        hess = y_true * hess_p + (1 - y_true) * (prior * hess_p + (1 - prior) * hess_n)
        return grad, np.maximum(hess, 1e-6)


class _FocalBoostingClassifier(BasePUClassifier):
    """LightGBM with a custom focal-loss objective; shared by IFPU and IFPNU.

    Subclasses store the LightGBM parameters listed in ``_lgbm_params``, plus ``prior``,
    ``gamma`` and ``lgbm_params``, and call ``_fit_booster`` with their objective.
    """

    def _lgbm_params(self) -> dict:
        name = type(self).__name__
        extra = {} if self.lgbm_params is None else dict(self.lgbm_params)
        if "objective" in extra:
            raise ValueError(f"lgbm_params cannot set the objective; it is {name}'s loss.")
        rounds = sorted(_N_ESTIMATORS_ALIASES.intersection(extra))
        if rounds:
            raise ValueError(
                f"lgbm_params cannot set the number of rounds ({', '.join(rounds)}); use "
                "n_estimators."
            )
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
            # Keep features that cannot be split under min_child_samples (they never are, so
            # the trees do not change). With a custom objective LightGBM fails when every
            # feature is filtered out, which happens on very small datasets.
            "feature_pre_filter": False,
            **extra,
        }

    def _check_prior_and_gamma(self, s) -> tuple[float, float]:
        """Return the validated ``prior`` and ``gamma``."""
        if self.prior is None:
            raise ValueError(
                f"{type(self).__name__} needs the class prior: pass prior=P(y=1), the fraction "
                "of positives in the training data."
            )
        prior = check_prior(self.prior, s)
        gamma = self.gamma
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, numbers.Real)
            or not math.isfinite(gamma)
            or gamma < 0
        ):
            raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}.")
        return prior, float(gamma)

    def _fit_booster(self, X, s, objective):
        model = lgb.LGBMClassifier(objective=objective, **self._lgbm_params())
        self.booster_ = model.fit(X, s).booster_

    def _decision_function(self, X):
        return self.booster_.predict(X, raw_score=True)

    def _predict_proba(self, X):
        return expit(self._decision_function(X))


class IFPU(_FocalBoostingClassifier):
    r"""Gradient boosting with a focal-loss objective for PU data.

    IFPU trains LightGBM trees on a custom objective built from the focal loss
    :math:`\ell_\gamma`, which down-weights samples the model already classifies well, so
    that training concentrates on the hard ones. Each labeled positive contributes
    :math:`\ell_\gamma^+(x)`, the focal loss of the positive class. Each unlabeled sample is
    treated as positive with probability :math:`\pi` and contributes
    :math:`\pi \, \ell_\gamma^+(x) + (1 - \pi) \, \ell_\gamma^-(x)`. With ``gamma=0`` the
    focal loss is the cross-entropy.

    The defaults are those of the original iFPU experiments: 10 trees, learning rate 0.1,
    ``gamma=3``. Raising ``n_estimators`` usually needs a lower ``learning_rate``.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    gamma : float, default=3.0
        Focusing parameter of the focal loss, ``>= 0``.
    n_estimators : int, default=10
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
        Seed for LightGBM's random choices (only used when sampling features or rows).
    n_jobs : int or None, default=None
        Number of threads; ``None`` lets LightGBM decide.
    lgbm_params : dict, default=None
        Further LightGBM parameters, such as ``{"subsample": 0.8, "subsample_freq": 1}``.
        The objective and the number of rounds cannot be set here.

    Attributes
    ----------
    booster_ : lightgbm.Booster
        The fitted trees.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    Notes
    -----
    ``decision_function`` returns the raw boosting score, whose sigmoid is
    ``predict_proba(X)[:, 1]``. The scores rank well but are not calibrated probabilities,
    so a threshold derived from the prior, as in :func:`pypu.metrics.f1_at_prior`, can suit
    better than ``predict``'s ``0.5``.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import IFPU
    >>> X, y = make_classification(n_samples=1000, weights=[0.8], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = IFPU(prior=y_train.mean()).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        prior=None,
        gamma=3.0,
        n_estimators=10,
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
        self._fit_booster(X, s, _FocalPUObjective(gamma=gamma, prior=prior))
