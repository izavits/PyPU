"""Gradient boosting (XGBoost) on PU risks: uPU, nnPU and imbalanced nnPU."""

from __future__ import annotations

import numbers

import numpy as np
from scipy.special import expit

from pypu.base import BasePUClassifier
from pypu.utils.validation import check_prior

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "UPUBoost, NNPUBoost and ImbalancedNNPUBoost need XGBoost: install it with "
        "`pip install 'pypu-learn[xgboost]'`."
    ) from exc

# Added to the predicted probabilities, as in the original objectives.
_EPS = 1e-9

_COMMON_PARAMETERS_DOC = """
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.3
        Shrinkage applied to each tree (XGBoost's ``eta``).
    max_depth : int, default=6
        Maximum tree depth.
    min_child_weight : float, default=1.0
        Minimum sum of hessians in a leaf.
    reg_lambda : float, default=0.0
        L2 regularization of the leaf values; the original experiments used none.
    reg_alpha : float, default=0.0
        L1 regularization of the leaf values.
    tree_method : str, default="exact"
        XGBoost's tree construction algorithm.
    random_state : int or None, default=None
        Seed for XGBoost's random choices (only used when sampling rows or features).
    n_jobs : int or None, default=None
        Number of threads; ``None`` lets XGBoost decide.
    xgb_params : dict, default=None
        Further XGBoost parameters, such as ``{"subsample": 0.8}``. The objective cannot be
        set here.
"""


class _RiskBoost(BasePUClassifier):
    """XGBoost trained on a PU risk given as a custom objective."""

    def _xgb_params(self) -> dict:
        extra = {} if self.xgb_params is None else dict(self.xgb_params)
        if "objective" in extra or "obj" in extra:
            raise ValueError(
                f"xgb_params cannot set the objective; it is {type(self).__name__}'s PU risk."
            )
        random_state = self.random_state
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral)
        ):
            raise ValueError(f"random_state must be an int or None, got {random_state!r}.")
        n_estimators = self.n_estimators
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, numbers.Integral)
            or n_estimators < 1
        ):
            raise ValueError(f"n_estimators must be a positive int, got {n_estimators!r}.")
        params = {
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "tree_method": self.tree_method,
            "verbosity": 0,
            "disable_default_eval_metric": True,
        }
        if random_state is not None:
            params["seed"] = int(random_state)
        if self.n_jobs is not None:
            params["nthread"] = self.n_jobs
        params.update(extra)
        return params

    def _fit(self, X, s):
        if self.prior is None:
            raise ValueError(
                f"{type(self).__name__} needs the class prior: pass prior=P(y=1), the fraction "
                "of positives in the training data."
            )
        prior = check_prior(self.prior, s)
        objective = self._objective(prior, float(np.mean(s)))
        params = self._xgb_params()
        dtrain = xgb.DMatrix(X, label=s)
        self.booster_ = xgb.train(
            params, dtrain, num_boost_round=int(self.n_estimators), obj=objective
        )

    def _objective(self, prior, labeled_fraction):  # pragma: no cover - abstract
        raise NotImplementedError

    def _decision_function(self, X):
        return self.booster_.inplace_predict(X)

    def _predict_proba(self, X):
        return expit(self._decision_function(X))


class UPUBoost(_RiskBoost):
    r"""Gradient boosting on the unbiased PU risk (uPU).

    Under SCAR, with label frequency :math:`c = P(s = 1) / \pi`, the cross-entropy risk of
    the true labels can be estimated without bias from PU labels: each labeled sample
    contributes :math:`\frac{1}{c} \ell^+(x) + (1 - \frac{1}{c}) \ell^-(x)` and each unlabeled
    sample :math:`\ell^-(x)`, with :math:`\ell^\pm` the cross-entropy of the positive and the
    negative class [1]_. XGBoost trees are fitted to that risk.

    The labeled samples' negative-class weight :math:`1 - 1/c` is negative, so the risk is
    unbounded below: flexible models drive the scores of labeled samples to large values and
    overfit, which is what nnPU was designed to fix. The defaults are the settings of the
    original experiments: 100 rounds, learning rate 0.3, no L2 regularization.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    """

    __doc__ += _COMMON_PARAMETERS_DOC
    __doc__ += r"""
    Attributes
    ----------
    booster_ : xgboost.Booster
        The fitted trees.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    NNPUBoost : The objective of the original nnPU experiments.

    Notes
    -----
    Needs XGBoost: ``pip install 'pypu-learn[xgboost]'``. ``decision_function`` returns the
    raw boosting score, whose sigmoid is ``predict_proba(X)[:, 1]``.

    References
    ----------
    .. [1] M. C. du Plessis, G. Niu and M. Sugiyama. "Convex formulation for learning from
       positive and unlabeled data." ICML 2015.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import UPUBoost
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = UPUBoost(prior=y_train.mean(), random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        prior=None,
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        min_child_weight=1.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        tree_method="exact",
        random_state=None,
        n_jobs=None,
        xgb_params=None,
    ):
        self.prior = prior
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.xgb_params = xgb_params

    def _objective(self, prior, labeled_fraction):
        return _UPUObjective(prior / labeled_fraction)


class NNPUBoost(_RiskBoost):
    r"""Gradient boosting on the PU objective of the original nnPU experiments.

    The non-negative PU risk (nnPU) corrects the unbiased risk of :class:`UPUBoost`, whose
    negative part can go below zero [1]_. The objective used in the original experiments
    applies the correction sample by sample: it clips each labeled sample's negative-class
    weight :math:`1 - 1/c` at zero, and since that weight is always negative it is always
    removed. The objective is therefore a weighted cross-entropy: each labeled sample counts
    as a positive with weight :math:`1/c` (:math:`c = P(s = 1) / \pi`, the label frequency),
    and each unlabeled sample as a negative with weight 1. PyPU reproduces it, so that the
    experiments' results can be reproduced; it is not Kiryo et al.'s correction, which clips
    the total negative risk.

    The defaults are the settings of the original experiments: 100 rounds, learning rate
    0.3, no L2 regularization.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    """

    __doc__ += _COMMON_PARAMETERS_DOC
    __doc__ += r"""
    Attributes
    ----------
    booster_ : xgboost.Booster
        The fitted trees.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    UPUBoost : The unbiased PU risk.
    ImbalancedNNPUBoost : The objective of the original imbalanced nnPU experiments.

    Notes
    -----
    Needs XGBoost: ``pip install 'pypu-learn[xgboost]'``. ``decision_function`` returns the
    raw boosting score, whose sigmoid is ``predict_proba(X)[:, 1]``.

    References
    ----------
    .. [1] R. Kiryo, G. Niu, M. C. du Plessis and M. Sugiyama. "Positive-unlabeled learning
       with non-negative risk estimator." NeurIPS 2017.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import NNPUBoost
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = NNPUBoost(prior=y_train.mean(), random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        prior=None,
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        min_child_weight=1.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        tree_method="exact",
        random_state=None,
        n_jobs=None,
        xgb_params=None,
    ):
        self.prior = prior
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.xgb_params = xgb_params

    def _objective(self, prior, labeled_fraction):
        return _NNPUObjective(prior / labeled_fraction)


class ImbalancedNNPUBoost(_RiskBoost):
    r"""Gradient boosting on the PU objective of the original imbalanced nnPU experiments.

    Imbalanced nnPU [1]_ trains as if the class prior were ``oversampled_prior``
    (:math:`\rho`, typically 0.5) rather than the small true prior :math:`\pi`, so that the
    rare positive class is not neglected. As for :class:`NNPUBoost`, the objective of the
    original experiments applies the non-negative correction sample by sample, which always
    removes the labeled samples' negative term. The objective is therefore a weighted
    cross-entropy: each labeled sample counts as a positive with weight :math:`\rho` and each
    unlabeled sample as a negative with weight :math:`(1 - \rho) / (1 - \pi)`. It does not
    depend on the label frequency.

    The defaults are the settings of the original experiments: :math:`\rho = 0.5`, 100
    rounds, learning rate 0.3, no L2 regularization.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    oversampled_prior : float, default=0.5
        The class prior :math:`\rho` the model is trained for, in ``(0, 1)``.
    """

    __doc__ += _COMMON_PARAMETERS_DOC
    __doc__ += r"""
    Attributes
    ----------
    booster_ : xgboost.Booster
        The fitted trees.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    NNPUBoost : The objective of the original nnPU experiments.

    Notes
    -----
    Needs XGBoost: ``pip install 'pypu-learn[xgboost]'``. ``decision_function`` returns the
    raw boosting score, whose sigmoid is ``predict_proba(X)[:, 1]``; the probabilities are
    calibrated for the prior :math:`\rho`, not :math:`\pi`.

    References
    ----------
    .. [1] G. Su, W. Chen and M. Xu. "Positive-unlabeled learning from imbalanced data."
       IJCAI 2021.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import ImbalancedNNPUBoost
    >>> X, y = make_classification(n_samples=1000, weights=[0.9], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = ImbalancedNNPUBoost(prior=y_train.mean(), random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        prior=None,
        oversampled_prior=0.5,
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        min_child_weight=1.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        tree_method="exact",
        random_state=None,
        n_jobs=None,
        xgb_params=None,
    ):
        self.prior = prior
        self.oversampled_prior = oversampled_prior
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.xgb_params = xgb_params

    def _objective(self, prior, labeled_fraction):
        rho = check_prior(self.oversampled_prior)
        return _ImbalancedNNPUObjective(prior, rho)


# The objectives follow the original CSBoost code operation by operation. That code ran under
# NumPy 1, where a float32 array times a float64 scalar stays float32; the explicit float32
# casts below reproduce those dtypes, so that the gradients, and the trees, are the same.


def _probabilities(raw_score):
    return expit(np.asarray(raw_score, dtype=np.float32)) + _EPS


class _UPUObjective:
    """uPU: labeled ``(1/c) l+ + (1 - 1/c) l-``, unlabeled ``l-`` (cross-entropies)."""

    def __init__(self, c_inverse: float):
        self.c_inverse = c_inverse

    def __call__(self, raw_score, dtrain):
        y = dtrain.get_label()
        p = _probabilities(raw_score)
        c32, c64 = np.float32(self.c_inverse), np.float64(self.c_inverse)
        y64, p64 = y.astype(np.float64), p.astype(np.float64)
        one_minus_p64 = (1 - p).astype(np.float64)
        grad = (
            (-y * c32).astype(np.float64) * one_minus_p64
            + y64 * (1 - c64) * p64
            + (1 - y).astype(np.float64) * p64
        )
        hess = (
            ((y * c32).astype(np.float64) + y64 * (1 - c64) + (1 - y).astype(np.float64))
            * p64
            * one_minus_p64
        )
        return grad, hess


class _NNPUObjective:
    """The original nnPU objective: labeled ``(1/c) l+``, unlabeled ``l-``."""

    def __init__(self, c_inverse: float):
        self.c_inverse = c_inverse

    def __call__(self, raw_score, dtrain):
        y = dtrain.get_label()
        p = _probabilities(raw_score)
        c32 = np.float32(self.c_inverse)
        # max(0, 1 - y/c): 0 for labeled samples (c < 1), 1 for unlabeled ones.
        weight_negative = np.maximum(np.zeros_like(y), 1 - y * c32)
        grad = -y * (1 - p) * c32 + weight_negative * p
        hess = (y * c32 + weight_negative) * p * (1 - p)
        return grad, hess


class _ImbalancedNNPUObjective:
    """The original imbalanced nnPU objective: weighted cross-entropies.

    Labeled samples contribute ``rho l+``, unlabeled samples ``(1 - rho) / (1 - prior) l-``.
    """

    def __init__(self, prior: float, rho: float):
        self.prior = prior
        self.rho = rho

    def __call__(self, raw_score, dtrain):
        y = dtrain.get_label()
        p = _probabilities(raw_score)
        rho, prior = self.rho, self.prior
        one_minus_prior = np.float32(1 - prior)
        # max(0, ...): (1 - rho) / (1 - prior) for unlabeled samples, 0 for labeled ones.
        weight_negative = np.maximum(
            np.zeros_like(y),
            (1 - rho) * (1 - y) / one_minus_prior
            - np.float32((1 - rho) * prior) * y / one_minus_prior,
        )
        grad = -y * rho * (1 - p) + weight_negative * p
        hess = (y * rho + weight_negative) * p * (1 - p)
        return grad, hess
