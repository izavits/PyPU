"""PUe: PU learning with normalized inverse propensity weights."""

# Ported from the PUe code of the iFPNU experiments, keeping its initialization, its
# sequence of optimizer steps and its arithmetic, so that the results match it.

from __future__ import annotations

import numpy as np
from scipy.special import expit

from pypu.base import BasePUClassifier
from pypu.models._torch import (
    as_float32_tensor,
    check_positive_float,
    check_positive_int,
    init_linear_,
    make_generator,
    single_thread,
    torch,
)
from pypu.utils.validation import check_prior

F = torch.nn.functional


class PUe(BasePUClassifier):
    r"""PUe: a PU classifier trained with normalized inverse propensity weights.

    When labels are selected at random (SAR), PUe [1]_ estimates the propensity
    :math:`e(x)`, the probability that a sample gets labeled, and weights each labeled sample
    by its inverse, normalized so that the weights of the labeled samples sum to one. PyPU
    ports the code of the original experiments, whose settings are the defaults:

    - **Class prior:** the KM1 estimate of :class:`~pypu.prior.KMPrior`, unless ``prior``
      is given.
    - **Propensity:** a logistic regression :math:`e`, trained for ``n_steps`` Adam steps
      on the cross-entropy of the labels :math:`s`, averaged over the samples, plus
      ``alpha`` times :math:`|\sum_i e(x_i) - n_l|`, which pulls the sum of the propensities
      towards the number :math:`n_l` of labeled samples.
    - **Classifier:** a logistic regression :math:`f`, trained for ``n_steps`` Adam steps on
      :math:`\pi \sum_{s_i = 1} w_i \ell^+(x_i) + \sum_{s_i = 0} \ell^-(x_i)
      - \pi \sum_{s_i = 1} w_i \ell^-(x_i)`, with :math:`\ell^\pm` the cross-entropies of
      the two classes and :math:`w_i \propto 1 / e(x_i)` the normalized weights.

    The unlabeled term is a sum over the unlabeled samples, while the labeled weights sum to
    one: compared with a risk estimate, which would average the unlabeled term, the
    unlabeled samples weigh :math:`n_u` times more. The negative term is not clipped.

    The experiments trained each model for only 20 Adam steps with learning rate 0.001,
    from random initial weights, which barely moves them: the scores stay close to those of
    the initial weights, and depend on ``random_state`` more than on the data (across the
    experiments' 84 settings, the mean test ROC AUC was 0.54). PyPU keeps these defaults so
    that the experiments can be reproduced; raise ``n_steps`` and ``learning_rate`` for a
    trained model.

    Parameters
    ----------
    prior : float or None, default=None
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. ``None``
        estimates it with KM1, as the experiments did, which needs cvxopt.
    n_steps : int, default=20
        Number of Adam steps for each of the two models, on the full batch.
    learning_rate : float, default=0.001
        Adam's learning rate.
    alpha : float, default=15.0
        Weight of the propensity's penalty :math:`|\sum_i e(x_i) - n_l|`.
    device : str or torch.device, default="cpu"
        Device to train on, such as ``"cuda"``.
    random_state : int, RandomState instance or None, default=None
        Seed of the initial weights of both models. An int ``i`` gives the same weights as
        ``seed(i)`` did in the experiments (``torch.manual_seed(i)``).

    Attributes
    ----------
    prior_ : float
        The class prior used: ``prior``, or the KM1 estimate.
    coef_ : ndarray of shape (`n_features_in_`,)
        Weights of the classifier :math:`f`.
    intercept_ : float
        Intercept of the classifier :math:`f`.
    propensity_coef_ : ndarray of shape (`n_features_in_`,)
        Weights of the propensity model :math:`e`.
    propensity_intercept_ : float
        Intercept of the propensity model :math:`e`.
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
    pypu.prior.KMPrior : The KM estimator of the class prior.
    SAREM : EM for SAR labels with scikit-learn classifiers.

    Notes
    -----
    Needs PyTorch (``pip install 'pypu-learn[torch]'``), and cvxopt for the KM estimate of
    the prior (``pip install 'pypu-learn[cvxopt]'``; GPL-3.0). Training uses float32, as the
    original code did, and gives the same weights. Predictions are computed from them in
    float64 with NumPy (the original used float32), so that how the samples are batched
    only affects their last bits. ``decision_function`` returns the classifier's logit,
    ``X @ coef_ + intercept_``, whose sigmoid is ``predict_proba(X)[:, 1]``.

    References
    ----------
    .. [1] X. Wang, H. Chen, T. Guo and Y. Wang. "PUe: Biased positive-unlabeled learning
       enhancement by causal inference." NeurIPS 2023.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import PUe
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = PUe(prior=y_train.mean(), random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        prior=None,
        n_steps=20,
        learning_rate=0.001,
        alpha=15.0,
        device="cpu",
        random_state=None,
    ):
        self.prior = prior
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.device = device
        self.random_state = random_state

    def _fit(self, X, s):
        n_steps = check_positive_int("n_steps", self.n_steps)
        learning_rate = check_positive_float("learning_rate", self.learning_rate)
        alpha = self.alpha
        if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not alpha >= 0:
            raise ValueError(f"alpha must be a number >= 0, got {alpha!r}.")
        device = torch.device(self.device)
        prior = _km1_prior(X, s) if self.prior is None else check_prior(self.prior, s)

        with single_thread():
            weights = _train(X, s, prior, n_steps, learning_rate, alpha, device, self.random_state)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "PUe diverged: its weights are not finite. Standardize the features or lower "
                "learning_rate."
            )
        self.prior_ = float(prior)
        self.coef_, (self.intercept_,), self.propensity_coef_, (self.propensity_intercept_,) = (
            weights
        )

    def _decision_function(self, X):
        # In float64 from the fitted float32 weights. The rounding of a matrix product can
        # depend on how many rows are multiplied at once: in float32 that changes predictions
        # by up to 1e-7 depending on batching, in float64 only in the last bits.
        return X @ self.coef_ + self.intercept_

    def _predict_proba(self, X):
        return expit(self._decision_function(X))


def _km1_prior(X, s):
    """Return the experiments' prior: the KM1 estimate, combined as the original did.

    The original's PU labels were float32, so their mean was a float32 (the rest of the
    arithmetic was float64).
    """
    from pypu.prior._km import km_default

    km1 = km_default(X[s == 0], X[s == 1])[1]
    labeled_fraction = float(np.mean(s.astype(np.float32)))
    return (1 - labeled_fraction) * km1 + labeled_fraction


def _train(X, s, prior, n_steps, learning_rate, alpha, device, random_state):
    """Train both models as the original code does; return their weights as float64 arrays."""
    n_features = X.shape[1]
    generator = make_generator(random_state)
    propensity = _Linear(n_features)
    init_linear_(propensity.weight, propensity.bias, generator)
    propensity.to(device)
    X = as_float32_tensor(X, device)
    s = torch.from_numpy(s).float().view(-1, 1).to(device)

    # The propensity model: cross-entropy of s plus the penalty on the sum of propensities.
    n_labeled = torch.sum(s)
    n_unlabeled = s.shape[0] - torch.sum(s)
    optimizer = torch.optim.Adam(propensity.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = _propensity_loss(propensity(X), s, n_labeled, n_unlabeled, alpha)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        propensity_scores = torch.sigmoid(propensity(X))
    normalized_scores = torch.sum(torch.where(s == 1, 1 / propensity_scores, 0)) * propensity_scores

    # The classifier, initialized from the same random stream after the propensity model.
    classifier = _Linear(n_features)
    init_linear_(classifier.weight, classifier.bias, generator)
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = _classifier_loss(classifier(X), s, normalized_scores, prior)
        loss.backward()
        optimizer.step()

    return [
        p.detach().cpu().numpy().astype(np.float64).ravel()
        for p in (classifier.weight, classifier.bias, propensity.weight, propensity.bias)
    ]


class _Linear(torch.nn.Module):
    """A logistic regression's logit of shape (n, 1), as the original's ``nn.Linear``."""

    def __init__(self, n_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1, n_features))
        self.bias = torch.nn.Parameter(torch.empty(1))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# The losses follow the original code operation by operation, so that the results are the
# same to the last bit.


def _propensity_loss(logit, s, n_labeled, n_unlabeled, alpha):
    """Return the original propensity loss (its ``CustomLoss_e``)."""
    loss1 = torch.sum(-1 / (n_labeled + n_unlabeled) * s * F.logsigmoid(logit))
    loss2 = torch.sum(-1 / (n_labeled + n_unlabeled) * (1 - s) * F.logsigmoid(-logit))
    regularisation = alpha * torch.abs(torch.sum(torch.sigmoid(logit)) - n_labeled)
    return loss1 + loss2 + regularisation


def _classifier_loss(logit, s, propensity_scores, prior):
    """Return the original classifier loss (its ``CustomLoss_clf``)."""
    loss1 = torch.sum(-1 / propensity_scores * s * F.logsigmoid(logit))
    loss2 = torch.sum(-(1 - s) * F.logsigmoid(-logit))
    loss3 = torch.sum(-1 / propensity_scores * s * F.logsigmoid(-logit))
    return prior * loss1 + loss2 - prior * loss3
