"""LBE: PU learning with labeling bias estimation, by expectation-maximization."""

# Ported from the LBE code of the iFPNU experiments, keeping its initialization, its
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

F = torch.nn.functional


class LBE(BasePUClassifier):
    r"""LBE: learns a classifier and the labeling bias together, by EM.

    LBE [1]_ drops the SCAR assumption: the probability that a positive gets labeled,
    :math:`\eta(x) = P(s = 1 \mid y = 1, x)`, may depend on the sample. A classifier
    :math:`h(x) = P(y = 1 \mid x)` and :math:`\eta(x)`, both logistic regressions on all the
    features, are learned together by expectation-maximization, with Adam on the full batch:

    - **Initialization:** :math:`h` is first trained for ``n_pretrain_steps`` steps as a
      classifier of labeled versus unlabeled samples (cross-entropy on :math:`s`);
      :math:`\eta` starts from zero weights, :math:`\eta(x) = 0.5`.
    - **E-step:** the probability that each sample is positive, given the current models: 1
      for labeled samples, and
      :math:`\frac{h(x) (1 - \eta(x))}{h(x) (1 - \eta(x)) + 1 - h(x)}` for unlabeled ones.
    - **M-step:** ``n_m_steps`` Adam steps on the expected negative log-likelihood of both
      models.

    PyPU ports the code of the original experiments, whose settings are the defaults. In its
    expected log-likelihood, the term of an unlabeled sample being negative includes
    :math:`\log(1 - \eta(x))`, which the method does not have (a negative is never labeled).
    With it, the terms of :math:`\eta` add up to the log-likelihood of a logistic regression
    of :math:`s` on :math:`x`, so :math:`\eta` is fitted as :math:`P(s = 1 \mid x)`, the
    labeled-versus-unlabeled classifier, instead of the propensity. Since
    :math:`P(s = 1 \mid x) = h(x) \eta(x)` is lower than :math:`\eta(x)`, the E-step gives
    unlabeled samples a higher probability of being positive than the published method
    would. PyPU keeps this, so that the experiments' results can be reproduced.

    Parameters
    ----------
    n_iter : int, default=100
        Number of EM iterations. There is no convergence test.
    n_m_steps : int, default=100
        Number of Adam steps in each M-step.
    n_pretrain_steps : int, default=1000
        Number of Adam steps that train the classifier on the labels :math:`s` before EM.
    learning_rate : float, default=0.01
        Adam's learning rate, for the pre-training and for EM.
    device : str or torch.device, default="cpu"
        Device to train on, such as ``"cuda"``.
    random_state : int, RandomState instance or None, default=None
        Seed of the classifier's initial weights. An int ``i`` gives the same weights as
        ``torch.manual_seed(i)`` did in the original code.

    Attributes
    ----------
    coef_ : ndarray of shape (`n_features_in_`,)
        Weights of the classifier :math:`h`.
    intercept_ : float
        Intercept of the classifier :math:`h`.
    propensity_coef_ : ndarray of shape (`n_features_in_`,)
        Weights of :math:`\eta`; as explained above, it is a classifier of labeled versus
        unlabeled samples.
    propensity_intercept_ : float
        Intercept of :math:`\eta`.
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
    SAREM : EM for SAR labels with scikit-learn classifiers and a convergence test.

    Notes
    -----
    Needs PyTorch: ``pip install 'pypu-learn[torch]'``. Training uses float32, as the
    original code did, and gives the same weights. Predictions are computed from them in
    float64 with NumPy (the original used float32), so that how the samples are batched
    only affects their last bits. ``decision_function`` returns the classifier's logit,
    ``X @ coef_ + intercept_``, whose sigmoid is ``predict_proba(X)[:, 1]``. Standardize the
    features: the models are linear and trained with a fixed learning rate.

    References
    ----------
    .. [1] C. Gong, Q. Wang, T. Liu, B. Han, J. You, J. Yang and D. Tao. "Instance-dependent
       positive and unlabeled learning with labeling bias estimation." IEEE Transactions on
       Pattern Analysis and Machine Intelligence, 44(8), 2022.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import LBE
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = LBE(random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        n_iter=100,
        n_m_steps=100,
        n_pretrain_steps=1000,
        learning_rate=0.01,
        device="cpu",
        random_state=None,
    ):
        self.n_iter = n_iter
        self.n_m_steps = n_m_steps
        self.n_pretrain_steps = n_pretrain_steps
        self.learning_rate = learning_rate
        self.device = device
        self.random_state = random_state

    def _fit(self, X, s):
        n_iter = check_positive_int("n_iter", self.n_iter)
        n_m_steps = check_positive_int("n_m_steps", self.n_m_steps)
        n_pretrain_steps = check_positive_int("n_pretrain_steps", self.n_pretrain_steps)
        learning_rate = check_positive_float("learning_rate", self.learning_rate)
        device = torch.device(self.device)
        with single_thread():
            weights = _train(
                X, s, n_iter, n_m_steps, n_pretrain_steps, learning_rate, device, self.random_state
            )
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "LBE diverged: its weights are not finite. Standardize the features or lower "
                "learning_rate."
            )
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


def _train(X, s, n_iter, n_m_steps, n_pretrain_steps, learning_rate, device, random_state):
    """Train both models as the original code does; return their weights as float64 arrays."""
    n_features = X.shape[1]
    classifier = _Logistic(n_features)
    init_linear_(classifier.weight, classifier.bias, make_generator(random_state))
    propensity = _Logistic(n_features)
    with torch.no_grad():
        propensity.weight.zero_()
        propensity.bias.zero_()
    classifier.to(device)
    propensity.to(device)
    X = as_float32_tensor(X, device)
    s = torch.from_numpy(s).float().to(device)

    # Pre-training: the classifier separates labeled from unlabeled samples.
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    for _ in range(n_pretrain_steps):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(classifier(X), s)
        loss.backward()
        optimizer.step()

    # EM, with a new optimizer over both models.
    optimizer = torch.optim.Adam(
        [*classifier.parameters(), *propensity.parameters()], lr=learning_rate
    )
    for _ in range(n_iter):
        posterior = _e_step(classifier, propensity, X, s)
        for _ in range(n_m_steps):
            optimizer.zero_grad()
            loss = _negative_log_likelihood(classifier(X), propensity(X), s, posterior)
            loss.backward()
            optimizer.step()
    return [
        p.detach().cpu().numpy().astype(np.float64).ravel()
        for p in (classifier.weight, classifier.bias, propensity.weight, propensity.bias)
    ]


class _Logistic(torch.nn.Module):
    """A logistic regression's logit, as the original's one-layer ``nn.Sequential``."""

    def __init__(self, n_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1, n_features))
        self.bias = torch.nn.Parameter(torch.empty(1))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias).squeeze(1)


# The E- and M-step follow the original code operation by operation, so that the results are
# the same to the last bit.


@torch.no_grad()
def _e_step(classifier, propensity, X, s):
    """Return the probabilities of the negative (column 0) and positive class of each sample."""
    h = torch.sigmoid(classifier(X))
    eta = torch.sigmoid(propensity(X))
    positive = torch.where(s == 1, eta, 1 - eta) * h
    negative = torch.where(s == 1, 0, 1) * (1 - h)
    posterior = torch.cat([negative.reshape(-1, 1), positive.reshape(-1, 1)], dim=1)
    posterior /= posterior.sum(dim=1).reshape(-1, 1)
    return posterior


def _negative_log_likelihood(h_logit, eta_logit, s, posterior):
    """Return the expected negative log-likelihood, with the original's terms (see LBE)."""
    log_h = F.logsigmoid(h_logit)
    log_1_minus_h = F.logsigmoid(-h_logit)
    log_eta = F.logsigmoid(eta_logit)
    log_1_minus_eta = F.logsigmoid(-eta_logit)
    log_likelihood = torch.where(
        s == 1,
        posterior[:, 1] * (log_h + log_eta) + posterior[:, 0] * (log_1_minus_h + log_eta),
        posterior[:, 1] * (log_h + log_1_minus_eta)
        + posterior[:, 0] * (log_1_minus_h + log_1_minus_eta),
    )
    return -torch.sum(log_likelihood)
