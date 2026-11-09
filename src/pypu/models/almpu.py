"""ALM-PU: PU learning as constrained optimization, solved with an augmented Lagrangian."""

# Ported from the ALM-PU tabular benchmark code of the PyPU author, which adapts the code of
# the ALM-PU authors (github.com/JiazheWei/ALM-PU) and uses the EMA model and cosine schedule
# of FixMatch-pytorch (MIT). It keeps the order of every random draw and the arithmetic, so
# that the results match it. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import copy
import math
import numbers

import numpy as np
from scipy.special import expit

from pypu.base import BasePUClassifier
from pypu.models._torch import (
    as_float32_tensor,
    check_positive_float,
    check_positive_int,
    single_thread,
    torch,
    torch_seed,
)

try:
    import jenkspy
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "ALMPU needs jenkspy: install it with `pip install 'pypu-learn[torch]'`."
    ) from exc

F = torch.nn.functional
DataLoader = torch.utils.data.DataLoader
RandomSampler = torch.utils.data.RandomSampler
TensorDataset = torch.utils.data.TensorDataset

# Internally, as in the original, class 0 is the positive class and class 1 the negative one.
_POSITIVE, _NEGATIVE = 0, 1


class ALMPU(BasePUClassifier):
    r"""ALM-PU: a neural network trained on PU data with an augmented Lagrangian.

    ALM-PU [1]_ casts PU learning as constrained optimization: minimize the classification
    loss on the labeled positives, subject to a constraint on the loss of treating the
    unlabeled samples as negatives, and solves it with an augmented Lagrangian method. PyPU
    ports the code of the original tabular experiments, whose settings are the defaults:

    - **Phase 1** (``n_warmup_epochs`` epochs of ``n_steps_per_epoch`` steps): a small MLP is
      trained on batches of labeled samples (cross-entropy with ``label_smoothing``) and of
      unlabeled samples taken as negatives, whose cross-entropy :math:`L_n` enters through
      the augmented-Lagrangian penalty of the constraint :math:`L_n \le \alpha`. After each
      epoch the multiplier is updated from the epoch's mean :math:`L_n` and
      ``false_alarm_cutoff``, the penalty weight is multiplied by ``penalty_multiplier`` while
      the constraint is violated, and an exponential moving average of the model
      (``ema_decay``) predicts every unlabeled sample.
    - **Pseudo-labels:** each unlabeled sample's trend is the mean of
      :math:`\log(1 + d + d^2 / 2)` over the changes :math:`d` of its predicted
      probability of being positive between epochs. Jenks natural breaks split the trends in
      two, and the samples above the break become pseudo-positives.
    - **Phase 2** (``n_finetune_epochs`` epochs): a new MLP is trained on the labeled samples
      and on the unlabeled ones, whose targets move from negative towards their pseudo-labels
      over the epochs (weight :math:`(e / E)^{0.8}` at epoch :math:`e` of :math:`E`). The
      model after the last epoch is kept.

    The MLP has hidden layers of 16, 18, 20 and 24 ReLU units, with dropout 0.25 after the
    second, and is trained by SGD with Nesterov momentum 0.9, weight decay on the weights
    (not the biases) and a cosine learning-rate schedule. Each step uses ``batch_size``
    labeled samples, drawn with replacement, and ``batch_size`` unlabeled ones.

    Parameters
    ----------
    batch_size : int, default=64
        Number of labeled samples, and of unlabeled samples, in each step.
    n_steps_per_epoch : int, default=100
        Number of optimizer steps in an epoch.
    n_warmup_epochs : int, default=20
        Number of epochs of phase 1, ``>= 2`` (the trends need two predictions).
    n_finetune_epochs : int, default=30
        Number of epochs of phase 2.
    learning_rate : float, default=0.01
        Initial learning rate of both phases.
    weight_decay : float, default=5e-4
        L2 penalty on the weights (not the biases).
    ema_decay : float or None, default=0.99
        Decay of the moving average of the model whose predictions make the trends; ``None``
        uses the model itself.
    label_smoothing : float, default=0.1
        Label smoothing of the labeled samples' cross-entropy in phase 1, in ``[0, 1)``.
    alpha : float, default=7e-5
        Target :math:`\alpha` of the constraint on the unlabeled samples' loss, inside the
        penalty.
    false_alarm_cutoff : float, default=0.05
        Level of the unlabeled samples' mean loss above which the multiplier grows.
    penalty_multiplier : float, default=1.05
        Factor by which the penalty weight grows after each epoch while the constraint is
        violated.
    device : str or torch.device, default="cpu"
        Device to train on, such as ``"cuda"``.
    random_state : int, RandomState instance or None, default=None
        Seed of PyTorch's random numbers during ``fit`` (initial weights, batches and
        dropout). An int ``i`` gives the same numbers as ``set_seed(i)`` did in the original
        experiments; PyTorch's global random state is restored afterwards.

    Attributes
    ----------
    network_ : torch.nn.Module
        The MLP trained in phase 2. Its output column 0 is the positive class.
    pseudo_labels_ : ndarray of shape (n_unlabeled,)
        Pseudo-labels of the unlabeled training samples, in their order in ``X``: 1 for
        pseudo-positive, 0 for pseudo-negative.
    trends_ : ndarray of shape (n_unlabeled,)
        Trends of the unlabeled training samples' predicted probabilities in phase 1.
    break_point_ : float
        The Jenks break between the two groups of trends.
    pollution_ : float
        Fraction of the unlabeled training samples pseudo-labeled positive, an estimate of
        the share of positives among them.
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
    Needs PyTorch and jenkspy: ``pip install 'pypu-learn[torch]'``. Training uses float32 on
    one thread, as the original experiments did, and gives the same network. Predictions are
    computed from it in float64, so that how the samples are batched only affects their last
    bits; ``decision_function`` returns the difference of the network's two outputs, whose
    sigmoid is ``predict_proba(X)[:, 1]``. Standardize the features.

    The pseudo-labels can go wrong. In 159 of the 840 runs of the original tabular benchmark,
    mostly on its smallest datasets, about 60% of the unlabeled samples were pseudo-labeled
    positive where about 3% were positive: the true positives' trends fell mostly below the
    break, and phase 2 learned an inverted ranking (test ROC AUC below 0.5). A ``pollution_``
    far above the expected share of positives among the unlabeled samples is the sign.

    References
    ----------
    .. [1] J. Wei, Y. Wu, B. Shi, K. Li and B. Dong. "ALM-PU: positive and unlabeled learning
       with constrained optimization." Machine Learning, 114, 210 (2025).

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import ALMPU
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = ALMPU(random_state=0).fit(X_train, s_train)
    >>> clf.predict_proba(X_test).shape
    (250, 2)
    """

    def __init__(
        self,
        batch_size=64,
        n_steps_per_epoch=100,
        n_warmup_epochs=20,
        n_finetune_epochs=30,
        learning_rate=0.01,
        weight_decay=5e-4,
        ema_decay=0.99,
        label_smoothing=0.1,
        alpha=7e-5,
        false_alarm_cutoff=0.05,
        penalty_multiplier=1.05,
        device="cpu",
        random_state=None,
    ):
        self.batch_size = batch_size
        self.n_steps_per_epoch = n_steps_per_epoch
        self.n_warmup_epochs = n_warmup_epochs
        self.n_finetune_epochs = n_finetune_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.false_alarm_cutoff = false_alarm_cutoff
        self.penalty_multiplier = penalty_multiplier
        self.device = device
        self.random_state = random_state

    def _settings(self) -> dict:
        n_warmup_epochs = check_positive_int("n_warmup_epochs", self.n_warmup_epochs)
        if n_warmup_epochs < 2:
            raise ValueError(f"n_warmup_epochs must be at least 2, got {n_warmup_epochs}.")
        ema_decay = self.ema_decay
        if ema_decay is not None and not _is_number_in(ema_decay, 0, 1, closed_low=True):
            raise ValueError(f"ema_decay must be None or a number in [0, 1), got {ema_decay!r}.")
        if not _is_number_in(self.label_smoothing, 0, 1, closed_low=True):
            raise ValueError(
                f"label_smoothing must be a number in [0, 1), got {self.label_smoothing!r}."
            )
        for name in ("weight_decay", "alpha", "false_alarm_cutoff"):
            value = getattr(self, name)
            if not _is_number_in(value, 0, math.inf, closed_low=True):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}.")
        return {
            "batch_size": check_positive_int("batch_size", self.batch_size),
            "n_steps_per_epoch": check_positive_int("n_steps_per_epoch", self.n_steps_per_epoch),
            "n_warmup_epochs": n_warmup_epochs,
            "n_finetune_epochs": check_positive_int("n_finetune_epochs", self.n_finetune_epochs),
            "learning_rate": check_positive_float("learning_rate", self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "ema_decay": None if ema_decay is None else float(ema_decay),
            "label_smoothing": float(self.label_smoothing),
            "alpha": float(self.alpha),
            "false_alarm_cutoff": float(self.false_alarm_cutoff),
            "penalty_multiplier": check_positive_float(
                "penalty_multiplier", self.penalty_multiplier
            ),
            "device": torch.device(self.device),
        }

    def _fit(self, X, s):
        settings = self._settings()
        device = settings["device"]
        labeled = TensorDataset(
            as_float32_tensor(X[s == 1]), torch.full((int(s.sum()),), _POSITIVE)
        )
        X_unlabeled = as_float32_tensor(X[s == 0])
        unlabeled = TensorDataset(X_unlabeled, torch.arange(len(X_unlabeled)))
        n_features = X.shape[1]

        # The original seeded PyTorch's global generator and drew everything from it: do the
        # same, and restore the caller's random state afterwards.
        devices = [device] if device.type == "cuda" else []
        with single_thread(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(torch_seed(self.random_state))
            trends = _phase1(labeled, unlabeled, n_features, settings)
            try:
                break_point = float(jenkspy.jenks_breaks(trends, n_classes=2)[1])
            except ValueError as exc:
                raise ValueError(
                    "ALMPU cannot split the unlabeled samples: the trends of their predicted "
                    "probabilities take fewer than two distinct values."
                ) from exc
            pseudo_targets = np.where(trends > break_point, _POSITIVE, _NEGATIVE)
            network = _phase2(labeled, unlabeled, pseudo_targets, n_features, settings)

        self.network_ = network.cpu().eval()
        self.trends_ = trends
        self.break_point_ = break_point
        self.pseudo_labels_ = (pseudo_targets == _POSITIVE).astype(np.int64)
        self.pollution_ = float(self.pseudo_labels_.mean())

    def _decision_function(self, X):
        network = copy.deepcopy(self.network_).double().eval()
        with single_thread(), torch.no_grad():
            logits = network(torch.from_numpy(np.asarray(X, dtype=np.float64)))
        return (logits[:, _POSITIVE] - logits[:, _NEGATIVE]).numpy()

    def _predict_proba(self, X):
        return expit(self._decision_function(X))


def _is_number_in(value, low, high, *, closed_low) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        return False
    return (low <= value if closed_low else low < value) and value < high


class _MLP(torch.nn.Module):
    """The original's ``CardMLP``: hidden layers of 16, 18, 20 and 24 units, two outputs."""

    def __init__(self, n_features: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, 16)
        self.fc2 = torch.nn.Linear(16, 18)
        self.fc3 = torch.nn.Linear(18, 20)
        self.fc4 = torch.nn.Linear(20, 24)
        self.fc5 = torch.nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class _EMA:
    """Exponential moving average of a model's parameters (FixMatch-pytorch's ``ModelEMA``)."""

    def __init__(self, model, decay: float):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            source = model.state_dict()
            for key, value in self.model.state_dict().items():
                value.copy_(value * self.decay + (1.0 - self.decay) * source[key].detach())


def _make_model(n_features, total_steps, settings):
    """Return the MLP, SGD with weight decay on the weights only, and the cosine schedule."""
    model = _MLP(n_features).to(settings["device"])
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if "bias" not in n],
            "weight_decay": settings["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if "bias" in n], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.SGD(groups, lr=settings["learning_rate"], momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_schedule(total_steps))
    return model, optimizer, scheduler


def _cosine_schedule(total_steps, num_cycles=7.0 / 16.0):
    """FixMatch-pytorch's cosine schedule, without warm-up."""

    def factor(step):
        progress = float(step) / float(max(1, total_steps))
        return max(0.0, math.cos(math.pi * num_cycles * progress))

    return factor


def _batches(labeled, unlabeled, settings):
    """Endless batches of labeled samples (with replacement) and of unlabeled samples."""
    batch_size, n_steps = settings["batch_size"], settings["n_steps_per_epoch"]
    labeled_loader = DataLoader(
        labeled,
        batch_size=batch_size,
        sampler=RandomSampler(labeled, replacement=True, num_samples=batch_size * n_steps),
    )
    unlabeled_loader = DataLoader(
        unlabeled, batch_size=batch_size, shuffle=True, drop_last=len(unlabeled) >= batch_size
    )
    return _cycle(labeled_loader), _cycle(unlabeled_loader)


def _cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def _predict_positive(model, X, device):
    """P(positive) of each sample, in float32 as in the original."""
    model.eval()
    proba = torch.softmax(model(X.to(device)), dim=1)[:, _POSITIVE].cpu().numpy()
    model.train()
    return proba


def _phase1(labeled, unlabeled, n_features, settings):
    """Train with the augmented-Lagrangian penalty; return the unlabeled samples' trends."""
    device, n_steps = settings["device"], settings["n_steps_per_epoch"]
    alpha, smoothing = settings["alpha"], settings["label_smoothing"]
    model, optimizer, scheduler = _make_model(
        n_features, settings["n_warmup_epochs"] * n_steps, settings
    )
    ema = None if settings["ema_decay"] is None else _EMA(model, settings["ema_decay"])
    labeled_batches, unlabeled_batches = _batches(labeled, unlabeled, settings)

    multiplier, penalty_weight = 0.0, 1.0
    predictions = []
    model.train()
    for _ in range(settings["n_warmup_epochs"]):
        unlabeled_loss_total = 0.0
        for _ in range(n_steps):
            inputs_x, targets_x = next(labeled_batches)
            inputs_u, _ = next(unlabeled_batches)
            batch_size = inputs_x.shape[0]
            logits = model(torch.cat((inputs_x, inputs_u)).to(device))
            logits_x, logits_u = logits[:batch_size], logits[batch_size:]
            targets_u = torch.full((len(inputs_u),), _NEGATIVE, device=device)

            loss_x = F.cross_entropy(logits_x, targets_x.to(device), label_smoothing=smoothing)
            loss_u = F.cross_entropy(logits_u, targets_u)
            # The augmented-Lagrangian (PHR) penalty of the constraint loss_u <= alpha
            violation = loss_u - alpha
            if penalty_weight * violation.item() + multiplier >= 0:
                penalty = violation * multiplier + (penalty_weight / 2) * violation * violation
            else:
                penalty = torch.tensor(
                    -(multiplier * multiplier) / (2 * penalty_weight), device=device
                )
            loss = loss_x + loss_u + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            unlabeled_loss_total += loss_u.item()

        # The multiplier and the penalty weight are updated once per epoch.
        mean_unlabeled_loss = unlabeled_loss_total / n_steps
        constraint = mean_unlabeled_loss - settings["false_alarm_cutoff"]
        if constraint * penalty_weight + multiplier >= 0:
            multiplier += constraint
        else:
            multiplier += -multiplier / penalty_weight
        if mean_unlabeled_loss - alpha >= 0:
            penalty_weight *= settings["penalty_multiplier"]

        recorder = model if ema is None else ema.model
        predictions.append(_predict_positive(recorder, unlabeled.tensors[0], device))

    predictions = np.stack(predictions, axis=1)
    changes = np.diff(predictions, axis=1)
    return np.log(1 + changes + 0.5 * changes**2).mean(axis=1)


def _phase2(labeled, unlabeled, pseudo_targets, n_features, settings):
    """Train a new MLP towards the pseudo-labels; return it."""
    device, n_steps = settings["device"], settings["n_steps_per_epoch"]
    n_epochs = settings["n_finetune_epochs"]
    model, optimizer, scheduler = _make_model(n_features, n_epochs * n_steps, settings)
    pseudo_targets = torch.from_numpy(pseudo_targets).long()
    labeled_batches, unlabeled_batches = _batches(labeled, unlabeled, settings)

    model.train()
    for epoch in range(n_epochs):
        weight = (epoch / n_epochs) ** 0.8
        for _ in range(n_steps):
            inputs_x, targets_x = next(labeled_batches)
            inputs_u, index_u = next(unlabeled_batches)
            batch_size = inputs_x.shape[0]
            logits = model(torch.cat((inputs_x, inputs_u)).to(device))
            logits_x, logits_u = logits[:batch_size], logits[batch_size:]

            loss_x = F.cross_entropy(logits_x, targets_x.to(device))
            observed = F.one_hot(torch.full((len(inputs_u),), _NEGATIVE), 2).float()
            pseudo = F.one_hot(pseudo_targets[index_u], 2).float()
            targets_u = (weight * pseudo + (1 - weight) * observed).to(device)
            loss = loss_x + F.cross_entropy(logits_u, targets_u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model
