"""Helpers shared by the models built on PyTorch (an optional dependency)."""

from __future__ import annotations

import contextlib
import math
import numbers
import sys

import numpy as np
from sklearn.utils import check_random_state

# On macOS, PyTorch's wheels bundle their own OpenMP runtime, while LightGBM (a dependency of
# PyPU) and XGBoost use Homebrew's. With both runtimes in one process, LightGBM crashes if
# PyTorch was loaded first, and PyTorch crashes when it runs on several threads. So LightGBM
# is loaded before PyTorch, and the PyTorch models run on a single thread (single_thread).
if sys.platform == "darwin":  # pragma: no cover - platform dependent
    with contextlib.suppress(ImportError, OSError):
        import lightgbm  # noqa: F401

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "LBE and PUe need PyTorch: install it with `pip install 'pypu-learn[torch]'` (see "
        "https://pytorch.org for builds with GPU support)."
    ) from exc

_MAX_SEED = np.iinfo(np.int32).max


@contextlib.contextmanager
def single_thread():
    """Run PyTorch's CPU operations on one thread, then restore the number of threads.

    Besides avoiding the OpenMP conflict on macOS (see above), this makes the results
    independent of the number of cores: PyTorch splits large reductions across threads.
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n_threads)


def make_generator(random_state) -> torch.Generator:
    """Return a CPU generator seeded from ``random_state``.

    An int is used as the seed itself, so ``random_state=i`` gives the same numbers as
    ``torch.manual_seed(i)`` did in the original code. ``None`` and ``RandomState`` instances
    draw the seed from NumPy.
    """
    if isinstance(random_state, numbers.Integral) and not isinstance(random_state, bool):
        seed = int(random_state)
    else:
        seed = int(check_random_state(random_state).randint(_MAX_SEED))
    return torch.Generator().manual_seed(seed)


def init_linear_(weight, bias, generator: torch.Generator) -> None:
    """Initialize a linear layer's parameters in place as ``torch.nn.Linear`` does.

    Draws the weights, then the bias, from the same uniform distributions and in the same order
    as ``torch.nn.Linear.reset_parameters``, but from ``generator`` instead of PyTorch's global
    random state.
    """
    fan_in = weight.shape[1]
    # kaiming_uniform_(weight, a=sqrt(5)), with PyTorch's formula for the bound
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    bound = math.sqrt(3.0) * (gain / math.sqrt(fan_in))
    with torch.no_grad():
        weight.uniform_(-bound, bound, generator=generator)
        bias_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bias.uniform_(-bias_bound, bias_bound, generator=generator)


def check_positive_int(name: str, value) -> int:
    """Return ``value`` as an int, or raise ``ValueError`` unless it is a positive int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}.")
    return int(value)


def check_positive_float(name: str, value) -> float:
    """Return ``value`` as a float, or raise ``ValueError`` unless it is a positive number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    return float(value)


def as_float32_tensor(X, device=None) -> torch.Tensor:
    """Convert an array to a float32 tensor, as the original code's ``.float()`` did."""
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
