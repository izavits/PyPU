"""Generate ``eif_reference.npz``: scores of the original ``eif`` package, for ``test_eif.py``.

Run it with the Python environment of the original iFPNU experiments, which has eif 2.0.2
(its C++ ``eif`` module and its pure-Python ``eif_old`` module)::

    /path/to/pu_monorepo/iFPNU/venv/bin/python tests/anomaly/data/make_eif_reference.py

It saves a synthetic dataset and two kinds of scores of its samples:

- ``eif_old`` from a NumPy seed, for cases of ``EXACT_CASES``. ``eif_old`` draws the samples
  of each tree with Python's ``random.sample``; here they come from NumPy instead
  (``np.random.choice`` without replacement, as PyPU does), so that every random number comes
  from the one seeded NumPy generator.
- The C++ ``eif``, as the experiments ran it (100 trees of 256 samples, extension level 1),
  with the seeds in ``CPP_SEEDS``.

Nothing runs on import, so pytest can collect this file safely.
"""

from pathlib import Path

# Case name -> (seed, eif_old's iForest keyword arguments).
EXACT_CASES = {
    "level1": (0, {"ntrees": 20, "sample_size": 128, "ExtensionLevel": 1}),
    "full": (1, {"ntrees": 20, "sample_size": 300, "ExtensionLevel": 3}),
    "axis_parallel_depth4": (2, {"ntrees": 10, "sample_size": 64, "ExtensionLevel": 0, "limit": 4}),
}
CPP_SEEDS = range(0, 10_000, 1000)  # tree i of seed k uses k + i: no shared trees


def main():
    import random

    import eif
    import eif_old
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:5] += 4.0  # a few clear outliers
    arrays = {"X": X}

    class NumPySampling:
        """Stands in for eif_old's ``random`` module: the samples of each tree from NumPy."""

        @staticmethod
        def sample(population, k):
            return np.random.choice(len(population), k, replace=False)  # noqa: NPY002 - as PyPU

    eif_old.rn = NumPySampling
    for case, (seed, kwargs) in EXACT_CASES.items():
        np.random.seed(seed)  # noqa: NPY002 - eif_old draws from NumPy's global generator
        forest = eif_old.iForest(X, **kwargs)
        arrays[f"{case}_scores"] = forest.compute_paths(X)
    eif_old.rn = random

    experiments = {"ntrees": 100, "sample_size": 256, "ExtensionLevel": 1}
    arrays["cpp_scores"] = np.array(
        [eif.iForest(X, seed=seed, **experiments).compute_paths(X) for seed in CPP_SEEDS]
    )
    np.savez_compressed(Path(__file__).with_name("eif_reference.npz"), **arrays)


if __name__ == "__main__":
    main()
