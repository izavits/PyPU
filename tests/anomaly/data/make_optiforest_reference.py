"""Generate ``optiforest_reference.npz``: scores of the original OptIForest, for the tests.

Run it with the Python environment of the original iFPNU experiments, passing the iFPNU
directory of the research code, whose ``models/optiforest`` is the authors' code::

    /path/to/pu_monorepo/iFPNU/venv/bin/python tests/anomaly/data/make_optiforest_reference.py \
        /path/to/pu_monorepo/iFPNU

It saves a synthetic dataset and, for each case of ``CASES``, the original's
``decision_function`` on it, from a NumPy seed. The first case is the experiments' detector,
``OptIForest('L2OPT', 100, 403, 20, 1)``; the others use smaller trees (the ``max_samples``
of PyPU) to exercise the regrouping's merges. With ``branch=0`` the original draws from
Python's ``random``; here those draws come from NumPy, so that every random number comes from
the one seeded generator. Nothing runs on import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> (seed, PyPU's OptIForest parameters).
SMALL = {"n_estimators": 10, "max_samples": 128, "threshold": 20}
CASES = {
    "experiments": (0, {}),
    "l1_pairs": (1, {**SMALL, "lsh_family": "L1OPT", "branch": 2}),
    "angle_random_merges": (2, {**SMALL, "lsh_family": "ALOPT", "branch": 0}),
    "triples_granularity0": (3, {**SMALL, "branch": 4, "granularity": 0}),
}
DEFAULTS = {
    "n_estimators": 100,
    "max_samples": 512,
    "lsh_family": "L2OPT",
    "threshold": 403,
    "branch": 20,
    "granularity": 1,
}


def main(ifpnu_dir):
    sys.path.insert(0, str(Path(ifpnu_dir) / "models"))
    import numpy as np
    from optiforest import opt_tree
    from optiforest.opt import E2LSH, AngleLSH
    from optiforest.opt_forest import OptForest
    from optiforest.opt_iforest import OptIForest
    from optiforest.sampling import VSSampling
    from scipy.spatial import distance

    families = {
        "L2OPT": (lambda: E2LSH(norm=2), distance.euclidean),
        "L1OPT": (lambda: E2LSH(norm=1), distance.cityblock),
        "ALOPT": (AngleLSH, distance.cosine),
    }

    class NumPyRandom:
        """Stands in for opt_tree's ``random`` module: its draws come from NumPy."""

        @staticmethod
        def random():
            return np.random.random()  # noqa: NPY002 - the original's global generator

    opt_tree.random = NumPyRandom

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    X[:5] += 4.0  # a few clear outliers
    arrays = {"X": X}
    for case, (seed, params) in CASES.items():
        p = {**DEFAULTS, **params}
        np.random.seed(seed)  # noqa: NPY002 - the original draws from NumPy's global generator
        if case == "experiments":
            detector = OptIForest("L2OPT", 100, 403, 20, 1)  # as main_2step_ad.py
        else:
            lsh, dist = families[p["lsh_family"]]
            sampler = VSSampling(p["n_estimators"], p["max_samples"], p["max_samples"])
            detector = OptForest(
                p["n_estimators"], sampler, lsh(), p["threshold"], p["branch"], dist,
                p["granularity"],
            )  # fmt: skip
        detector.fit(X)
        arrays[f"{case}_scores"] = detector.decision_function(X)
    np.savez_compressed(Path(__file__).with_name("optiforest_reference.npz"), **arrays)


if __name__ == "__main__":
    main(sys.argv[1])
