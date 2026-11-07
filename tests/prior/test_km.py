import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

pytest.importorskip("cvxopt")

from pypu.prior import KMPrior

REFERENCE = Path(__file__).parents[1] / "models" / "data" / "pue_reference.npz"


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case", ["scar", "sar"])
def test_reproduces_the_original(reference, case):
    # The original code's (KM2, KM1) estimates of the fraction of positives among the
    # unlabeled samples; the bisection makes them insensitive to the solver's last bits.
    X, s = reference["X_train"], reference[f"{case}_s"]
    km2, km1 = reference[f"{case}_km"]
    assert KMPrior().fit(X, s).pollution_ == pytest.approx(km1, rel=1e-12)
    assert KMPrior(method="km2").fit(X, s).pollution_ == pytest.approx(km2, rel=1e-12)


def test_estimates_are_consistent(reference):
    X, s = reference["X_train"], reference["scar_s"]
    estimator = KMPrior().fit(X, s)
    assert 0 < estimator.pollution_ < 1
    assert estimator.prior_ * estimator.label_frequency_ == pytest.approx(s.mean())
    n_unlabeled = (s == 0).sum()
    assert estimator.pollution_ == pytest.approx(
        (estimator.prior_ * s.size - s.sum()) / n_unlabeled
    )


def test_estimate_is_close_to_the_truth(reference):
    X, s, y = reference["X_train"], reference["scar_s"], reference["y_train"]
    assert KMPrior().fit(X, s).prior_ == pytest.approx(y.mean(), abs=0.05)


def test_accepts_dataframes(reference):
    X, s = reference["X_train"], reference["scar_s"]
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    estimator = KMPrior().fit(frame, s)
    np.testing.assert_array_equal(estimator.feature_names_in_, frame.columns)


def test_unknown_method(reference):
    with pytest.raises(ValueError, match="method must be 'km1' or 'km2'"):
        KMPrior(method="km3").fit(reference["X_train"], reference["scar_s"])


def test_can_be_cloned():
    estimator = KMPrior(method="km2")
    assert clone(estimator).get_params() == {"method": "km2"}


def test_missing_cvxopt_gives_an_install_hint():
    code = (
        "import sys\n"
        "class HideCvxopt:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'cvxopt':\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, HideCvxopt())\n"
        "import numpy as np\n"
        "from pypu.prior import KMPrior\n"
        "try:\n"
        "    KMPrior().fit(np.eye(4), [1, 0, 0, 1])\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "pip install 'pypu-learn[cvxopt]'" in output
