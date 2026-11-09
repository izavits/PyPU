# Contributing to PyPU

Thanks for your interest in PyPU. Bug reports, new methods, fixes and documentation
improvements are all welcome.

## Reporting bugs and requesting features

Open an issue on GitHub. For a bug, include a minimal example that reproduces it and the
output of:

```bash
python -c "import pypu; pypu.show_versions()"
```

## Development setup

You need Python 3.11 or newer and git.

```bash
git clone https://github.com/izavits/PyPU.git
cd PyPU
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip   # pip >= 25.1 is needed for --group
pip install -e . --group dev
pre-commit install
```

With [uv](https://docs.astral.sh/uv/), `uv sync` creates `.venv` and installs the `dev` group
in one step.

The version number is computed from git tags by hatch-vcs. An editable install writes it to
`src/pypu/_version.py`, which is not committed.

## Running the checks

```bash
pytest                       # tests, the examples included (they need matplotlib)
pytest tests/test_examples.py   # only the examples
pytest --cov                 # tests with a coverage report
ruff check .                 # lint
ruff format .                # format
pre-commit run --all-files   # everything the git hook runs
```

CI runs the linter and the tests on Linux, macOS and Windows for Python 3.11–3.14, runs the
examples and builds the documentation on Linux, and checks that the package builds.

## Building the documentation

The documentation site, [pypu.readthedocs.io](https://pypu.readthedocs.io), is built with
Sphinx from `docs/`. The `dev` group installs the documentation tools and every optional
library, so that all the examples run:

```bash
sphinx-build -b html docs docs/_build/html                     # runs every example (~2 min)
sphinx-build -b html -D plot_gallery=0 docs docs/_build/html   # quick: skips the examples
sphinx-autobuild -D plot_gallery=0 docs docs/_build/html       # live preview while editing
```

Then open `docs/_build/html/index.html` (or http://127.0.0.1:8000 for the live preview).
CI and Read the Docs treat warnings as errors, so check with `-W --keep-going` before
pushing.

- The pages are MyST Markdown. The user guide's code is in ```` ```pycon ```` blocks
  written as doctests: `tests/test_user_guide.py` runs each page and checks the output, so
  print rounded values or Python scalars, as in the docstrings. A page whose code needs an
  optional module starts with a comment such as `% test-requires: cvxopt`.
- `docs/api/index.rst` lists the public API; a test checks that it names every public
  object and nothing else.
- The docstrings are rendered by numpydoc. In a type, quote a name that ends with an
  underscore, as in ``ndarray of shape (`n_features_in_`,)``: otherwise reStructuredText
  reads it as a link.
- The examples gallery is built from `examples/`; its order follows `examples/README.rst`.

Read the Docs builds `latest` from every push to `main`, and a version of the
documentation for every release tag.

## Project layout

```
src/pypu/
  base.py       BasePUClassifier: the interface every model implements
  models/       one module per method, registered in models/__init__.py
  datasets/     PU data from labeled datasets
  metrics/      evaluation with true labels, and estimates from PU labels
  prior/        class-prior estimation from PU data
  utils/        validation and shared helpers
tests/          pytest suite; test_model_contract.py runs on every registered model
examples/       sphinx-gallery scripts: getting_started/, models/ (one per model), comparison/
docs/           the documentation site: user guide, API reference, examples gallery
```

`pytest` also runs the examples in the docstrings, so keep their output exact. Print lists
(`.tolist()`) or Python scalars (`int(...)`, `float(...)`) rather than NumPy arrays and
scalars, whose repr differs between NumPy versions and platforms.

## Adding a model

1. Create `src/pypu/models/<name>.py` with a class that derives from
   `pypu.base.BasePUClassifier`:
   - `__init__` only stores its keyword arguments, unchanged, as attributes with the same
     names. No validation or other logic: scikit-learn's `clone` and `get_params` rely on it.
   - Implement `_fit(X, s)` and `_predict_proba(X)`. `X` and `s` arrive already validated.
     Attributes learned during fitting end with an underscore (`coef_`, `estimator_`).
   - Override `_decision_function` if the model has raw scores (such as boosting margins),
     and `_compute_threshold` if `predict` should not threshold the probability at 0.5.
   - Take a `random_state` parameter if the model is random, and a `prior` parameter if it
     needs the class prior.
   - Import optional dependencies (PyTorch, XGBoost, ...) inside the model's module only,
     never in `pypu/__init__.py` or `pypu/models/__init__.py`, and raise an `ImportError`
     that says which extra to install.
   - Write a NumPy-style docstring with Parameters, Attributes, References and Examples.
2. Register the model in `src/pypu/models/__init__.py`: in `_MODELS`, in `__all__` and in the
   `TYPE_CHECKING` block (a test checks that the three agree).
3. Add tests in `tests/models/test_<name>.py`. The shared contract tests in
   `tests/test_model_contract.py` pick the model up automatically. When porting an existing
   implementation, add a regression test that compares against outputs saved from the
   original code.
4. Add an example, `examples/models/plot_<name>.py`, in the style of the others: data, fit
   against a naive baseline, what the model learned, a figure. List it in
   `examples/README.rst` (a test checks that every script is listed), and add it to
   `REQUIRES` in `tests/test_examples.py` if it needs an optional dependency.
5. Add the model to the API reference, in its family's section of `docs/api/index.rst` (a
   test checks that every model is listed), and to the user guide's
   `docs/user_guide/models.md`: the table of all the models, and the lists of which models
   need the prior.
6. Add the model to the table in `README.md`, and to
   `examples/comparison/plot_compare_methods.py`, and an entry under `## [Unreleased]` in
   `CHANGELOG.md`.

### Porting code from other projects

Only port code whose license is compatible with BSD-3-Clause (for example MIT, BSD or
Apache-2.0). Keep the original copyright notice in the file header, and list the source and
its license in `THIRD_PARTY_NOTICES.md`. If the license is unknown or incompatible,
reimplement the method from the paper instead.

## Pull requests

- Branch from `main` and keep each pull request to one change.
- Add or update tests, and add a changelog entry.
- Make sure `pre-commit run --all-files` and `pytest` pass.

## Versioning and releases

PyPU follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`. Before 1.0.0, a
minor version may change the public API; the changelog says so when it does.

Version numbers come from git tags, so there is no version string to edit. To release:

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a new empty
   `## [Unreleased]` section above it, and update the links at the bottom of the file.
2. Commit the changelog and push to `main`.
3. Tag the release and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```

The `Release` workflow then checks that the changelog has a section for the version, builds
the sdist and the wheel, publishes them to TestPyPI and then to PyPI, and creates a GitHub
release whose notes are that changelog section. Read the Docs builds the documentation of
the tag, and its `stable` version moves to the new release.

Pre-release tags (`vX.Y.ZrcN`, `vX.Y.ZaN`, `vX.Y.ZbN`, `vX.Y.Z.devN`) are published to
TestPyPI only, so you can try a release without touching PyPI. To install one, take the
dependencies from PyPI and only PyPU from TestPyPI (anyone can upload to TestPyPI, so do not let
it supply other packages):

```bash
pip install numpy scipy scikit-learn
pip install --no-deps --index-url https://test.pypi.org/simple/ "pypu-learn==X.Y.ZrcN"
```

### One-time publishing setup

Publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is stored in GitHub.

1. On [pypi.org](https://pypi.org/manage/account/publishing/) and
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a pending publisher
   with project name `pypu-learn`, owner `izavits`, repository `PyPU`, workflow `release.yml`,
   and environment `pypi` (on PyPI) or `testpypi` (on TestPyPI).
2. In the GitHub repository settings, create the environments `pypi` and `testpypi`.
   Optionally make yourself a required reviewer on `pypi`, so every release waits for your
   approval before it reaches PyPI.
