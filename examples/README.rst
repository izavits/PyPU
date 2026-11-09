Examples
========

Short, runnable examples of PyPU. Each one makes PU data from a labeled dataset, so that the
results can be checked against the true labels, fits a model, and plots what it learned.

They are Python scripts divided into cells by ``# %%`` comments: run one with
``python examples/getting_started/plot_quickstart.py``, or open it as a notebook in VS Code,
PyCharm or Jupyter (with `jupytext <https://jupytext.readthedocs.io>`_). They need
matplotlib, and some models need an optional extra::

    pip install pypu-learn matplotlib
    pip install "pypu-learn[xgboost,torch,cvxopt]"  # for every example

The scripts follow the `sphinx-gallery <https://sphinx-gallery.github.io>`_ format.

Getting started
---------------

- ``getting_started/plot_quickstart.py``: PU data, the naive baseline and a first PU method.
- ``getting_started/plot_labeling_mechanisms.py``: which positives get labeled (SCAR and
  SAR) and why it matters.
- ``getting_started/plot_prior_estimation.py``: estimating the class prior.
- ``getting_started/plot_model_selection.py``: tuning a model from PU labels alone.

Models
------

One example per model in ``models/``:

- The baseline and corrections of the naive classifier: ``plot_naive_pu.py``,
  ``plot_elkan_noto.py``, ``plot_rebalanced_pu.py``.
- Reliable negatives and pruning: ``plot_spy_pu.py``, ``plot_pebl.py``,
  ``plot_rank_pruning.py``, ``plot_ifpnu.py``.
- Ensembles and trees: ``plot_pu_bagging.py``, ``plot_pu_hellinger_tree.py``,
  ``plot_pu_hellinger_forest.py``.
- SVMs: ``plot_biased_svm.py``.
- PU risks and losses: ``plot_upu_boost.py``, ``plot_nnpu_boost.py``,
  ``plot_imbalanced_nnpu_boost.py`` (XGBoost), ``plot_ifpu.py``, ``plot_almpu.py``
  (PyTorch).
- Labels not chosen at random (SAR): ``plot_sarem.py``, ``plot_propensity_weighting.py``,
  ``plot_lbe.py``, ``plot_pue.py`` (PyTorch).

Comparison
----------

- ``comparison/plot_compare_methods.py``: every model on the breast cancer dataset.
