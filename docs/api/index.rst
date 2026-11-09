API reference
=============

The public API of PyPU, by subpackage. Every model is a scikit-learn estimator derived from
:class:`~pypu.base.BasePUClassifier`, fitted with ``fit(X, s)``: ``s[i] = 1`` marks a
labeled positive and ``s[i] = 0`` an unlabeled sample.

Models
------

.. automodule:: pypu.models

.. currentmodule:: pypu.models

The models are grouped as in the :doc:`examples <../auto_examples/index>`. The ones
marked XGBoost or PyTorch need an optional extra (see :doc:`../install`).

The naive classifier and its corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NaivePU
   ElkanNoto
   RebalancedPU

Reliable negatives and pruning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SpyPU
   PEBL
   RankPruning
   IFPNU

Ensembles and trees
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PUBagging
   PUHellingerTree
   PUHellingerForest

SVMs
~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BiasedSVM

PU risks and losses
~~~~~~~~~~~~~~~~~~~

``UPUBoost``, ``NNPUBoost`` and ``ImbalancedNNPUBoost`` need XGBoost; ``ALMPU`` needs
PyTorch.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   UPUBoost
   NNPUBoost
   ImbalancedNNPUBoost
   IFPU
   ALMPU

Labels not chosen at random (SAR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LBE`` and ``PUe`` need PyTorch.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SAREM
   PropensityWeightedPU
   LBE
   PUe

Base class
~~~~~~~~~~

.. currentmodule:: pypu.base

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BasePUClassifier

PU data
-------

.. automodule:: pypu.datasets

.. currentmodule:: pypu.datasets

Splits
~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   make_pu_split
   make_case_control

.. autosummary::
   :toctree: generated/
   :template: namedtuple.rst

   PUSplit

Labeling mechanisms
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   label_scar
   label_sar_knn
   label_sar_propensity
   label_sar_cluster
   label_sar_confidence

Label frequency and summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pu_summary
   label_frequency_from_pollution
   pollution_from_label_frequency

Evaluation
----------

.. automodule:: pypu.metrics

.. currentmodule:: pypu.metrics

With the true labels
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   f1_at_prior
   max_f1_score
   r_precision
   precision_at_k
   recall_at_k

From PU labels alone
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lee_liu_score
   pu_precision_score
   pu_recall_score
   pu_f1_score
   pu_roc_auc_score

Class-prior estimation
----------------------

.. automodule:: pypu.prior

.. currentmodule:: pypu.prior

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BBEPrior
   ElkanNotoPrior
   KMPrior
   BasePriorEstimator

Utilities
---------

.. automodule:: pypu.utils

.. currentmodule:: pypu.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   check_pu_labels
   check_true_labels
   check_prior
   threshold_from_prior
   show_versions
