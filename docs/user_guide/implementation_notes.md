# Implementation notes

PyPU's models come from two sources:

- **Ported from the code of published experiments.** The defaults are the settings of
  those experiments, and the test suite checks the port against outputs saved from the
  original code; for most models it reproduces them exactly. Where that code differs from
  the paper it implements, PyPU follows the code, so that the experiments' results can be
  reproduced, and documents the difference.
- **Implemented from the papers**, where there was no code to port, or none with a
  compatible license.

This page lists, model by model, what this means in practice. The docstrings give the
details.

## Ported models

### IFPU

- The defaults are those of the iFPU experiments: 10 trees, learning rate 0.1, focal
  `gamma=3`. (The experiments' scripts asked for 100 trees, but also passed a parameter
  that LightGBM read as 10 boosting rounds, so their results come from 10 trees.)
- Unlabeled samples count as positives with probability `prior` and as negatives
  otherwise. A clipping step left commented out in the original code is not ported: it
  would have removed the negative-class term of every unlabeled sample.
- With the experiments' labels and splits, it reproduces their saved test scores exactly.
- The scores rank well but are not calibrated probabilities.

### IFPNU

- Both steps are ported as run, with the experiments' defaults: 100 trees, `gamma=3`,
  `eta=0.5`.
- The original objective applies a non-negative correction that never triggers, so the
  second step is a weighted focal loss over the labeled positives, the reliable negatives
  and the remaining unlabeled samples.
- The experiments used the DIF, EIF and OptIForest anomaly detectors, which PyPU does not
  include. It uses scikit-learn's {class}`~sklearn.ensemble.IsolationForest` by default,
  and accepts any PyOD detector or scikit-learn outlier detector instead. With the same
  seeded detector, it gives the same labels and scores as the original code.
- It assumes the positives look anomalous. When they do not, the detector misses them and
  hidden positives become reliable negatives. `eta` sets how much the reliable negatives
  weigh against the remaining unlabeled samples.

### PUBagging

- It builds the same bags as the earlier `BaggingPuClassifier`, from the same seeds.
- The defaults are 100 bags, each as large as the labeled set, as in the experiments that
  used it, with decision trees as the base classifier, as in the original class.
- `oob_proba_` uses the bags the model was trained on; the original redrew them.

### PUHellingerTree and PUHellingerForest

- A NumPy rewrite of the original Cython code, which builds only on old versions of
  Python and scikit-learn. On the reference data the trees are identical, node for node.
- `predict_proba` is the fraction of labeled samples in each leaf, as the experiments
  used: it ranks samples but estimates $P(s = 1 \mid x)$, so its values stay low.
  `predict_pu_proba` gives the original code's corrected probability.
- A tree grown to full depth overfits; limit `max_depth`. The forest's `max_samples`
  defaults to the number of unlabeled samples, the experiments' setting.

### UPUBoost, NNPUBoost and ImbalancedNNPUBoost

- They port the XGBoost objectives of the experiments, with their settings: 100 rounds,
  learning rate 0.3, depth 6, no L2 regularization.
- The unbiased PU risk of `UPUBoost` is unbounded below: flexible models push the scores
  of labeled samples to large values and overfit.
- The nnPU and imbalanced nnPU objectives of the experiments apply the non-negative
  correction sample by sample. Since the corrected term is always negative for labeled
  samples, it is always removed, and the objectives become weighted cross-entropies. They
  are not the published nnPU (Kiryo et al.) or imbalanced nnPU (Su et al.), which clip the
  total negative risk.
- In the original environment the trees are identical to the original's. Other XGBoost
  versions and processors change the scores slightly; the ranking barely moves.

### SAREM

- It ports SAR-PU's EM and matches the original to about $10^{-8}$.
- With `refit=True`, the default as in the original, the classifier is retrained with
  propensity weights after EM. That objective is not convex, and in the examples the
  refitted classifier still ranks well, but its probabilities are too flat and `predict`
  labels too few positives. Try `refit=False`, or threshold at the prior, when the labels
  matter.
- `propensity_features` and `predict_propensity` are new. Restrict the propensity to the
  features that drive labeling: with all features, the propensity is much harder to
  recover.
- EM's estimate of a constant propensity is too high when the classes overlap; to estimate
  the label frequency, prefer {mod}`pypu.prior`.

### LBE

- The defaults are what the original code ran, whatever its wrapper asked for: logistic
  models, 1000 pre-training steps, then 100 EM iterations of 100 Adam steps.
- As in that code, the model of the labeling bias is fitted as a classifier of labeled
  versus unlabeled samples, $P(s = 1 \mid x)$, rather than as the propensity, and the
  E-step overestimates the chance that an unlabeled sample is positive.
- Under the same PyTorch it gives the original's weights to the last bit. Standardize the
  features: the models are linear and trained with a fixed learning rate.

### PUe

- The defaults are the experiments' settings, including only 20 Adam steps per model at
  learning rate 0.001. They barely train it: its scores stay close to those of its random
  initial weights, and its ROC AUC varies widely with `random_state`. For a trained model,
  use `n_steps=2000, learning_rate=0.01`.
- As in the original, the loss sums over the unlabeled samples while the labeled weights
  sum to one, so the unlabeled samples weigh far more than in a risk estimate. A trained
  model may then predict hardly any positives: threshold at the prior.
- Without `prior`, it estimates the prior with KM1 ({class}`~pypu.prior.KMPrior`), which
  needs cvxopt.
- With the experiments' labels, splits and seeds, it regenerates their saved test scores.

### ALMPU

- It ports the code of the tabular benchmark, with its settings: a small MLP, 20 epochs of
  the constrained phase, then 30 towards the pseudo-labels.
- Under the same PyTorch it trains the same network to the last bit. Training with SGD
  amplifies rounding differences, so other PyTorch versions and platforms can give
  noticeably different networks.
- The pseudo-labels can go wrong. In 159 of the benchmark's 840 runs, mostly on its
  smallest datasets, most unlabeled samples were pseudo-labeled positive and the test
  ROC AUC fell below 0.5. Check `pollution_`, the share of unlabeled samples
  pseudo-labeled positive, against the share you expect.

### RankPruning

- It is the rankpruning package's PU learning mode, with the package's defaults (three
  folds), and gives the original's results to the last bit. A given `prior` sets the share
  of positives left unlabeled, which the package otherwise estimates.

## Models implemented from the papers

- **ElkanNoto** estimates the label frequency on out-of-fold scores. Its weighted form
  trains a classifier on weighted copies of the unlabeled samples, as in the paper; an
  earlier implementation named after it scaled the scores instead.
- **RebalancedPU** uses the weights of Bekker and Davis's survey.
- **PropensityWeightedPU** gives labeled samples a negative weight as negatives, which
  makes the objective non-convex: results depend on the solver and its tolerance, and the
  probabilities can come out flat. A regularized linear model is the safe choice; rank, or
  threshold at the prior.
- **BiasedSVM** chooses the positive class weight from `(0.7, 0.75, ..., 0.95)` by Lee and
  Liu's criterion.
- **SpyPU**'s optional EM refinement (`n_em_iter`) follows S-EM; S-EM's model selection is
  not included.
- **PEBL** takes its reliable negatives from spies by default (Spy-SVM). The original 1-DNF
  (`negatives="1dnf"`) suits binary or count features, such as word counts; on ordinary
  continuous features it finds no reliable negatives.

## Data and class-prior estimation

- {func}`~pypu.datasets.label_sar_knn`: the original experiments averaged the distance to
  the first negatives in row order rather than the nearest ones. PyPU uses the nearest;
  `legacy=True` reproduces the original labels, and `label_scar(..., legacy=True)` the
  original SCAR labels.
- {class}`~pypu.prior.BBEPrior` and {class}`~pypu.prior.ElkanNotoPrior` are implemented
  from the papers. {class}`~pypu.prior.KMPrior` is the authors' code, unchanged; its solver,
  CVXOPT, is licensed under the GPL-3.0 and is an optional extra. Its estimates are
  multiples of its bisection's resolution, so nearby datasets often get exactly the same
  estimate.

## Reproducibility

- Models that take a `random_state` give the same results for the same value, on the same
  versions of their dependencies.
- The PyTorch models run on one thread, so their results do not depend on the number of
  cores, and compute their predictions in float64.
- [Installation](../install.md#platform-notes) explains how PyPU keeps PyTorch and
  LightGBM working together on macOS.
