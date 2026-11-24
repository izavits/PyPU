# Third-party notices

PyPU is distributed under the BSD 3-Clause License (see `LICENSE`). Parts of it are derived
from the projects below, credited here with their license texts where they have one.

## ALM-PU

- **Used in:** `src/pypu/models/almpu.py` (`ALMPU`), ported from the PyPU author's tabular
  benchmark of ALM-PU, which adapts the authors' code: the network architecture, the
  augmented-Lagrangian penalty and multiplier updates, the trend of the predictions, the
  Jenks pseudo-labels and the schedule of the second phase's targets.
- **Source:** https://github.com/JiazheWei/ALM-PU, by Jiazhe Wei and colleagues, the code of
  "ALM-PU: positive and unlabeled learning with constrained optimization" (Machine Learning,
  2025).
- **License:** none is stated with the code.

## eif (Extended Isolation Forest)

- **Used in:** `src/pypu/anomaly/_eif.py` (`ExtendedIsolationForest`, and `IFPNU`'s
  `detector="EIF"`): a NumPy port of the trees, splits and scores of the package's C++
  `iForest`, drawing its random numbers in the order of the package's `eif_old.py`.
- **Source:** eif 2.0.2, https://github.com/sahandha/eif, by Matias Carrasco Kind and Sahand
  Hariri, the code of "Extended isolation forest" (S. Hariri, M. Carrasco Kind and R. J.
  Brunner, IEEE Transactions on Knowledge and Data Engineering, 2021).
- **License:** University of Illinois/NCSA Open Source License.

```
Copyright (c) 2018 University of Illinois at Urbana-Champaign
All rights reserved.

Developed by:     Matias Carrasco Kind & Sahand Hariri
                  NCSA/UIUC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
Neither the names of Matias Carrasco Kind, Sahand Hariri, University of Illinois at Urbana-Champaign and NCSA, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
```

## FixMatch-pytorch

- **Used in:** `src/pypu/models/almpu.py`: the exponential moving average of the model and
  the cosine learning-rate schedule, which ALM-PU's code took from FixMatch-pytorch.
- **Source:** FixMatch-pytorch, https://github.com/kekmodel/FixMatch-pytorch, by Jungdae Kim
  and Qing Yu.
- **License:** MIT.

```
MIT License

Copyright (c) 2019 Jungdae Kim, Qing Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## pu-learning notebooks

- **Used in:** `src/pypu/models/biased_svm.py`, `two_step.py` and the tests of
  `tests/models/test_classic.py`: the grid of class weights of `BiasedSVM`, the selection of
  `PEBL`'s iteration by Lee and Liu's criterion, and the reference procedures of the tests
  follow the notebooks.
- **Source:** https://github.com/hkiyomaru/pu-learning, by Hirokazu Kiyomaru and Yukiya Wada.
- **License:** MIT.

```
The MIT License (MIT)

Copyright (c) 2021 Hirokazu KIYOMARU

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PU-Bench

- **Used in:** `src/pypu/datasets/_labeling.py` (`label_sar_lbe`, `label_sar_pusb`): the
  draw probabilities of the LBE-A and LBE-B labeling mechanisms and the classifier
  posterior they start from, after `create_pu_training_set` in `data/data_utils.py`. The
  tests (`tests/datasets/test_labeling.py`, `tests/metrics/test_pu_labels.py`) compare
  against copies of PU-Bench's code.
- **Source:** https://github.com/XiXiphus/PU-Bench, the code of "PU-Bench: a unified
  benchmark for rigorous and reproducible PU learning" (Q. Chen, H. Zhang, L. Zhang, C. Li,
  J. Wang and W. Wang, ICLR 2026).
- **License:** MIT.

```
MIT License

Copyright (c) 2026 XiXiphus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PUlearning (nnPUSB)

- **Used in:** `src/pypu/datasets/_labeling.py` (`label_sar_pusb`): the accept-reject rule
  that labels positives, from `BiasedPUlearning/nnPUSB/main_nnPUSB_mnist.py`. The tests
  (`tests/datasets/test_labeling.py`) compare against a copy of it.
- **Source:** https://github.com/MasaKat0/PUlearning, by MasaKat0, with the code of
  "Learning from positive and unlabeled data with a selection bias" (M. Kato, T. Teshima
  and J. Honda, ICLR 2019).
- **License:** MIT.

```
MIT License

Copyright (c) 2018 MasaKat0

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Kernel mixture proportion estimation (KM)

- **Used in:** `src/pypu/prior/_km.py` (`KMPrior`, and `PUe` when no prior is given), kept
  unchanged apart from linting, as modified for the iFPNU experiments.
- **Source:** http://web.eecs.umich.edu/~cscott/code/kernel_MPE.zip, by Harish G. Ramaswamy,
  Clayton Scott and Ambuj Tewari, the code of "Mixture Proportion Estimation via Kernel
  Embedding of Distributions" (ICML 2016).
- **License:** none is stated with the code.
- It calls CVXOPT (GPL-3.0), an optional dependency (`pypu-learn[cvxopt]`) that PyPU does
  not distribute.

## OptIForest

- **Used in:** `src/pypu/anomaly/_optiforest.py` (`OptIForest`, and `IFPNU`'s
  `detector="OIF"`): the sampling, the locality-sensitive hash families, the trees, their
  hierarchical regrouping and the scores, ported from the copy in the iFPNU experiments,
  which is the authors' code unchanged apart from formatting.
- **Source:** https://github.com/xiagll/OptIForest (`detectors/`), by Haolong Xiang and
  colleagues, the code of "OptIForest: Optimal Isolation Forest for Anomaly Detection"
  (IJCAI 2023).
- **License:** none is stated with the code.

## rankpruning

- **Used in:** `src/pypu/models/rank_pruning.py` (`RankPruning`), ported from the
  `RankPruning` class and its helpers in their PU learning mode, keeping their estimates and
  pruning rule.
- **Source:** rankpruning, https://github.com/cgnorthcutt/rankpruning, by Curtis G. Northcutt.
- **License:** MIT.

```
MIT License

Copyright (c) 2017 cgnorthcutt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SAR-PU

- **Used in:** `src/pypu/models/sarem.py` (`SAREM`), ported from `pu_learn_sar_em` and its
  helpers, keeping their sequence of fits and their arithmetic.
- **Source:** SAR-PU, https://github.com/ML-KULeuven/SAR-PU, by Jessa Bekker, Pieter
  Robberechts and Jesse Davis.
- **License:** MIT.

```
MIT License

Copyright (c) 2024 KU Leuven - DTAI - Machine Learning Research Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## scikit-learn

Two parts of PyPU follow scikit-learn code (https://github.com/scikit-learn/scikit-learn):

- **`BaggingClassifier`, adapted for PU learning by Roy Wright.** Used in
  `src/pypu/models/bagging.py` (`PUBagging`): the way bags are drawn, including the order of
  the random draws, follows Roy Wright's `BaggingPuClassifier`, an adaptation of Gilles
  Louppe's `BaggingClassifier`, so that both build the same bags.
- **Decision trees (scikit-learn 1.0), as forked by the PU Hellinger trees code.** Used in
  `src/pypu/models/_hellinger_tree.py` and `src/pypu/models/hellinger.py`
  (`PUHellingerTree`, `PUHellingerForest`): a NumPy rewrite of the depth-first tree builder,
  the best splitter and its random number generator (`sklearn/tree/_tree.pyx`,
  `_splitter.pyx`, `_utils.pyx`), and of the forest's seeding and bootstrap, so that the
  trees are those of the original code.
- **License:** BSD 3-Clause.

```
Copyright (c) The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
