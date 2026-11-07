# Third-party notices

PyPU is distributed under the BSD 3-Clause License (see `LICENSE`). Parts of it are derived
from the projects below, whose licenses require this notice.

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
