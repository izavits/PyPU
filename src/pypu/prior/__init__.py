r"""Estimate the class prior from PU data.

Many PU methods need the class prior :math:`\pi = P(y = 1)`. When it is unknown, estimate it
from the PU training data and pass the estimate on::

    from pypu.models import IFPU
    from pypu.prior import BBEPrior

    prior = BBEPrior().fit(X_train, s_train).prior_
    clf = IFPU(prior=prior).fit(X_train, s_train)

- :class:`BBEPrior`: Best Bin Estimation, robust when the classes overlap.
- :class:`ElkanNotoPrior`: the classic estimator of Elkan and Noto.
- :class:`KMPrior`: kernel mean embedding (KM1 and KM2) of Ramaswamy, Scott and Tewari, on
  the features directly; needs the optional cvxopt.

All assume SCAR labels and also estimate the label frequency and the share of positives
among the unlabeled samples.
"""

from pypu.prior._base import BasePriorEstimator
from pypu.prior._estimators import BBEPrior, ElkanNotoPrior, KMPrior

__all__ = ["BBEPrior", "BasePriorEstimator", "ElkanNotoPrior", "KMPrior"]
