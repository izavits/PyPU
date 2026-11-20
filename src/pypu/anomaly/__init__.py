"""Unsupervised anomaly detectors, as used in the first step of IFPNU.

:class:`~pypu.models.IFPNU` takes the samples an anomaly detector scores as most anomalous
as possible positives. Besides scikit-learn's and PyOD's detectors, it can use these ports of
the detectors of its original experiments, which no maintained package provides:

- :class:`ExtendedIsolationForest`: isolation trees whose splits have random orientations,
  ``IFPNU(detector="EIF")``.
- :class:`OptIForest`: an isolation forest of locality-sensitive hash trees, regrouped by
  clustering, ``IFPNU(detector="OIF")``.

They follow scikit-learn's outlier detectors: ``fit(X)``, then ``score_samples(X)``, the
lower the more abnormal::

    from pypu.anomaly import ExtendedIsolationForest
    from pypu.models import IFPNU

    clf = IFPNU(prior=0.2, detector=ExtendedIsolationForest(extension_level=3))
"""

from pypu.anomaly._eif import ExtendedIsolationForest
from pypu.anomaly._optiforest import OptIForest

__all__ = ["ExtendedIsolationForest", "OptIForest"]
