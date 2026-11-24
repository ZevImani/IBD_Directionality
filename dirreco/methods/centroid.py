"""Vector-centroid estimator: direction of the mean displacement."""
import numpy as np

from . import Method, register


@register("centroid")
class Centroid(Method):
    """theta = atan2(<dy>, <dx>).

    discrete=False : true (continuous) displacements -- the unbinned limit.
    discrete=True  : vertex and capture snapped to their cell centres ("cell hopping"),
                     i.e. what a segmented detector actually reads out.
    """
    needs_ref = False

    def __init__(self, discrete=True, **common):
        super().__init__(**common)
        self.discrete = discrete

    def estimate(self, meas, ref, geom):
        d = geom.discrete_disp(meas) if self.discrete else meas.disp
        return float(np.degrees(np.arctan2(np.mean(d[:, 1]), np.mean(d[:, 0]))))
