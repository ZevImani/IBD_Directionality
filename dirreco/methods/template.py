"""Template-matching (scan) methods.

Each ScanMethod defines a loss L(phi) between the measurement rotated CCW by phi and the
reference sample (at theta = 0). The estimate is theta_hat = -phi*, where phi* is the
refined minimum of L over the scan grid.

Common options (all scan methods):
  scan:   [lo, hi, step] in degrees, inclusive. [-180, 179, 1] = full circle (circular
          refinement); [-15, 15, 1] = Max's clipped window.
  refine: argmin    -- grid point with the smallest loss
          parabolic -- 3-point parabolic vertex around the argmin ("Paper C")
          sine      -- global fit L = a|sin((phi - phi0)/2)| + c (Gabe's CFND)
"""
import numpy as np
from scipy.optimize import curve_fit

from . import Method, register
from ..data import wrap

CHUNK = 4_000_000  # max (angles x events) values held in memory at once


def _abs_sine(phi, a, phi0, c):
    return a * np.abs(np.sin(np.radians(phi - phi0) / 2.0)) + c


def _rotations(x, y, phis):
    """(len(phis), N) arrays of x, y rotated CCW by each phi."""
    t = np.radians(phis)[:, None]
    c, s = np.cos(t), np.sin(t)
    return c * x[None, :] - s * y[None, :], s * x[None, :] + c * y[None, :]


def _chunks(phis, n_events):
    step = max(1, CHUNK // max(1, n_events))
    for i in range(0, len(phis), step):
        yield phis[i:i + step]


class ScanMethod(Method):
    """zoom: [half_width, step] (deg) turns the scan into a two-stage search: the `scan` grid
    (e.g. the full circle at 5 deg) locates the minimum, then a fine grid of +-half_width around
    it is evaluated and refined (re-centred, up to 3 times, if its minimum sits on an edge).
    Unbiased like a full 1-deg circle scan, at a fraction of the loss evaluations."""

    def __init__(self, scan=(-180, 179, 1), refine="parabolic", zoom=None, **common):
        super().__init__(**common)
        lo, hi, step = scan
        self.phis = np.arange(lo, hi + step / 2.0, step, dtype=float)
        self.step = float(step)
        self.circular = abs(len(self.phis) * step - 360.0) < 1e-9
        if refine not in ("argmin", "parabolic", "sine"):
            raise ValueError(f"refine must be argmin|parabolic|sine, got {refine}")
        self.refine = refine
        self.zoom = None
        if zoom is not None:
            if refine == "sine":
                raise ValueError("zoom works with refine: argmin | parabolic")
            self.zoom = (float(zoom[0]), float(zoom[1]))

    # --- to implement -------------------------------------------------------
    def loss(self, meas, ref, geom):
        """Return L(phi) for phi in self.phis."""
        raise NotImplementedError

    def loss_options(self):
        """Options that change loss() (not refine). Methods with equal keys share losses."""
        return ()

    # --- shared -------------------------------------------------------------
    def loss_key(self, geom):
        return (type(self).__name__, self.phis.tobytes(), self.loss_options(), geom, self.ref_mult)

    def loss_at(self, meas, ref, geom, phis):
        """loss() on an arbitrary angle grid (used by the zoom stage)."""
        saved = self.phis
        self.phis = np.asarray(phis, dtype=float)
        try:
            return self.loss(meas, ref, geom)
        finally:
            self.phis = saved

    def estimate(self, meas, ref, geom, losses=None):
        L = self.loss(meas, ref, geom) if losses is None else losses
        if self.zoom is None:
            return float(wrap(-self.refine_min(L)))
        half, step = self.zoom
        offs = np.arange(-half, half + step / 2.0, step)
        centre = self.phis[int(np.argmin(L))]
        for _ in range(4):
            phis = centre + offs
            Lf = self.loss_at(meas, ref, geom, phis)
            i = int(np.argmin(Lf))
            if 0 < i < len(phis) - 1:
                break
            centre = phis[i]
        return float(wrap(-self.refine_min(Lf, phis, step, circular=False)))

    def refine_min(self, L, phis=None, step=None, circular=None):
        phis = self.phis if phis is None else phis
        step = self.step if step is None else step
        circular = self.circular if circular is None else circular
        n = len(L)
        i = int(np.argmin(L))
        if self.refine == "argmin":
            return phis[i]
        if circular:
            im, ip = (i - 1) % n, (i + 1) % n
        elif 0 < i < n - 1:
            im, ip = i - 1, i + 1
        else:
            im = ip = None
        phi = phis[i]
        if im is not None:
            dp, dz, dm = L[ip], L[i], L[im]
            denom = dp - 2.0 * dz + dm
            phi = phis[i] + (-0.5 * (dp - dm) / denom if denom > 0 else 0.0) * step
        if self.refine == "parabolic":
            return phi
        try:
            popt, _ = curve_fit(_abs_sine, phis, L, p0=[np.ptp(L), phis[i], np.min(L)], maxfev=1500)
            return popt[1]
        except Exception:
            return phi


@register("fnd")
class FND(ScanMethod):
    """Frobenius norm of the difference of 2D count histograms on the detector grid
    (continuous displacements, re-binned after each rotation)."""

    def loss(self, meas, ref, geom):
        G2 = geom.grid_size ** 2
        r = ref.disp
        H_ref = geom.hist(r[:, 0], r[:, 1]).ravel()
        x, y = meas.disp[:, 0], meas.disp[:, 1]
        out = []
        for ph in _chunks(self.phis, max(len(x), G2)):
            rx, ry = _rotations(x, y, ph)
            b = geom.bin_index(rx, ry)
            b = np.where(b >= 0, b + (np.arange(len(ph)) * G2)[:, None], -1)
            H = np.bincount(b[b >= 0], minlength=len(ph) * G2).reshape(len(ph), G2)
            out.append(np.sqrt(((H - H_ref[None, :]) ** 2).sum(axis=1)))
        return np.concatenate(out)


@register("image_l2")
class ImageL2(ScanMethod):
    """Pixel-space baseline: L2 between the normalised reference histogram and the
    measurement histogram rotated as an image (scipy.ndimage, bilinear)."""

    def loss(self, meas, ref, geom):
        from scipy.ndimage import rotate
        H_ref = geom.hist(ref.disp[:, 0], ref.disp[:, 1])
        H_ref /= max(H_ref.sum(), 1.0)
        H = geom.hist(meas.disp[:, 0], meas.disp[:, 1])
        H /= max(H.sum(), 1.0)
        # H is indexed [ix, iy]; ndimage's positive angle on axes (0, 1) is CCW in (x, y).
        return np.array([np.linalg.norm(H_ref - rotate(H, phi, axes=(0, 1), reshape=False, order=1))
                         for phi in self.phis])
