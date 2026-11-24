"""Data layer: paired (prompt vertex, neutron capture) MC events and detector geometry.

Everything downstream works on an `Events` object (positions in mm). The displacement
r = capture - vertex is what the reconstruction methods use; the beam is along +x
(theta = 0) in the raw MC.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

DEFAULT_DATA = {
    "vertices": "dataset/fid_vertices.npy",   # (8,999,997 x 3) prompt e+ vertices [mm]
    "captures": "dataset/fid_captures.npy",   # (9,999,999 x 3) n-capture sites [mm]
    "capture_offset": 3,                      # captures[3:] lines up 1:1 with vertices[:]
}


def wrap(angle_deg):
    """Wrap angle(s) to [-180, 180)."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def rotate_xy(x, y, theta_deg):
    """Counter-clockwise rotation of (x, y) by theta (degrees)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return c * x - s * y, s * x + c * y


class Events:
    """A set of IBD events: prompt vertex `vtx` and capture `cap`, both (n, 3) in mm."""

    def __init__(self, vtx, cap):
        assert vtx.shape == cap.shape
        self.vtx = vtx
        self.cap = cap
        self._disp = None

    def __len__(self):
        return len(self.vtx)

    @property
    def disp(self):
        """(n, 2) transverse displacement capture - vertex [mm]."""
        if self._disp is None:
            self._disp = self.cap[:, :2] - self.vtx[:, :2]
        return self._disp

    def take(self, idx):
        return Events(self.vtx[idx], self.cap[idx])

    def rotated(self, theta_deg):
        """Rotate both points about the z axis (CCW). theta = 0 is a no-op."""
        if theta_deg == 0:
            return self
        out = []
        for p in (self.vtx, self.cap):
            q = p.copy()
            q[:, 0], q[:, 1] = rotate_xy(p[:, 0], p[:, 1], theta_deg)
            out.append(q)
        return Events(*out)


def load_events(vertices=None, captures=None, capture_offset=None, max_events=None):
    """Load the paired MC pool. Paths are relative to the repo root unless absolute."""
    vertices = REPO / (vertices or DEFAULT_DATA["vertices"])
    captures = REPO / (captures or DEFAULT_DATA["captures"])
    off = DEFAULT_DATA["capture_offset"] if capture_offset is None else capture_offset
    v = np.load(vertices)
    c = np.load(captures)
    n = min(len(v), len(c) - off)
    if max_events:
        n = min(n, int(max_events))
    return Events(v[:n], c[off:off + n])


def theta0(events):
    """Centroid asymptotic resolution sigma_perp / r_bar [deg * sqrt(events)].

    Assumes the beam is along +x (unrotated pool). This is the reference the README
    calls Theta_0,cut; it is NOT a true Cramer-Rao bound.
    """
    d = events.disp
    return float(np.degrees(np.std(d[:, 1]) / np.mean(d[:, 0])))


@dataclass(frozen=True)
class Geometry:
    """Square segmented detector face of side `width` with cell pitch `pitch` (mm)."""
    pitch: float
    width: float = 450.0

    @property
    def grid_size(self):
        return int(round(self.width / self.pitch))

    @property
    def edges(self):
        return np.linspace(-self.width / 2.0, self.width / 2.0, self.grid_size + 1)

    @property
    def cut_radius(self):
        """Prompt-cell veto: events with |r| <= pitch/sqrt(2) are unusable."""
        return self.pitch / np.sqrt(2.0)

    def usable_mask(self, events):
        d = events.disp
        return d[:, 0] ** 2 + d[:, 1] ** 2 > self.cut_radius ** 2

    def cell_center(self, xy):
        """Snap positions to the centre of the cell containing them (cell edges at k*pitch)."""
        return np.floor(xy / self.pitch) * self.pitch + self.pitch / 2.0

    def discrete_disp(self, events):
        """(n, 2) displacement between the cells of the capture and of the vertex."""
        return self.cell_center(events.cap[:, :2]) - self.cell_center(events.vtx[:, :2])

    def bin_index(self, x, y):
        """Flat histogram-bin index on the displacement grid, -1 if outside.

        Reproduces np.histogram2d(x, y, bins=[edges, edges]) bin assignment exactly
        (right-closed last bin), but works on arrays of any shape so many rotations
        can be binned in one call.
        """
        e = self.edges
        G = self.grid_size
        ix = np.searchsorted(e, x, side="right")
        iy = np.searchsorted(e, y, side="right")
        ix[x == e[-1]] -= 1
        iy[y == e[-1]] -= 1
        ok = (ix >= 1) & (ix <= G) & (iy >= 1) & (iy <= G)
        return np.where(ok, (ix - 1) * G + (iy - 1), -1)

    def hist(self, x, y):
        """(G, G) count histogram H[ix, iy], identical to np.histogram2d(x, y, [edges, edges])."""
        G = self.grid_size
        b = self.bin_index(np.asarray(x), np.asarray(y))
        return np.bincount(b[b >= 0], minlength=G * G).reshape(G, G).astype(float)
