"""Event sample -> CNN input image. Shared by training (cnn/train.py) and by the `cnn`
reconstruction method (dirreco/methods/cnn.py), so both see bit-identical inputs.

One image = the 2D histogram of the transverse displacements r = capture - vertex of the N
events in a sample. The grid has cells of side `bin_mm` (default: the detector pitch), is
centred on r = 0, and covers |r_x|, |r_y| <= ~extent_mm. It is S x S with S = 2*ceil(extent/bin) + 1
(5 at 150 mm pitch, 9 at 50 mm, 81 at 5 mm). Events outside the window are dropped (<0.1% at the
200 mm default). Indexing is H[ix, iy] (x along axis 0), like dirreco.data.Geometry.hist.
Pixel values are counts * S^2 / N (density with mean ~1 per pixel), so that images of
different N are on the same scale.

representation:
  continuous -- true displacement, rounded to the grid (what fnd / emd_* see by default)
  discrete   -- vertex and capture first snapped to their detector cells (cell edges at
                k * pitch), i.e. cell hopping: what a segmented detector actually reads out
                (same as centroid discrete: true). Requires bin_mm == pitch.

Rotations are CCW about the z axis, applied to the absolute vertex and capture positions
(dirreco.data.Events.rotated does the same), so the cell phase is handled exactly.
"""
import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ImageSpec:
    pitch: float
    representation: str = "continuous"
    extent_mm: float = 200.0
    bin_mm: float = None

    def __post_init__(self):
        if self.representation not in ("continuous", "discrete"):
            raise ValueError(f"representation must be continuous|discrete, got {self.representation}")
        if self.bin_mm is None:
            object.__setattr__(self, "bin_mm", float(self.pitch))
        if self.representation == "discrete" and abs(self.bin_mm - self.pitch) > 1e-9:
            raise ValueError("discrete representation requires bin_mm == pitch")

    @property
    def half(self):
        return int(math.ceil(self.extent_mm / self.bin_mm - 1e-9))

    @property
    def size(self):
        return 2 * self.half + 1

    def to_dict(self):
        return asdict(self)


def rotate(vtx, cap, theta_deg):
    """Rotate (T, 2) position tensors CCW by per-point angles theta_deg (T,)."""
    t = torch.deg2rad(theta_deg.to(vtx.dtype))
    c, s = torch.cos(t), torch.sin(t)

    def rot(p):
        return torch.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], dim=1)
    return rot(vtx), rot(cap)


def displacement(vtx, cap, spec):
    """(T, 2) displacement [mm]: true (continuous) or between cell centres (discrete)."""
    if spec.representation == "discrete":
        p = spec.pitch
        return (torch.floor(cap / p) - torch.floor(vtx / p)) * p
    return cap - vtx


def histogram(disp, img_id, n_images, spec):
    """Displacements (T, 2) of all images, img_id (T,) -> (n_images, 1, S, S) float32."""
    M, S = spec.half, spec.size
    k = torch.round(disp / spec.bin_mm).long() + M
    ok = ((k >= 0) & (k < S)).all(dim=1)
    flat = (img_id * S + k[:, 0]) * S + k[:, 1]
    H = torch.bincount(flat[ok], minlength=n_images * S * S).view(n_images, 1, S, S).float()
    n = torch.bincount(img_id, minlength=n_images).clamp(min=1).float()
    return H * (S * S / n).view(-1, 1, 1, 1)


def make_images(vtx, cap, img_id, n_images, spec, theta_deg=None):
    """Full pipeline. vtx, cap: (T, 2) float64 positions [mm] of the events of all images;
    img_id: (T,) long image index; theta_deg: optional (n_images,) CCW rotation per image.
    Returns (images (n_images, 1, S, S), displacements (T, 2))."""
    if theta_deg is not None:
        vtx, cap = rotate(vtx, cap, theta_deg[img_id])
    d = displacement(vtx, cap, spec)
    return histogram(d, img_id, n_images, spec), d
