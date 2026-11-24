"""IBD events for the diffusion model: paired prompt-vertex / neutron-capture MC (emd_paper/dataset).

What is modelled
----------------
One training example is the displacement r = capture - vertex of one event, in units of
`scale_mm`: x = r / scale_mm. The scale is the same for every component and nothing is
subtracted, so the normalisation commutes with rotations (rotating x rotates r). The
direction signal is the +x offset of r (mean 16.8 mm on a width of 44 mm).

`dims: 3` models (r_x, r_y, r_z); `dims: 2` models (r_x, r_y), which is all that dirreco and
cnn use.

The vertex is not modelled. In the MC it is uniform in a 40 m cube and independent of r
(checked on the full pool: |corr| < 0.003; mean r_x within 400 mm of a wall differs from the
bulk by 0.16 mm, and 0.4% of captures are clipped to the wall). sample.py draws the vertex
uniformly and adds the generated r.

Symmetry
--------
Beam along +x, so the physics is symmetric under y -> -y (verified: KS p = 0.43 between r_y
and -r_y, mean r_y = -0.015 +- 0.015 mm). `flip_y: true` flips r_y of each training event
with probability 1/2. Note there is NO y <-> z (axial) symmetry in this MC: mean r_z = +0.90 mm
(61 sigma), so do not symmetrise z.

Index split (identical to cnn/README.md, so no generator ever sees the run.py events)
-------------------------------------------------------------------------------------
  eval        [0, 2,000,000)          reserved for run.py studies and for evaluating samples
  validation  [2,000,000, 2,500,000)
  train       [2,500,000, end)
"""
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

EMD_PAPER = Path(__file__).resolve().parents[3]
DATA = {
    "vertices": EMD_PAPER / "dataset" / "fid_vertices.npy",
    "captures": EMD_PAPER / "dataset" / "fid_captures.npy",
    "capture_offset": 3,                  # captures[3:] lines up 1:1 with vertices[:]
}
SPLITS = {
    "eval": (0, 2_000_000),
    "validation": (2_000_000, 2_500_000),
    "train": (2_500_000, None),
}
BOX_MM = 20_000.0                         # vertices are uniform in [-BOX_MM, BOX_MM]^3


def load_split(split, max_events=None, data=None):
    """(vertices, captures), each (n, 3) float64 in mm, for one split of the paired pool."""
    d = dict(DATA, **(data or {}))
    v = np.load(d["vertices"], mmap_mode="r")
    c = np.load(d["captures"], mmap_mode="r")
    off = int(d["capture_offset"])
    n_pool = min(len(v), len(c) - off)
    lo, hi = SPLITS[split]
    hi = n_pool if hi is None else min(hi, n_pool)
    if max_events:
        hi = min(hi, lo + int(max_events))
    return np.array(v[lo:hi]), np.array(c[off + lo:off + hi])


def load_displacements(split, dims=3, max_events=None, data=None):
    """(n, dims) float64 displacement capture - vertex [mm]."""
    v, c = load_split(split, max_events, data)
    return (c - v)[:, :dims]


class IBDDisplacements(IterableDataset):
    """Yields whole batches {"event": (B, dims) float32} of scaled displacements.

    Batching inside the dataset (one fancy-index per batch) is ~1000x faster than the
    default per-item collate for 6.5M tiny examples. Use with DataLoader(batch_size=None).
    """

    def __init__(self, split, batch_size, dims=3, scale_mm=50.0, flip_y=False, shuffle=None,
                 max_events=None, seed=0, data=None):
        super().__init__()
        r = load_displacements(split, dims, max_events, data)
        self.x = torch.from_numpy(r / scale_mm).float()
        self.split, self.batch_size, self.flip_y = split, int(batch_size), flip_y
        self.shuffle = (split == "train") if shuffle is None else shuffle
        self.seed, self.epoch = seed, 0

    @property
    def n_events(self):
        return len(self.x)

    def __len__(self):                     # number of batches; the last partial batch is kept
        return -(-self._shard_size() // self.batch_size)

    def _shard(self):
        """(rank, world) under DDP so each GPU sees a disjoint part of every epoch."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _shard_size(self):
        rank, world = self._shard()
        return len(range(rank, self.n_events, world))

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        self.epoch += 1
        order = torch.randperm(self.n_events, generator=g) if self.shuffle else torch.arange(self.n_events)
        rank, world = self._shard()
        order = order[rank::world]
        for i in range(0, len(order), self.batch_size):
            x = self.x[order[i:i + self.batch_size]]
            if self.flip_y:
                sign = torch.randint(0, 2, (len(x),), generator=g).float() * 2 - 1
                x = x.clone()
                x[:, 1] *= sign
            yield {"event": x}


class IBDDataModule(pl.LightningDataModule):
    """Train/validation loaders for main.py (config `data.target: ldm.data.ibd.IBDDataModule`)."""

    def __init__(self, batch_size, dims=3, scale_mm=50.0, flip_y=True, max_train_events=None,
                 max_val_events=None, seed=0, data=None):
        super().__init__()
        self.batch_size = batch_size
        self.kw = dict(batch_size=batch_size, dims=dims, scale_mm=scale_mm, seed=seed,
                       data=dict(data) if data else None)
        self.flip_y = flip_y
        self.max_train_events, self.max_val_events = max_train_events, max_val_events
        self.datasets = {}

    def setup(self, stage=None):
        if self.datasets:                  # main.py calls setup() itself, then Lightning again
            return
        self.datasets = {
            "train": IBDDisplacements("train", flip_y=self.flip_y, max_events=self.max_train_events, **self.kw),
            "validation": IBDDisplacements("validation", max_events=self.max_val_events, **self.kw),
        }
        for k, ds in self.datasets.items():
            print(f"IBD {k}: {ds.n_events:,} events, {len(ds)} batches of {self.batch_size}")

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=None, num_workers=0, pin_memory=True)
