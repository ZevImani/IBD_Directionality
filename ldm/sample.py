#!/usr/bin/env python
"""Generate IBD events with a trained EventDDPM and write them in the dirreco format.

    python sample.py -r logs/<run>/checkpoints/last.ckpt -n 2000000 -o samples/<name>

Writes to the output directory:
  vertices.npy   (n, 3) float64 mm, uniform in the MC's 40 m cube (independent of r; see ldm/data/ibd.py)
  captures.npy   (n, 3) float64 mm, vertex + generated displacement
  meta.json      checkpoint, sampler settings, seed, timing

The files are paired 1:1 (capture_offset 0). Use them in any study with
    python run.py configs/<study>.yaml --data ldm/samples/<name> --tag gen     (from emd_paper/)

By default each event's r_y is multiplied by a random sign. y -> -y is an exact symmetry of the
MC, and this makes the mean r_y (which would bias the direction) exactly 0 in expectation.
--no-symmetrize keeps the raw model output (use it to check the model's own y-asymmetry).
For a dims=2 model r_z is not modelled and is written as 0.

The config is taken from the run directory (<run>/configs/*-project.yaml) unless -b is given.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldm.data.ibd import BOX_MM  # noqa: E402
from ldm.models.diffusion.event_ddpm import load_event_ddpm  # noqa: E402


def find_config(ckpt):
    run = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    found = sorted(glob.glob(os.path.join(run, "configs", "*-project.yaml")))
    if not found:
        raise FileNotFoundError(f"no configs/*-project.yaml next to {ckpt}; pass -b")
    return found[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--resume", required=True, help="checkpoint (.ckpt)")
    ap.add_argument("-b", "--base", help="config yaml (default: the run's saved project config)")
    ap.add_argument("-n", "--n-events", type=int, default=2_000_000)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--ddim-steps", type=int, default=1000,
                    help="DDIM steps; 0 = ancestral DDPM over all timesteps. <1000 biases the width "
                         "(exact-denoiser test: -0.4%% at 250, -1.1%% at 100)")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0 = deterministic)")
    ap.add_argument("--batch-size", type=int, default=262_144)
    ap.add_argument("--no-ema", action="store_true", help="use the raw instead of the EMA weights")
    ap.add_argument("--no-symmetrize", action="store_true", help="do not randomise the sign of r_y")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg_path = a.base or find_config(a.resume)
    config = OmegaConf.load(cfg_path)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = load_event_ddpm(config, a.resume)
    print(f"Model {a.resume} ({cfg_path}) on {model.device}; dims={model.dims}, scale={model.scale_mm} mm")

    t0 = time.time()
    r = model.sample_events(a.n_events, ddim_steps=a.ddim_steps or None, eta=a.eta,
                            batch_size=a.batch_size, use_ema=not a.no_ema, progress=True).numpy()
    dt = time.time() - t0
    if model.dims == 2:
        r = np.hstack([r, np.zeros((len(r), 1))])
    raw_mean = r.mean(0)
    if not a.no_symmetrize:
        r[:, 1] *= rng.choice([-1.0, 1.0], size=len(r))

    vtx = rng.uniform(-BOX_MM, BOX_MM, size=(len(r), 3))
    os.makedirs(a.outdir, exist_ok=True)
    np.save(os.path.join(a.outdir, "vertices.npy"), vtx)
    np.save(os.path.join(a.outdir, "captures.npy"), vtx + r)
    meta = dict(checkpoint=os.path.abspath(a.resume), config=os.path.abspath(cfg_path), n_events=len(r),
                dims=model.dims, ddim_steps=a.ddim_steps, eta=a.eta, ema=not a.no_ema,
                symmetrize_y=not a.no_symmetrize, seed=a.seed, seconds=round(dt, 1),
                raw_mean_r_mm=[round(float(v), 4) for v in raw_mean],
                host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    with open(os.path.join(a.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{len(r):,} events in {dt:.0f} s -> {a.outdir}")
    print(f"raw model mean r = {np.round(raw_mean, 3)} mm (real pool: [16.825, -0.015, 0.899])")


if __name__ == "__main__":
    main()
