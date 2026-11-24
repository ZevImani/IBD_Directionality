#!/usr/bin/env python
"""Train a ResNet to reconstruct the neutrino direction theta from a bulk sample of N IBD
events (one image = displacement histogram of N events, see cnn/images.py).

Run from the repo root:

    python -m cnn.train --pitch 50                                  # defaults below
    python -m cnn.train --pitch 50 --representation discrete --arch resnet18 --epochs 30
    python -m cnn.train --resume cnn/checkpoints/<name>/last.pt     # continue (same args)
    sbatch cnn/submit_train.sh --pitch 50                           # on a GPU node

Training samples are generated on the fly on the GPU: per image, N ~ log-uniform in
[n_min, n_max] usable events are drawn (with replacement) from the training pool and rotated
CCW by theta ~ U(theta_range). The network outputs a 2-vector u; theta_hat = atan2(u_y, u_x).
Loss per image: w * (1 - cos(theta_hat - theta)) ~ w * err^2 / 2, with w = N for
--weighting N (default; minimises the mean Theta^2 = N sigma^2, so every N counts equally
in units of the 1/sqrt(N) scaling) or w = 1 for --weighting none (dominated by the smallest N).

Data split (indices into the paired pool of dirreco.data.load_events):
    [0, holdout)                    never seen in training: reserved for run.py studies
                                    (set `data: {max_events: <holdout>}` in the study config)
    [holdout, holdout + n_val)      validation
    [holdout + n_val, end)          training
Each split gets the usable cut |r| > pitch/sqrt(2), like the engine.

Output: cnn/checkpoints/<name>/{best.pt, last.pt, log.csv, args.json}. Each validation pass
also reports the centroid (continuous, or discrete for --representation discrete) on the same
samples, so the `ratio` column (CNN rms / centroid rms) tells you directly whether the network
beats the centroid.
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cnn.images import ImageSpec, make_images  # noqa: E402
from cnn.resnet import ARCHS, build_model  # noqa: E402
from dirreco.data import load_events  # noqa: E402

EXIT_INTERRUPTED = 75  # stopped by SIGTERM (time limit) after saving last.pt: resume, don't trust best.pt yet


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("image")
    g.add_argument("--pitch", type=float, help="detector pitch [mm] (required unless --resume)")
    g.add_argument("--representation", default="continuous", choices=["continuous", "discrete"])
    g.add_argument("--extent-mm", type=float, default=200.0, help="half-width of the displacement window")
    g.add_argument("--bin-mm", type=float, default=None, help="image cell size (default: pitch)")
    g = ap.add_argument_group("samples")
    g.add_argument("--n-min", type=int, default=30, help="smallest N per image")
    g.add_argument("--n-max", type=int, default=10000, help="largest N per image")
    g.add_argument("--theta-range", type=float, nargs=2, default=[-180.0, 180.0],
                   help="training theta range [deg]; a window < 360 gives the net a prior")
    g.add_argument("--holdout", type=int, default=2_000_000, help="first events reserved for evaluation")
    g.add_argument("--n-val", type=int, default=500_000, help="validation pool size (events)")
    g.add_argument("--max-events", type=int, default=None, help="load only this many events (testing)")
    g = ap.add_argument_group("model / optimisation")
    g.add_argument("--arch", default="resnet50", choices=sorted(ARCHS))
    g.add_argument("--stem", default="small", choices=["small", "imagenet"])
    g.add_argument("--norm", default="batch", choices=["batch", "instance"])
    g.add_argument("--weighting", default="N", choices=["N", "none"])
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--steps-per-epoch", type=int, default=500)
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g = ap.add_argument_group("validation / io")
    g.add_argument("--val-N", default="30,100,300,1000,3000,10000", help="comma list of N")
    g.add_argument("--val-per-N", type=int, default=2000, help="validation images per N")
    g.add_argument("--name", help="checkpoint subdirectory (default: <arch>_p<pitch>_<representation>)")
    g.add_argument("--out", default=str(REPO / "cnn" / "checkpoints"))
    g.add_argument("--resume", help="last.pt to continue from (its saved args are used)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default=None)
    g.add_argument("--quiet", action="store_true", help="no per-step progress line")
    a = ap.parse_args(argv)
    if a.pitch is None and not a.resume:
        ap.error("--pitch is required")
    return a


# ----------------------------------------------------------------------------- data
class Pool:
    """Usable events of one split, as float64 (x, y) tensors on the device."""

    def __init__(self, events, lo, hi, pitch, device):
        v, c = events.vtx[lo:hi, :2], events.cap[lo:hi, :2]
        d = c - v
        ok = d[:, 0] ** 2 + d[:, 1] ** 2 > pitch ** 2 / 2.0
        self.eff = float(ok.mean())
        self.vtx = torch.as_tensor(np.ascontiguousarray(v[ok]), dtype=torch.float64, device=device)
        self.cap = torch.as_tensor(np.ascontiguousarray(c[ok]), dtype=torch.float64, device=device)

    def __len__(self):
        return len(self.vtx)

    def batch(self, spec, n_per_image, theta_range, gen):
        """n_per_image: (B,) long tensor. -> images, theta_true [deg], centroid estimate [deg]."""
        dev = self.vtx.device
        B = len(n_per_image)
        img_id = torch.repeat_interleave(torch.arange(B, device=dev), n_per_image)
        idx = torch.randint(len(self), (len(img_id),), device=dev, generator=gen)
        lo, hi = theta_range
        theta = lo + (hi - lo) * torch.rand(B, device=dev, generator=gen, dtype=torch.float64)
        x, d = make_images(self.vtx[idx], self.cap[idx], img_id, B, spec, theta)
        s = torch.zeros(B, 2, device=dev, dtype=d.dtype).index_add_(0, img_id, d)
        centroid = torch.rad2deg(torch.atan2(s[:, 1], s[:, 0]))
        return x, theta.float(), centroid.float()


def sample_n(B, n_min, n_max, gen, device):
    """N ~ log-uniform integers in [n_min, n_max]."""
    u = torch.rand(B, device=device, generator=gen)
    n = torch.exp(math.log(n_min) + u * (math.log(n_max + 1) - math.log(n_min))).long()
    return n.clamp(n_min, n_max)


# ----------------------------------------------------------------------------- loss
def angle_deg(out):
    return torch.rad2deg(torch.atan2(out[:, 1], out[:, 0]))


def wrap(deg):
    return torch.remainder(deg + 180.0, 360.0) - 180.0


def angular_loss(out, theta_deg, weight):
    """weight * (1 - cos(theta_hat - theta)); out is normalised so only its direction counts."""
    u = out / out.norm(dim=1, keepdim=True).clamp(min=1e-8)
    t = torch.deg2rad(theta_deg)
    cos_err = u[:, 0] * torch.cos(t) + u[:, 1] * torch.sin(t)
    return (weight * (1.0 - cos_err)).mean()


@torch.no_grad()
def validate(model, pool, spec, args, val_N, device):
    """Fixed validation set (same seed every epoch). Returns (loss, {N: (rms_cnn, rms_centroid)})."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    stats, losses = {}, []
    for N in val_N:
        err_c, err_m = [], []
        for start in range(0, args.val_per_N, args.batch_size):
            B = min(args.batch_size, args.val_per_N - start)
            n = torch.full((B,), N, dtype=torch.long, device=device)
            x, theta, cen = pool.batch(spec, n, args.theta_range, gen)
            out = model(x)
            w = n.float() if args.weighting == "N" else torch.ones_like(theta)
            losses.append(angular_loss(out, theta, w).item() * B)
            err_m.append(wrap(angle_deg(out) - theta))
            err_c.append(wrap(cen - theta))
        rms = [float(torch.cat(e).pow(2).mean().sqrt()) for e in (err_m, err_c)]
        stats[N] = rms
    model.train()
    return sum(losses) / (len(val_N) * args.val_per_N), stats


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = get_args(argv)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved = ckpt["args"]
        for k in ("resume", "device", "quiet", "max_events"):
            saved[k] = getattr(args, k)
        args = argparse.Namespace(**saved)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    spec = ImageSpec(args.pitch, args.representation, args.extent_mm, args.bin_mm)
    name = args.name or f"{args.arch}_p{args.pitch:g}_{args.representation}"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=1)
    val_N = [int(v) for v in str(args.val_N).split(",")]

    t0 = time.time()
    ev = load_events(max_events=args.max_events)
    n_tot = len(ev)
    v_lo, v_hi = args.holdout, args.holdout + args.n_val
    if v_hi >= n_tot:
        raise SystemExit(f"pool has {n_tot:,} events; holdout + n_val = {v_hi:,} leaves nothing to train on")
    val_pool = Pool(ev, v_lo, v_hi, args.pitch, device)
    train_pool = Pool(ev, v_hi, n_tot, args.pitch, device)
    del ev
    print(f"[data] {n_tot:,} events; held out [0, {v_lo:,}); val {len(val_pool):,} / train "
          f"{len(train_pool):,} usable (eff {train_pool.eff:.3f}) at pitch {args.pitch:g} mm "
          f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[image] {spec.size}x{spec.size}, {spec.representation}, bin {spec.bin_mm:g} mm, "
          f"extent {spec.extent_mm:g} mm | N in [{args.n_min}, {args.n_max}] | theta in {args.theta_range}",
          flush=True)

    model = build_model(args.arch, num_classes=2, channels=1, norm=args.norm, stem=args.stem).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    start_epoch, best = 0, float("inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch, best = ckpt["epoch"], ckpt["best_val_loss"]
        print(f"[resume] from epoch {start_epoch}, best val loss {best:.5g}", flush=True)
    print(f"[model] {args.arch} stem={args.stem} norm={args.norm}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, device {device}", flush=True)

    def save(path, epoch, val_loss):
        torch.save({"epoch": epoch, "best_val_loss": best, "val_loss": val_loss, "args": vars(args),
                    "image_spec": spec.to_dict(), "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict()},
                   path)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))  # SLURM --signal: finish epoch

    log_path = out_dir / "log.csv"
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr,seconds," +
                    ",".join(f"rms_N{N},ratio_N{N}" for N in val_N) + "\n")

    gen = torch.Generator(device=device).manual_seed(args.seed + start_epoch)
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t_ep, run_loss = time.time(), 0.0
        for step in range(args.steps_per_epoch):
            n = sample_n(args.batch_size, args.n_min, args.n_max, gen, device)
            x, theta, _ = train_pool.batch(spec, n, args.theta_range, gen)
            w = n.float() if args.weighting == "N" else torch.ones_like(theta)
            loss = angular_loss(model(x), theta, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
            if not args.quiet and (step + 1) % 50 == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{args.steps_per_epoch} "
                      f"loss {run_loss / (step + 1):.5g}", end="\r", flush=True)
        train_loss = run_loss / args.steps_per_epoch

        val_loss, stats = validate(model, val_pool, spec, args, val_N, device)
        if val_loss < best:
            best = val_loss
            save(out_dir / "best.pt", epoch + 1, val_loss)
        save(out_dir / "last.pt", epoch + 1, val_loss)
        secs = time.time() - t_ep
        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{train_loss:.6g},{val_loss:.6g},{sched.get_last_lr()[0]:.4g},{secs:.1f}," +
                    ",".join(f"{m:.4g},{m / c:.4g}" for m, c in stats.values()) + "\n")
        table = "  ".join(f"N={N}: {m:.2f}deg ({m / c:.2f}x cen)" for N, (m, c) in stats.items())
        gpu = (f", GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device.type == "cuda" else "")
        print(f"Epoch {epoch + 1}/{args.epochs} train {train_loss:.5g} val {val_loss:.5g}"
              f"{' *' if val_loss == best else ''} lr {sched.get_last_lr()[0]:.3g} ({secs:.0f}s{gpu})\n    {table}",
              flush=True)
        if stop["flag"] and epoch + 1 < args.epochs:
            print(f"[SIGTERM] stopped after epoch {epoch + 1}/{args.epochs}; "
                  f"resume with --resume {out_dir / 'last.pt'}", flush=True)
            sys.exit(EXIT_INTERRUPTED)
    print(f"Done. best val loss {best:.5g} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
