#!/usr/bin/env python
"""Measure the resources cnn/train.py needs, to size cnn/submit_train.sh requests.

    python -m cnn.benchmark                                   # default grid below
    python -m cnn.benchmark --configs 50:resnet50:small:256 5:resnet50:small:128

Each config is pitch:arch:stem:batch_size. Reports per config: host peak RSS, GPU peak
memory, seconds per training step (data generation and model separately), seconds per
validation pass, and process CPU time / wall time (CPU utilisation; 1.0 = one core busy).
Uses the real Pool / make_images / model code from train.py, on the full event pool.
"""
import argparse
import resource
import time

import torch

from cnn.images import ImageSpec
from cnn.resnet import build_model
from cnn.train import Pool, angular_loss, get_args, sample_n, validate
from dirreco.data import load_events

DEFAULT = ["150:resnet50:small:256", "50:resnet50:small:256", "5:resnet50:small:256",
           "5:resnet50:small:128", "5:resnet50:small:64", "5:resnet18:small:256",
           "5:resnet50:imagenet:256"]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--val", action="store_true", help="also time one full validation pass")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).multi_processor_count} SMs, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)

    t = time.time()
    ev = load_events()
    print(f"[load] {len(ev):,} events in {time.time() - t:.0f}s, host peak RSS {rss_gb():.2f} GB", flush=True)
    pools = {}
    for cfg in a.configs:
        pitch, arch, stem, bs = cfg.split(":")
        pitch, bs = float(pitch), int(bs)
        if pitch not in pools:
            pools.clear()
            torch.cuda.empty_cache()
            t = time.time()
            pools[pitch] = Pool(ev, 2_500_000, len(ev), pitch, dev)
            print(f"[pool] pitch {pitch:g}: {len(pools[pitch]):,} usable in {time.time() - t:.0f}s, "
                  f"host peak RSS {rss_gb():.2f} GB, GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        pool = pools[pitch]
        args = get_args(["--pitch", f"{pitch:g}", "--arch", arch, "--stem", stem, "--batch-size", str(bs)])
        spec = ImageSpec(pitch)
        model = build_model(arch, stem=stem).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        gen = torch.Generator(device=dev).manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        try:
            t_data = t_model = 0.0
            for i in range(3 + a.steps):  # 3 warm-up steps (cudnn benchmark)
                sync()
                t0 = time.time()
                if i == 3:
                    cpu0, wall0 = time.process_time(), t0
                n = sample_n(bs, args.n_min, args.n_max, gen, dev)
                x, theta, _ = pool.batch(spec, n, args.theta_range, gen)
                sync(); t1 = time.time()
                loss = angular_loss(model(x), theta, n.float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sync(); t2 = time.time()
                if i >= 3:
                    t_data += t1 - t0
                    t_model += t2 - t1
            cpu_util = (time.process_time() - cpu0) / (time.time() - wall0)
            msg = (f"[train] {cfg:28s} img {spec.size}x{spec.size}: {(t_data + t_model) / a.steps:.3f} s/step "
                   f"(data {t_data / a.steps:.3f}, model {t_model / a.steps:.3f}), "
                   f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                   f"(reserved {torch.cuda.max_memory_reserved() / 1e9:.2f}), CPU util {cpu_util:.2f}, "
                   f"host peak RSS {rss_gb():.2f} GB")
            if a.val:
                sync(); t0 = time.time()
                validate(model, pool, spec, args, [int(v) for v in args.val_N.split(",")], dev)
                sync()
                msg += f", validation pass {time.time() - t0:.1f}s"
            print(msg, flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[train] {cfg:28s} img {spec.size}x{spec.size}: OOM "
                  f"(> {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)", flush=True)
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
