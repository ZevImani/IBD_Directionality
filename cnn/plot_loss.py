#!/usr/bin/env python
"""Plot a training log: loss, and CNN rms error relative to the centroid per validation N.

    python cnn/plot_loss.py cnn/checkpoints/resnet50_p50_continuous     # -> <dir>/training.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main(run_dir):
    run_dir = Path(run_dir)
    log = pd.read_csv(run_dir / "log.csv")
    Ns = [c[len("ratio_N"):] for c in log.columns if c.startswith("ratio_N")]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(log.epoch, log.train_loss, "o-", ms=3, label="train")
    ax.plot(log.epoch, log.val_loss, "s-", ms=3, label="validation")
    ax.set(yscale="log", xlabel="epoch", ylabel="loss", title="loss")
    ax.legend()

    for i, N in enumerate(Ns):
        c = cmap(i / max(1, len(Ns) - 1))
        axes[1].plot(log.epoch, log[f"rms_N{N}"], "o-", ms=3, color=c, label=f"N={N}")
        axes[2].plot(log.epoch, log[f"ratio_N{N}"], "o-", ms=3, color=c, label=f"N={N}")
    axes[1].set(yscale="log", xlabel="epoch", ylabel="rms error [deg]", title="CNN rms error (validation)")
    axes[2].axhline(1.0, color="gray", ls=":", lw=1.2)
    axes[2].set(xlabel="epoch", ylabel="rms(CNN) / rms(centroid)", title="< 1: CNN beats the centroid")
    axes[2].legend(fontsize=8)
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    out = run_dir / "training.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
