"""Physics-level comparisons of generated vs real IBD displacements r (n, dims) in mm.

Used by EventDDPM during validation (scalar metrics + a histogram figure) and by evaluate.py.
Conventions follow dirreco: beam along +x, usable events have |r_xy| > pitch / sqrt(2), and
theta0 = std(r_y) / mean(r_x) in degrees (the centroid's asymptotic Theta = sigma_theta * sqrt(N)).
"""
import numpy as np
from scipy.stats import wasserstein_distance

AXES = "xyz"


def usable(r, pitch):
    return r[:, 0] ** 2 + r[:, 1] ** 2 > (pitch / np.sqrt(2.0)) ** 2


def theta0(r):
    """Centroid asymptotic resolution [deg * sqrt(events)], as dirreco.data.theta0."""
    return float(np.degrees(np.std(r[:, 1]) / np.mean(r[:, 0])))


def centroid_angle(r):
    """(angle, error) [deg] of the mean displacement. Should be 0 for an unbiased sample."""
    mx, my = r[:, 0].mean(), r[:, 1].mean()
    err = np.std(r[:, 1]) / np.sqrt(len(r)) / mx
    return float(np.degrees(np.arctan2(my, mx))), float(np.degrees(err))


def compare(real, gen, pitch=50.0):
    """Scalar gen-vs-real summary (logged every validation epoch)."""
    out = {}
    for i in range(real.shape[1]):
        a = AXES[i]
        out[f"dmean_{a}"] = float(gen[:, i].mean() - real[:, i].mean())
        out[f"dstd_{a}"] = float(gen[:, i].std() - real[:, i].std())
        out[f"w1_{a}"] = float(wasserstein_distance(real[:, i], gen[:, i]))
    ur, ug = usable(real, pitch), usable(gen, pitch)
    out[f"eff{pitch:g}_ratio"] = float(ug.mean() / ur.mean())
    out[f"theta0_{pitch:g}_ratio"] = theta0(gen[ug]) / theta0(real[ur])
    out["angle_deg"] = centroid_angle(gen)[0]
    return out


def _features(r):
    """Named 1D projections shown in the marginal plots."""
    f = {f"r_{AXES[i]} [mm]": r[:, i] for i in range(r.shape[1])}
    f["|r_xy| [mm]"] = np.hypot(r[:, 0], r[:, 1])
    f["phi_xy [deg]"] = np.degrees(np.arctan2(r[:, 1], r[:, 0]))
    return f


_RANGES = {"|r_xy| [mm]": (0, 300), "phi_xy [deg]": (-180, 180)}


def comparison_figure(real, gen, labels=("real", "generated"), title=None, bins=80):
    """Marginals of `real` and `gen` (density, top) and their ratio with Poisson errors (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr, fg = _features(real), _features(gen)
    n = len(fr)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 4.6), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})
    for j, name in enumerate(fr):
        lo, hi = _RANGES.get(name, (-250, 250))
        edges = np.linspace(lo, hi, bins + 1)
        hr, _ = np.histogram(fr[name], edges)
        hg, _ = np.histogram(fg[name], edges)
        wr, wg = 1.0 / (len(real) * np.diff(edges)), 1.0 / (len(gen) * np.diff(edges))
        ax = axes[0, j]
        ax.stairs(hr * wr, edges, color="k", label=labels[0])
        ax.stairs(hg * wg, edges, color="C1", label=labels[1])
        ax.set_yscale("log" if name.startswith("r_") or name.startswith("|") else "linear")
        ax.set_title(name, fontsize=9)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (hg * wg) / (hr * wr)
            err = ratio * np.sqrt(1.0 / hg + 1.0 / hr)
        c = 0.5 * (edges[1:] + edges[:-1])
        ok = (hr > 0) & (hg > 0)
        axes[1, j].errorbar(c[ok], ratio[ok], err[ok], fmt=".", ms=2, color="C1", lw=0.6)
        axes[1, j].axhline(1, color="k", lw=0.6)
        axes[1, j].set_ylim(0.8, 1.2)
        axes[1, j].set_xlabel(name, fontsize=8)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_ylabel(f"{labels[1]} / {labels[0]}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def figure_to_tensor(fig):
    """Matplotlib figure -> (1, 3, H, W) float tensor in [-1, 1] (the range ImageLogger expects)."""
    import matplotlib.pyplot as plt
    import torch
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
    plt.close(fig)
    return torch.from_numpy(img / 127.5 - 1.0).permute(2, 0, 1)[None]
