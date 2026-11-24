#!/usr/bin/env python
"""Compare one dirreco study run on real events and on generated events (the closure test).

    python ldm/closure.py ldm_closure ldm_closure_gen          (from emd_paper/; run names or dirs)

Prints, per pitch / N / method, sigma_gen / sigma_real with its error and the bias of each, plus
the pool-level eff and theta0 ratios, and writes <gen run>/plots/closure.png. A generator that
is good enough for the paper gives sigma ratios consistent with 1 everywhere (pull |z| < 2-3) and
biases consistent with 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dirreco.results import load_run  # noqa: E402

KEYS = ["pitch_mm", "N", "theta_true", "method"]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    real, gen = load_run(sys.argv[1]), load_run(sys.argv[2])
    m = real.summary.merge(gen.summary, on=KEYS, suffixes=("_r", "_g"))
    if m.empty:
        sys.exit("no common (pitch, N, theta_true, method) grid points")
    m["ratio"] = m["sigma_g"] / m["sigma_r"]
    m["ratio_err"] = m["ratio"] * np.hypot(m["sigma_err_r"] / m["sigma_r"], m["sigma_err_g"] / m["sigma_g"])
    m["pull"] = (m["ratio"] - 1) / m["ratio_err"]
    m["bias_err_g"] = m["sigma_g"] / np.sqrt(m["K_g"])

    print(f"real: {real.dir}\ngen:  {gen.dir}\n")
    pools = m.groupby("pitch_mm")[["eff_r", "eff_g", "theta0_r", "theta0_g"]].first()
    print("pool level (usable fraction, theta0):")
    for p, r in pools.iterrows():
        print(f"  pitch {p:5g}: eff gen/real {r.eff_g / r.eff_r:.4f}   theta0 gen/real {r.theta0_g / r.theta0_r:.4f}")
    print(f"\n{'pitch':>6s} {'N':>6s} {'method':15s} {'sig_real':>9s} {'sig_gen':>9s} {'ratio':>14s} {'pull':>5s}"
          f" {'bias_gen':>14s}")
    for _, r in m.sort_values(KEYS).iterrows():
        print(f"{r.pitch_mm:6g} {r.N:6d} {r.method:15s} {r.sigma_r:9.3f} {r.sigma_g:9.3f} "
              f"{r.ratio:7.3f}+-{r.ratio_err:.3f} {r.pull:5.1f} {r.bias_g:+7.3f}+-{r.bias_err_g:.3f}")
    chi2 = float((m["pull"] ** 2).sum())
    print(f"\nsum pull^2 = {chi2:.1f} for {len(m)} points "
          f"(points share trials across methods, so this is only indicative)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pitches = sorted(m["pitch_mm"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(pitches), figsize=(4 * len(pitches), 3.6), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], pitches):
        sub = m[m["pitch_mm"] == p]
        methods = sorted(sub["method"].unique())
        for j, meth in enumerate(methods):
            s = sub[sub["method"] == meth].sort_values("N")
            x = s["N"] * (1 + 0.06 * (j - (len(methods) - 1) / 2))
            ax.errorbar(x, s["ratio"], s["ratio_err"], fmt="o", ms=4, capsize=2, label=meth)
        ax.axhline(1, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("N (usable events)")
        ax.set_title(f"pitch {p:g} mm")
    axes[0, 0].set_ylabel(r"$\sigma_\theta$ generated / real")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    out = gen.plot_dir / "closure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
