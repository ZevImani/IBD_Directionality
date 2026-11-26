#!/usr/bin/env python
"""Compare several generated closure runs against one real run, side by side.

    python ldm/closure_compare.py ldm_closure ldm_closure_gen_e137_fix ldm_closure_gen_e137_fix_cut \
        [-o results/studies/ldm_closure_gen_e137_fix_cut/plots/closure_compare.png]

Prints, per pitch / N / method, sigma_gen / sigma_real (+- error, pull) for each generated run,
the pool-level theta0 ratios and sum pull^2 per run, and draws one panel per (method, pitch) with
every run's ratio vs N. Ratios and pulls are computed exactly as in ldm/closure.py.

Different generated runs draw their trials independently (a pool that lost a few events gives a
different random draw), so the difference between two runs' ratios at one grid point has an
error of about sqrt(2) x 3% at K=1000, apart from what the change itself does.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dirreco.results import load_run  # noqa: E402

KEYS = ["pitch_mm", "N", "theta_true", "method"]
# Categorical slots 1-4 (dataviz reference palette, light mode), assigned in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D", "^"]


def ratios(real, gen):
    m = real.summary.merge(gen.summary, on=KEYS, suffixes=("_r", "_g"))
    m["ratio"] = m["sigma_g"] / m["sigma_r"]
    m["ratio_err"] = m["ratio"] * np.hypot(m["sigma_err_r"] / m["sigma_r"], m["sigma_err_g"] / m["sigma_g"])
    m["pull"] = (m["ratio"] - 1) / m["ratio_err"]
    return m.set_index(KEYS).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("real")
    ap.add_argument("gen", nargs="+", help="generated runs (names or dirs), 1-4")
    ap.add_argument("-o", "--out", help="figure path (default: <last gen run>/plots/closure_compare.png)")
    a = ap.parse_args()
    if len(a.gen) > len(COLORS):
        sys.exit(f"at most {len(COLORS)} generated runs")

    real = load_run(a.real)
    gens = [load_run(g) for g in a.gen]
    labels = [os.path.basename(str(g.dir).rstrip("/")) for g in gens]
    M = [ratios(real, g) for g in gens]
    idx = M[0].index
    for lab, m in zip(labels[1:], M[1:]):
        idx = idx.intersection(m.index)
    if idx.empty:
        sys.exit("no common (pitch, N, theta_true, method) grid points")

    print(f"real: {real.dir}")
    for i, g in enumerate(gens):
        print(f"gen{i + 1}: {g.dir}")

    print("\npool level: theta0 gen/real (eff gen/real)")
    for p in sorted(idx.get_level_values("pitch_mm").unique()):
        cells = []
        for m in M:
            r = m.xs(p, level="pitch_mm").iloc[0]
            cells.append(f"{r.theta0_g / r.theta0_r:.4f} ({r.eff_g / r.eff_r:.4f})")
        print(f"  pitch {p:5g}: " + "   ".join(f"gen{i + 1} {c}" for i, c in enumerate(cells)))

    head = f"\n{'pitch':>6s} {'N':>6s} {'method':15s} {'sig_real':>9s}"
    for i in range(len(M)):
        head += f" | {'gen' + str(i + 1) + ' ratio':>14s} {'pull':>5s}"
    print(head)
    for k in idx:
        r0 = M[0].loc[k]
        line = f"{k[0]:6g} {k[1]:6d} {k[3]:15s} {r0.sigma_r:9.3f}"
        for m in M:
            r = m.loc[k]
            line += f" | {r.ratio:7.3f}+-{r.ratio_err:.3f} {r.pull:5.1f}"
        print(line)
    print("\nsum pull^2 over " + f"{len(idx)} points (indicative; methods share trials): "
          + "   ".join(f"gen{i + 1} {float((m.loc[idx, 'pull'] ** 2).sum()):.1f}" for i, m in enumerate(M)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pitches = sorted(idx.get_level_values("pitch_mm").unique(), reverse=True)
    methods = sorted(idx.get_level_values("method").unique())
    fig, axes = plt.subplots(len(methods), len(pitches), figsize=(3.4 * len(pitches), 2.3 * len(methods)),
                             sharex=True, sharey=True, squeeze=False)
    for i, meth in enumerate(methods):
        for j, p in enumerate(pitches):
            ax = axes[i, j]
            ax.axhline(1, color="#888888", lw=0.8, zorder=0)
            for g, (lab, m) in enumerate(zip(labels, M)):
                s = m.loc[[k for k in idx if k[0] == p and k[3] == meth]].reset_index().sort_values("N")
                x = s["N"] * (1 + 0.08 * (g - (len(M) - 1) / 2))
                ax.errorbar(x, s["ratio"], s["ratio_err"], fmt=MARKERS[g], ms=6, lw=1.5, capsize=2,
                            color=COLORS[g], mec="white", mew=0.8, label=lab)
            ax.set_xscale("log")
            ax.grid(axis="y", color="#e5e5e5", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(f"pitch {p:g} mm", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{meth}\n" + r"$\sigma_{gen}/\sigma_{real}$", fontsize=9)
            if i == len(methods) - 1:
                ax.set_xlabel("N")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"closure vs {os.path.basename(str(real.dir).rstrip('/'))}", fontsize=10, y=0.995)
    fig.legend(handles, labs, loc="upper center", ncol=len(labs), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = a.out or os.path.join(str(gens[-1].dir), "plots", "closure_compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
