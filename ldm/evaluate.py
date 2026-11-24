#!/usr/bin/env python
"""Quality tests of generated IBD events against real MC the generator never saw.

    python evaluate.py samples/<name>                    # -> samples/<name>/eval/
    python evaluate.py samples/<name> -n 500000 --skip c2st,memorization

Real reference: the `eval` split (events [0, 2M), reserved for run.py studies; the generator trains
on [2.5M, end)). It is cut into two disjoint halves A and B. Every test is reported for
    gen vs A   (what we want to know)
    B   vs A   (real-vs-real baseline: the value a perfect generator would get at this sample size)
so no number needs an absolute threshold. A test passes when gen-vs-A is consistent with B-vs-A.

Tests (all on the displacement r = capture - vertex, mm):
  moments       mean, std, skewness, excess kurtosis per axis, with pulls
  physics       per pitch: usable fraction, theta0 = std(r_y)/mean(r_x) on usable events (the
                centroid's Theta, what sets every method's resolution), centroid angle (must be 0)
  marginals     KS statistic and W1 of r_x, r_y, r_z, |r_xy|, phi_xy, |r|; tail fractions
  binned        two-sample chi2/ndf of the 2D (r_x, r_y) histogram, and of the detector's cell-hop
                histogram (vertex and capture snapped to cells) at several pitches
  c2st          classifier two-sample test: AUC of a gradient-boosted classifier trained to tell
                gen from real (0.5 = indistinguishable). The most sensitive global test
  memorization  nearest-neighbour distance of gen events to the 6.5M training events, compared
                with the same distance for unseen real events (copies would sit at ~0)

Writes report.txt, metrics.json, marginals.png, xy_pulls.png and nn_distance.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldm import ibd_metrics as M  # noqa: E402
from ldm.data.ibd import load_displacements, load_split  # noqa: E402

AX = "xyz"


# ----------------------------------------------------------------------------- helpers
def chi2_two_sample(a, b):
    """chi2/ndf between two histograms of different totals (bins with a+b >= 10)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = (a + b) >= 10
    a, b = a[ok], b[ok]
    na, nb = a.sum(), b.sum()
    chi2 = np.sum((np.sqrt(nb / na) * a - np.sqrt(na / nb) * b) ** 2 / (a + b))
    ndf = ok.sum() - 1
    return float(chi2 / ndf), int(ndf)


def cell_hops(vtx, cap, pitch, half=8):
    """(2*half+1)^2 histogram of the cell-to-cell displacement (what a segmented detector reads)."""
    k = (np.floor(cap[:, :2] / pitch) - np.floor(vtx[:, :2] / pitch)).astype(int) + half
    ok = ((k >= 0) & (k <= 2 * half)).all(1)
    S = 2 * half + 1
    return np.append(np.bincount(k[ok, 0] * S + k[ok, 1], minlength=S * S), (~ok).sum())  # + overflow bin


def projections(r):
    rxy = np.hypot(r[:, 0], r[:, 1])
    out = {f"r_{AX[i]}": r[:, i] for i in range(3)}
    out.update({"|r_xy|": rxy, "phi_xy": np.degrees(np.arctan2(r[:, 1], r[:, 0])), "|r|": np.linalg.norm(r, axis=1)})
    return out


class Report:
    def __init__(self):
        self.lines, self.metrics = [], {}

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(s)


# ----------------------------------------------------------------------------- tests
def t_moments(rep, A, B, G):
    rep("== moments (mm). pull = (X - A) / err; B is the real-vs-real baseline")
    rep(f"{'':8s}{'A':>10s}{'B':>10s}{'gen':>10s}{'pull B':>9s}{'pull gen':>9s}")
    out = {}
    for i in range(3):
        for name, f, err in [
            ("mean", np.mean, lambda x: np.std(x) / np.sqrt(len(x))),
            ("std", np.std, lambda x: np.std(x) / np.sqrt(2 * len(x))),
            ("skew", stats.skew, lambda x: np.sqrt(6 / len(x))),
            ("kurt", stats.kurtosis, lambda x: np.sqrt(24 / len(x))),
        ]:
            a, b, g = f(A[:, i]), f(B[:, i]), f(G[:, i])
            ea = err(A[:, i])
            pb = (b - a) / np.hypot(ea, err(B[:, i]))
            pg = (g - a) / np.hypot(ea, err(G[:, i]))
            rep(f"{name + '_' + AX[i]:8s}{a:10.3f}{b:10.3f}{g:10.3f}{pb:9.1f}{pg:9.1f}")
            out[f"{name}_{AX[i]}"] = dict(A=float(a), B=float(b), gen=float(g), pull_B=float(pb), pull_gen=float(pg))
    return out


def t_physics(rep, A, B, G, pitches):
    rep("\n== physics per pitch: usable fraction, theta0 [deg sqrt(N)] on usable events, centroid angle [deg]")
    rep(f"{'pitch':>6s} | {'eff A':>7s} {'eff gen':>7s} {'ratio':>6s} | {'th0 A':>7s} {'th0 B':>7s} {'th0 gen':>7s}"
        f" {'B/A':>6s} {'gen/A':>6s} | {'angle gen':>14s}")
    out = {}
    for p in pitches:
        row = {}
        for k, R in (("A", A), ("B", B), ("gen", G)):
            u = M.usable(R, p) if p > 0 else np.ones(len(R), bool)
            ang, err = M.centroid_angle(R[u])
            row[k] = dict(eff=float(u.mean()), theta0=M.theta0(R[u]), angle=ang, angle_err=err, n_usable=int(u.sum()))
        rep(f"{p:6g} | {row['A']['eff']:7.4f} {row['gen']['eff']:7.4f} {row['gen']['eff'] / row['A']['eff']:6.3f} | "
            f"{row['A']['theta0']:7.2f} {row['B']['theta0']:7.2f} {row['gen']['theta0']:7.2f} "
            f"{row['B']['theta0'] / row['A']['theta0']:6.3f} {row['gen']['theta0'] / row['A']['theta0']:6.3f} | "
            f"{row['gen']['angle']:+.3f}+-{row['gen']['angle_err']:.3f}")
        out[f"{p:g}"] = row
    rep("theta0 sets the scale of every method's sigma_theta; a gen/A ratio of 1.01 means 1% too optimistic/pessimistic.")
    return out


def t_marginals(rep, A, B, G):
    rep("\n== 1D marginals: KS statistic (p) and W1 [mm or deg]")
    pa, pb, pg = projections(A), projections(B), projections(G)
    out = {}
    for k in pa:
        ksb, ksg = stats.ks_2samp(pa[k], pb[k]), stats.ks_2samp(pa[k], pg[k])
        wb, wg = stats.wasserstein_distance(pa[k], pb[k]), stats.wasserstein_distance(pa[k], pg[k])
        rep(f"{k:8s} KS  B: {ksb.statistic:.4f} (p={ksb.pvalue:.2g})  gen: {ksg.statistic:.4f} (p={ksg.pvalue:.2g})"
            f"   W1  B: {wb:.3f}  gen: {wg:.3f}")
        out[k] = dict(ks_B=float(ksb.statistic), ks_gen=float(ksg.statistic), ksp_gen=float(ksg.pvalue),
                      w1_B=float(wb), w1_gen=float(wg))
    rep("\n== tails: P(|r_xy| > R) x 1e3 (binomial error)")
    tails = {}
    for R in (100, 150, 200, 250, 300):
        vals = {}
        for k, X in (("A", pa), ("B", pb), ("gen", pg)):
            f = float(np.mean(X["|r_xy|"] > R))
            vals[k] = (f, float(np.sqrt(f * (1 - f) / len(X["|r_xy|"]))))
        rep(f"  R={R:3d} mm  A {1e3 * vals['A'][0]:8.3f}+-{1e3 * vals['A'][1]:.3f}  B {1e3 * vals['B'][0]:8.3f}"
            f"  gen {1e3 * vals['gen'][0]:8.3f}+-{1e3 * vals['gen'][1]:.3f}")
        tails[R] = vals
    out["tails"] = tails
    return out


def t_binned(rep, A, B, G, vA, cA, vB, cB, vG, cG, pitches, outdir):
    rep("\n== binned two-sample chi2/ndf")
    edges = np.arange(-250, 251, 5.0)
    h = {k: np.histogram2d(R[:, 0], R[:, 1], [edges, edges])[0] for k, R in (("A", A), ("B", B), ("gen", G))}
    n_in = {"A": len(A), "B": len(B), "gen": len(G)}
    hof = {k: np.append(v.ravel(), n_in[k] - v.sum()) for k, v in h.items()}   # + overflow bin
    cb, cg = chi2_two_sample(hof["A"], hof["B"]), chi2_two_sample(hof["A"], hof["gen"])
    rep(f"(r_x, r_y), 5 mm bins     B: {cb[0]:.3f}   gen: {cg[0]:.3f}   (ndf {cg[1]})")
    out = {"xy_5mm": dict(B=cb[0], gen=cg[0], ndf=cg[1])}
    for p in pitches:
        if p <= 0:
            continue
        ha, hb, hg = cell_hops(vA, cA, p), cell_hops(vB, cB, p), cell_hops(vG, cG, p)
        cb, cg = chi2_two_sample(ha, hb), chi2_two_sample(ha, hg)
        rep(f"cell hops, pitch {p:5g} mm   B: {cb[0]:.3f}   gen: {cg[0]:.3f}   (ndf {cg[1]})")
        out[f"cells_{p:g}"] = dict(B=cb[0], gen=cg[0], ndf=cg[1])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, k in zip(axes, ("B", "gen")):
        na, nk = h["A"].sum(), h[k].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = (np.sqrt(na / nk) * h[k] - np.sqrt(nk / na) * h["A"]) / np.sqrt(h[k] + h["A"])
        im = ax.pcolormesh(edges, edges, np.where(h[k] + h["A"] >= 10, pull, np.nan).T, cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_aspect("equal")
        ax.set_title(f"pull ({k} - A), 5 mm bins")
        ax.set_xlabel("r_x [mm]")
        ax.set_ylabel("r_y [mm]")
    fig.colorbar(im, ax=axes, label="pull [sigma]")
    fig.savefig(os.path.join(outdir, "xy_pulls.png"), dpi=130)
    plt.close(fig)
    return out


def t_c2st(rep, A, B, G, n, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rep(f"\n== classifier two-sample test (gradient-boosted trees, {n:,} per class, 50% held out)")

    def feats(R):
        return np.column_stack(list(projections(R).values()))

    rng = np.random.default_rng(seed)
    out = {}
    for k, X in (("B", B), ("gen", G)):
        m = min(n, len(A), len(X))
        Xa, Xk = feats(A[rng.permutation(len(A))[:m]]), feats(X[rng.permutation(len(X))[:m]])
        Xall, y = np.vstack([Xa, Xk]), np.r_[np.zeros(m), np.ones(m)]
        idx = rng.permutation(2 * m)
        tr, te = idx[:m], idx[m:]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, random_state=seed)
        clf.fit(Xall[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(Xall[te])[:, 1])
        err = np.sqrt(1.0 / (3 * m))      # AUC std under H0 for m/2 per class
        rep(f"  {k:4s} vs A: AUC = {auc:.4f}  (H0 spread ~{err:.4f})")
        out[k] = dict(auc=float(auc), h0_err=float(err))
    rep("  AUC - 0.5 is roughly the fraction of events a perfect classifier could tell apart.")
    return out


def t_memorization(rep, B, G, n, outdir, seed, data):
    from scipy.spatial import cKDTree
    rep(f"\n== memorization: nearest-neighbour distance to the training set ({n:,} queries)")
    t0 = time.time()
    train = load_displacements("train", dims=3, data=data)
    tree = cKDTree(train)
    rng = np.random.default_rng(seed)
    d = {}
    for k, X in (("unseen real", B), ("gen", G)):
        q = X[rng.permutation(len(X))[:n]]
        d[k] = tree.query(q, k=1)[0]
    out = {}
    for k, v in d.items():
        qs = np.quantile(v, [0.001, 0.01, 0.5])
        rep(f"  {k:12s} quantiles 0.1%/1%/50%: {qs[0]:.4f} / {qs[1]:.4f} / {qs[2]:.3f} mm;"
            f"  fraction < 1e-3 mm: {np.mean(v < 1e-3):.2e}")
        out[k] = dict(q001=float(qs[0]), q01=float(qs[1]), median=float(qs[2]), copies=float(np.mean(v < 1e-3)))
    ks = stats.ks_2samp(d["unseen real"], d["gen"])
    rep(f"  KS(gen, unseen real) = {ks.statistic:.4f} (p={ks.pvalue:.2g});  {len(train):,} training events, {time.time() - t0:.0f} s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bins = np.logspace(-3, 1.5, 80)
    for k, v in d.items():
        ax.hist(v, bins, histtype="step", density=True, label=k)
    ax.set_xscale("log")
    ax.set_xlabel("distance to nearest training event [mm]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "nn_distance.png"), dpi=130)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", help="directory with vertices.npy + captures.npy (from sample.py)")
    ap.add_argument("-n", type=int, default=1_000_000, help="events per sample (A, B, gen). Max 1M (eval split / 2)")
    ap.add_argument("--pitches", default="0,5,50,100,150", help="mm; 0 = no usable cut")
    ap.add_argument("--c2st-n", type=int, default=300_000)
    ap.add_argument("--nn-n", type=int, default=100_000)
    ap.add_argument("--skip", default="", help="comma list of: c2st, memorization, binned")
    ap.add_argument("-o", "--outdir", help="default: <samples>/eval")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    pitches = [float(p) for p in a.pitches.split(",")]
    outdir = a.outdir or os.path.join(a.samples, "eval")
    os.makedirs(outdir, exist_ok=True)

    n = min(a.n, 1_000_000)
    vG = np.load(os.path.join(a.samples, "vertices.npy"), mmap_mode="r")
    cG = np.load(os.path.join(a.samples, "captures.npy"), mmap_mode="r")
    n = min(n, len(vG))
    vG, cG = np.array(vG[:n]), np.array(cG[:n])
    v, c = load_split("eval", max_events=2 * n)
    vA, cA, vB, cB = v[:n], c[:n], v[n:2 * n], c[n:2 * n]
    A, B, G = cA - vA, cB - vB, cG - vG

    rep = Report()
    meta_path = os.path.join(a.samples, "meta.json")
    rep(f"Generated: {os.path.abspath(a.samples)}  ({n:,} events)")
    if os.path.exists(meta_path):
        rep("  " + json.dumps(json.load(open(meta_path))))
    rep(f"Real: eval split events [0, {2 * n:,}), halves A and B\n")
    if np.allclose(G[:, 2], 0):
        rep("NOTE: generated r_z == 0 (dims=2 model); ignore the z rows.\n")

    res = {"n": n}
    res["moments"] = t_moments(rep, A, B, G)
    res["physics"] = t_physics(rep, A, B, G, pitches)
    res["marginals"] = t_marginals(rep, A, B, G)
    if "binned" not in skip:
        res["binned"] = t_binned(rep, A, B, G, vA, cA, vB, cB, vG, cG, pitches, outdir)
    if "c2st" not in skip:
        res["c2st"] = t_c2st(rep, A, B, G, a.c2st_n, a.seed)
    if "memorization" not in skip:
        res["memorization"] = t_memorization(rep, B, G, a.nn_n, outdir, a.seed, None)

    fig = M.comparison_figure(A, G, labels=("real (A)", "generated"), title=os.path.basename(os.path.abspath(a.samples)))
    fig.savefig(os.path.join(outdir, "marginals.png"), dpi=130)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("\n".join(rep.lines) + "\n")
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nWrote {outdir}/report.txt, metrics.json and figures")


if __name__ == "__main__":
    main()
