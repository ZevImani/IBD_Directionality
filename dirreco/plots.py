"""Plotting layer: every figure is a function of a Run (summary / trials / landscape).

Register a new figure with

    @figure("my_fig")
    def my_fig(run, some_option=1):
        fig, ax = plt.subplots()
        ...
        return fig

and list it in a config's `plots:` section ({figure: my_fig, some_option: 2}).
"""
import inspect

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .data import wrap  # noqa: E402

plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.linewidth": 1.1,
                     "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
                     "legend.framealpha": 0.92, "savefig.dpi": 300})

FIGURES = {}
LEGACY_THETA0 = 129.58  # Max's single hard-coded constant (50 mm cut); used only if theta0 is missing

# Fixed colour per pitch (entity), never re-ordered by what a figure happens to contain.
PITCH_COLORS = {150: "#2a78d6", 50: "#eb6834", 5: "#1baf7a", 25: "#eda100", 10: "#e87ba4",
                1: "#008300", 100: "#6250d6", 0.5: "#e34948"}
STYLES = [("o", "-"), ("^", "--"), ("D", "-."), ("v", (0, (5, 1, 1, 1))), ("P", (0, (3, 1)))]


def figure(name):
    def deco(fn):
        FIGURES[name] = fn
        return fn
    return deco


def pcolor(p):
    return PITCH_COLORS.get(float(p) if float(p) % 1 else int(p), "#7f7f7f")


def plabel(p):
    return rf"$\Delta x = {p:g}$ mm"


def _methods(s, methods, exclude=()):
    if methods:
        return [m for m in methods if m in set(s["method"])]
    return [m for m in dict.fromkeys(s["method"]) if m not in exclude]


def _pitches(s):
    return sorted(s["pitch_mm"].unique(), reverse=True)


def _has(s, col):
    return col in s and s[col].notna().any()


def _pitch_legend(ax, pitches, extra=(), **kw):
    h = [Line2D([], [], color=pcolor(p), lw=2, label=plabel(p)) for p in pitches] + list(extra)
    ax.legend(handles=h, fontsize=8.5, **kw)


# ----------------------------------------------------------------------------- figures
@figure("capacity")
def capacity(run, methods=None, ref="centroid_disc", x="N"):
    """sigma_theta vs N_usable (x='N') or vs total exposure (x='n_total'); one panel per
    method, reference method dotted, per-pitch Theta0 bound dashed."""
    s = run.summary
    if (s["sweep_var"] == x).any():
        s = s[s["sweep_var"] == x]
    s = s[s["theta_true"] == s["theta_true"].iloc[0]]
    ms = _methods(s, methods, exclude=(ref,))
    ncol = min(4, len(ms))
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.8 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(ms):]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, ms):
        for p in _pitches(s):
            c = pcolor(p)
            sub = s[(s.pitch_mm == p) & (s.method == m)].sort_values(x)
            if sub.empty:
                continue
            ax.errorbar(sub[x], sub["sigma"], yerr=sub["sigma_err"], fmt="o-", color=c, ms=5, lw=1.8, capsize=2)
            r = s[(s.pitch_mm == p) & (s.method == ref)].sort_values(x)
            if ref and not r.empty:
                ax.plot(r[x], r["sigma"], "s:", color=c, ms=3.5, lw=1.2, alpha=0.8)
            if _has(sub, "theta0"):
                n_use = sub["N"]
                ax.plot(sub[x], sub["theta0"] / np.sqrt(n_use), "--", color=c, lw=1, alpha=0.7)
        if not _has(s, "theta0") and x == "N":
            xs = np.logspace(np.log10(s[x].min()), np.log10(s[x].max()), 50)
            ax.plot(xs, LEGACY_THETA0 / np.sqrt(xs), "k--", lw=1.2)
        ax.axhline(1.0, color="gray", ls=":", lw=1.2)
        ax.set(xscale="log", yscale="log", title=m,
               xlabel=r"usable events $N$" if x == "N" else r"total events $n_{\rm total} = N/\epsilon$")
    for row in axes:
        row[0].set_ylabel(r"$\sigma_\theta$ [deg]")
    bound = (r"$\Theta_{0,\rm cut}/\sqrt{N}$ (per pitch)" if _has(s, "theta0")
             else rf"$\Theta_0 = {LEGACY_THETA0}$ (legacy constant)")
    extra = [Line2D([], [], color="k", marker="o", lw=1.8, label="method"),
             Line2D([], [], color="k", marker="s", ls=":", ms=3.5, lw=1.2, label=f"{ref} (ref)"),
             Line2D([], [], color="k", ls="--", lw=1, label=bound)]
    _pitch_legend(axes.flat[len(ms) - 1], _pitches(s), extra, loc="upper right")
    fig.tight_layout()
    return fig


@figure("exposure")
def exposure(run, methods=None, ref="centroid_disc"):
    return capacity(run, methods=methods, ref=ref, x="n_total")


# ----------------------------------------------------------------------------- method comparison
# Colour = method family (entity, fixed), line style = readout. Slots 1-3 of the reference
# palette validate for all pairs; later families only matter when several are plotted at once.
FAMILY_COLORS = {"centroid": "#2a78d6", "cnn": "#eb6834", "fnd": "#1baf7a", "emd": "#eda100",
                 "image_l2": "#e87ba4"}
READOUT_STYLE = {"discrete": dict(ls="-", mfc=None), "continuous": dict(ls="--", mfc="white")}
FAMILY_MARKERS = {"centroid": "s", "cnn": "o", "fnd": "^", "emd": "D", "image_l2": "v"}


def _method_style(run, label):
    """(family, readout) of a method from its config entry; a CNN's readout is read from the
    args.json saved next to its checkpoint (cnn/train.py)."""
    import json
    from .data import REPO
    spec = dict(run.config.get("methods", {}).get(label) or {})
    typ = spec.get("type", label)
    family = "emd" if typ.startswith("emd") else typ if typ in FAMILY_COLORS else "other"
    readout = "discrete" if spec.get("discrete") else "continuous"
    if typ == "cnn":
        ck = spec.get("checkpoint")
        ck = ck if isinstance(ck, str) else (ck or [None])[0]
        try:
            args = json.load(open((REPO / ck).parent / "args.json"))
            readout = args.get("representation", "continuous")
        except (OSError, TypeError, ValueError):
            readout = "discrete" if "disc" in label else "continuous"
    return family, readout


def _paired_sigmas(t, methods, ref, n_boot, rng):
    """t: trials at one (pitch, N), all theta_true. Residuals are centred per (theta_true, method)
    so angles pool, then sigma (ddof=1) per method and sigma/sigma_ref, each with a 95% interval
    from resampling the same trials for every method (paired bootstrap)."""
    t = t.copy()
    t["res"] = wrap(t["theta_hat"] - t["theta_true"])
    t["res"] -= t.groupby(["theta_true", "method"])["res"].transform("mean")
    w = t.pivot_table(index=["theta_true", "trial"], columns="method", values="res")
    idx = rng.integers(0, len(w), (n_boot, len(w)))

    def sig(x):  # x: (..., K) with NaN for missing
        n = np.sum(np.isfinite(x), axis=-1)
        return np.sqrt(np.nansum(x ** 2, axis=-1) / np.maximum(n - 1, 1))
    out = {}
    r = w[ref].to_numpy() if ref in w else None
    for m in methods:
        if m not in w:
            continue
        x = w[m].to_numpy()
        s, sb = sig(x), sig(x[idx])
        row = dict(sigma=s, sigma_lo=np.percentile(sb, 2.5), sigma_hi=np.percentile(sb, 97.5),
                   K=int(np.isfinite(x).sum()))
        if r is not None:
            ok = np.isfinite(x) & np.isfinite(r)
            xo, ro = x[ok], r[ok]
            io = rng.integers(0, ok.sum(), (n_boot, ok.sum()))
            rb = sig(xo[io]) / sig(ro[io])
            row.update(ratio=sig(xo) / sig(ro), ratio_lo=np.percentile(rb, 2.5), ratio_hi=np.percentile(rb, 97.5))
        out[m] = row
    return out


@figure("methods")
def methods_compare(run, methods=None, ref="centroid_disc", ratio_range=(0.85, 1.15), n_boot=1000):
    """All methods on one set of axes, one row per pitch: sigma_theta vs N (log-log), Theta =
    sigma*sqrt(N) vs N (flat = ideal 1/sqrt(N) scaling, lower = fewer events needed), and
    sigma / sigma_ref with paired-bootstrap 95% intervals. Trials are pooled over theta_true.
    Colour = method family, solid = detector-cell readout, dashed = exact displacements.
    Methods whose ratio falls outside ratio_range are named in the ratio panel instead of plotted."""
    import pandas as pd
    t = run.trials
    if t is None:
        raise ValueError("needs trials (per-trial estimates); legacy tables have none")
    if "sweep_var" in t and (t["sweep_var"] == "N").any():
        t = t[t["sweep_var"] == "N"]
    ms = _methods(run.summary, methods)
    pitches = sorted(t["pitch_mm"].unique(), reverse=True)
    rng = np.random.default_rng(0)
    styles = {m: _method_style(run, m) for m in ms}
    fig, axes = plt.subplots(len(pitches), 3, figsize=(16, 4.6 * len(pitches)), squeeze=False)
    for row, p in zip(axes, pitches):
        tp = t[t.pitch_mm == p]
        stats = pd.DataFrame([dict(N=N, method=m, **v) for N, g in tp.groupby("N")
                              for m, v in _paired_sigmas(g, ms, ref, n_boot, rng).items()])
        theta0 = run.summary.loc[run.summary.pitch_mm == p, "theta0"].median() \
            if _has(run.summary, "theta0") else None
        off_scale, theta_span = [], []
        for m in ms:
            d = stats[stats.method == m].sort_values("N")
            if d.empty:
                continue
            fam, readout = styles[m]
            c = FAMILY_COLORS.get(fam, "#52514e")
            st = READOUT_STYLE[readout]
            kw = dict(color=c, ls=st["ls"], marker=FAMILY_MARKERS.get(fam, "o"), ms=6, lw=2,
                      mfc=st["mfc"] or c, mew=1.6, label=f"{m}  ({readout})")
            sq = np.sqrt(d["N"])
            row[0].plot(d["N"], d["sigma"], **kw)
            lo, hi = ratio_range
            in_range = m == ref or "ratio" not in d or d["ratio"].between(lo, hi).any()
            if in_range:  # Theta and ratio panels zoom on the competitive methods
                row[1].errorbar(d["N"], d["sigma"] * sq, yerr=[(d["sigma"] - d["sigma_lo"]) * sq,
                                (d["sigma_hi"] - d["sigma"]) * sq], capsize=3, **kw)
                theta_span += [(d["sigma_lo"] * sq).min(), (d["sigma_hi"] * sq).max()]
                if m != ref and "ratio" in d:
                    row[2].errorbar(d["N"], d["ratio"], yerr=[d["ratio"] - d["ratio_lo"], d["ratio_hi"] - d["ratio"]],
                                    capsize=3, **kw)
            else:
                off_scale.append((m, (d["sigma"] * sq).min(), (d["sigma"] * sq).max(),
                                  d["ratio"].min(), d["ratio"].max()))
        if theta0 is not None and np.isfinite(theta0):
            Ns = np.array(sorted(tp["N"].unique()), float)
            row[0].plot(Ns, theta0 / np.sqrt(Ns), color="#52514e", ls=":", lw=1.2,
                        label=r"$\Theta_{0,\rm cut}/\sqrt{N}$")
            row[1].axhline(theta0, color="#52514e", ls=":", lw=1.2, label=r"$\Theta_{0,\rm cut}$")
        row[2].axhline(1.0, color="#52514e", lw=1.2)
        row[2].set_ylim(*ratio_range)
        if theta_span:
            a, b = min(theta_span), max(theta_span + ([theta0] if theta0 is not None else []))
            row[1].set_ylim(a - 0.08 * (b - a), b + 0.08 * (b - a))
        for m, t_lo, t_hi, r_lo, r_hi in off_scale:
            for ax, txt in ((row[1], f"{t_lo:.0f}-{t_hi:.0f}"), (row[2], f"{r_lo:.2f}-{r_hi:.2f}")):
                ax.text(0.02, 0.03, f"off scale: {m} {txt}", transform=ax.transAxes,
                        fontsize=8.5, color="#52514e")
        K = int(stats["K"].min()) if not stats.empty else 0
        row[0].set(xscale="log", yscale="log", ylabel=r"$\sigma_\theta$ [deg]",
                   title=f"angular error ($\\Delta x = {p:g}$ mm, K = {K} per N)")
        row[1].set(xscale="log", ylabel=r"$\Theta = \sigma_\theta\sqrt{N}$ [deg]",
                   title=r"$\Theta$: lower = fewer events needed")
        row[2].set(xscale="log", ylabel=r"$\sigma_\theta$ / $\sigma_\theta$ of " + ref,
                   title=f"ratio to {ref} (95% CI)")
        for ax in row:
            ax.set_xlabel(r"usable events $N$")
        row[0].legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return fig


@figure("angle")
def angle(run, methods=None, n_boot=1000):
    """Orientation dependence: everything vs theta_true, one row per (pitch, N).
    Theta (95% CI); sigma / sigma of the centroid with the same readout (paired 95% CI); and the
    mean residual minus that centroid's, per trial (paired). Since the cell lattice repeats every
    90 deg and is mirror-symmetric, 0-45 deg covers every orientation. If results/crlb/crlb_angle.csv
    exists (python -m dirreco.fisher --theta ...), the cell-hop and continuous CRLB are drawn too."""
    import pandas as pd
    from .data import REPO
    t = run.trials
    if t is None:
        raise ValueError("needs trials (per-trial estimates); legacy tables have none")
    ms = _methods(run.summary, methods)
    styles = {m: _method_style(run, m) for m in ms}
    refs = {ro: m for m, (fam, ro) in styles.items() if fam == "centroid"}
    crlb_path = REPO / "results" / "crlb" / "crlb_angle.csv"
    crlb = pd.read_csv(crlb_path) if crlb_path.exists() else None
    rng = np.random.default_rng(0)
    keys = sorted({(p, N) for p, N in zip(t["pitch_mm"], t["N"])}, key=lambda k: (-k[0], k[1]))
    fig, axes = plt.subplots(len(keys), 3, figsize=(16, 4.4 * len(keys)), squeeze=False)
    for row, (p, N) in zip(axes, keys):
        tp = t[(t.pitch_mm == p) & (t.N == N)].copy()
        tp["res"] = wrap(tp["theta_hat"] - tp["theta_true"])
        res = tp.pivot_table(index=["theta_true", "trial"], columns="method", values="res")
        rows = []
        for th, g in tp.groupby("theta_true"):
            for ro, ref in refs.items():
                group = [m for m in ms if styles[m][1] == ro]
                for m, v in _paired_sigmas(g, group, ref, n_boot, rng).items():
                    d = (res.loc[th, m] - res.loc[th, ref]).dropna()
                    rows.append(dict(theta=th, method=m, ref=ref, bias=d.mean(),
                                     bias_se=d.std(ddof=1) / np.sqrt(max(len(d), 1)), **v))
        st = pd.DataFrame(rows)
        sq = np.sqrt(N)
        for m in ms:
            d = st[st.method == m].sort_values("theta")
            if d.empty:
                continue
            fam, ro = styles[m]
            c = FAMILY_COLORS.get(fam, "#52514e")
            sty = READOUT_STYLE[ro]
            kw = dict(color=c, ls=sty["ls"], marker=FAMILY_MARKERS.get(fam, "o"), ms=6, lw=2,
                      mfc=sty["mfc"] or c, mew=1.6, capsize=3, label=f"{m}  ({ro})")
            row[0].errorbar(d["theta"], d["sigma"] * sq, yerr=[(d["sigma"] - d["sigma_lo"]) * sq,
                            (d["sigma_hi"] - d["sigma"]) * sq], **kw)
            if fam != "centroid":
                row[1].errorbar(d["theta"], d["ratio"], yerr=[d["ratio"] - d["ratio_lo"], d["ratio_hi"] - d["ratio"]],
                                **{**kw, "label": f"{m} / {d['ref'].iloc[0]}"})
                row[2].errorbar(d["theta"], d["bias"], yerr=1.96 * d["bias_se"],
                                **{**kw, "label": f"{m} − {d['ref'].iloc[0]}"})
        if crlb is not None:
            cb = crlb[crlb.pitch_mm == p].sort_values("theta")
            if not cb.empty:
                row[0].plot(cb["theta"], cb["crlb_disc"], color="#52514e", ls="-", lw=1.2, label="CRLB cell-hop")
                row[0].plot(cb["theta"], cb["crlb_cont"], color="#52514e", ls=":", lw=1.2, label="CRLB continuous")
        row[1].axhline(1.0, color="#52514e", lw=1.2)
        row[2].axhline(0.0, color="#52514e", lw=1.2)
        K = int(st["K"].min()) if not st.empty else 0
        row[0].set(ylabel=r"$\Theta = \sigma_\theta\sqrt{N}$ [deg]",
                   title=f"$\\Delta x = {p:g}$ mm, N = {N:g}, K = {K} per angle")
        row[1].set(ylabel=r"$\sigma_\theta$ / $\sigma_\theta$ of same-readout centroid",
                   title="ratio (paired 95% CI; < 1 beats the centroid)")
        row[2].set(ylabel="mean residual − centroid's [deg]", title="relative bias (paired 95% CI)")
        for ax in row:
            ax.set_xlabel(r"$\theta_{\rm true}$ [deg] (0 = cloud along a lattice axis)")
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


@figure("scaling")
def scaling(run, methods=None, ref="centroid_disc"):
    """(a) sigma_theta vs occupancy rho = N/G^2; (b) Theta = sigma sqrt(N) vs rho."""
    s = run.summary
    s = s[s["theta_true"] == s["theta_true"].iloc[0]]
    ms = _methods(s, methods)
    if ref and ref not in ms and ref in set(s.method):
        ms = [ref] + ms
    style = {m: STYLES[i % len(STYLES)] for i, m in enumerate(ms)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    for p in _pitches(s):
        for m in ms:
            sub = s[(s.pitch_mm == p) & (s.method == m)].sort_values("rho")
            if sub.empty:
                continue
            mk, ls = style[m]
            ls = ":" if m == ref else ls
            a1.plot(sub["rho"], sub["sigma"], marker=mk, ls=ls, color=pcolor(p), ms=4.5, lw=1.5)
            a2.plot(sub["rho"], sub["Theta"], marker=mk, ls=ls, color=pcolor(p), ms=4.5, lw=1.2, alpha=0.9)
        if _has(s[s.pitch_mm == p], "theta0"):
            a2.axhline(s[s.pitch_mm == p]["theta0"].iloc[0], color=pcolor(p), lw=1, alpha=0.6)
    if not _has(s, "theta0"):
        a2.axhline(LEGACY_THETA0, color="k", lw=1.2)
    for ax in (a1, a2):
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set(xscale="log", yscale="log", xlabel=r"occupancy $\rho = N/G^2$")
    a1.axhline(1.0, color="gray", ls=":", lw=1.2)
    a1.set(ylabel=r"$\sigma_\theta$ [deg]", title=r"(a) resolution vs occupancy")
    a2.set(ylabel=r"$\Theta = \sigma_\theta\sqrt{N}$ [deg$\sqrt{\rm events}$]",
           title=r"(b) normalised resolution (lines: $\Theta_0$)")
    extra = [Line2D([], [], color="k", marker=style[m][0], ls=(":" if m == ref else style[m][1]), label=m)
             for m in ms]
    h = [Line2D([], [], color=pcolor(p), lw=2, label=plabel(p)) for p in _pitches(s)] + extra
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), fontsize=9)
    return fig


CRLB_KIND_COLORS = {"centroid": "#2a78d6", "emd_sliced": "#eb6834", "emd_exact": "#1baf7a",
                    "emd_sinkhorn": "#6250d6"}
CRLB_KIND_MARKERS = {"centroid": "s", "emd_sliced": "o", "emd_exact": "D", "emd_sinkhorn": "^"}


@figure("crlb")
def crlb(run, methods=None, theta=None):
    """Theta = sigma*sqrt(N) vs N against the Cramer-Rao bound, like for like. One row per pitch;
    left: continuous readout (true r_xy), right: discrete (cell hops). Horizontal line: the CRLB
    of that readout (results/crlb/crlb.csv, python -m dirreco.fisher); right axis: efficiency
    eta = (CRLB/Theta)^2. Colour = estimator, solid/filled = split-half template, dashed/hollow =
    large template (ref_mult > 1). Values below the bound at small N point to bias, not skill."""
    import pandas as pd
    from .data import REPO
    s = run.summary
    th = s["theta_true"].iloc[0] if theta is None else float(theta)
    s = s[np.isclose(s["theta_true"], th)]
    ms = _methods(s, methods)
    path = REPO / "results" / "crlb" / "crlb.csv"
    bound = pd.read_csv(path) if path.exists() else None
    specs = run.config.get("methods", {})

    def kind(m):
        typ = dict(specs.get(m) or {}).get("type", m)
        return typ if typ in CRLB_KIND_COLORS else "other"

    def mult(m):
        return int(dict(specs.get(m) or {}).get("ref_mult", 1))

    readout = {m: _method_style(run, m)[1] for m in ms}
    pitches = _pitches(s)
    fig, axes = plt.subplots(len(pitches), 2, figsize=(12.5, 3.9 * len(pitches)), squeeze=False, sharex=True)
    for row, p in zip(axes, pitches):
        for ax, ro, col in ((row[0], "continuous", "crlb_cont"), (row[1], "discrete", "crlb_disc")):
            for m in [m for m in ms if readout[m] == ro]:
                sub = s[(s.pitch_mm == p) & (s.method == m)].sort_values("N")
                if sub.empty:
                    continue
                k, big = kind(m), mult(m) > 1
                c = CRLB_KIND_COLORS.get(k, "#7f7f7f")
                ax.errorbar(sub["N"], sub["Theta"], yerr=sub["Theta_err"], color=c,
                            marker=CRLB_KIND_MARKERS.get(k, "v"), ms=5, lw=1.6, capsize=2,
                            ls="--" if big else "-", mfc="white" if big else c)
            cb = np.nan
            if bound is not None:
                r = bound[np.isclose(bound.pitch_mm, p)]
                cb = float(r[col].iloc[0]) if not r.empty and col in r else np.nan
            if np.isfinite(cb):
                ax.axhline(cb, color="#52514e", lw=1.4)
                sec = ax.secondary_yaxis("right", functions=(lambda t, cb=cb: (cb / np.maximum(t, 1e-9)) ** 2,
                                                             lambda e, cb=cb: cb / np.sqrt(np.maximum(e, 1e-9))))
                sec.set_ylabel(r"$\eta = (\Theta_{\rm CRLB}/\Theta)^2$", fontsize=9)
            ax.set(xscale="log", yscale="log",
                   title=f"{plabel(p)}, {ro} readout" + ("" if np.isfinite(cb) else " (CRLB n/a)"))
            ax.set_ylabel(r"$\Theta = \sigma_\theta\sqrt{N}$ [deg$\sqrt{\rm ev}$]")
    for ax in axes[-1]:
        ax.set_xlabel(r"usable events $N$")
    h = [Line2D([], [], color=CRLB_KIND_COLORS.get(kind(m), "#7f7f7f"), marker=CRLB_KIND_MARKERS.get(kind(m), "v"),
                ls="--" if mult(m) > 1 else "-", mfc="white" if mult(m) > 1 else None, label=m) for m in ms]
    h.append(Line2D([], [], color="#52514e", lw=1.4, label="CRLB (readout of the column)"))
    fig.suptitle(rf"Efficiency vs the Cramer-Rao bound, $\theta_{{\rm true}} = {th:g}^\circ$", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 5), fontsize=8.5)
    return fig


@figure("smearing")
def smearing(run, methods=None):
    """sigma_theta vs Gaussian position smearing, one panel per pitch."""
    s = run.summary
    xcol = next(c for c in s.columns if c.startswith("aug_smear"))
    ms = _methods(s, methods)
    ps = _pitches(s)
    fig, axes = plt.subplots(1, len(ps), figsize=(6 * len(ps), 4.8), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], ps):
        for i, m in enumerate(ms):
            sub = s[(s.pitch_mm == p) & (s.method == m)].sort_values(xcol)
            mk, ls = STYLES[i % len(STYLES)]
            ax.errorbar(sub[xcol], sub["sigma"], yerr=sub["sigma_err"], marker=mk, ls=ls,
                        color=f"C{i}", ms=5, lw=1.8, capsize=2, label=m)
        ax.axhline(1.0, color="gray", ls=":", lw=1.2)
        ax.set(title=plabel(p), xlabel=r"position smearing $\sigma_{\rm smear}$ [mm]")
    axes[0][0].set_ylabel(r"$\sigma_\theta$ [deg]")
    axes[0][0].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig


@figure("phase3d")
def phase3d(run, method="centroid_disc"):
    """3D surface log sigma_theta(dx, N) for one method."""
    s = run.summary
    s = s[(s.method == method) & (s["theta_true"] == s["theta_true"].iloc[0])]
    pv = s.pivot_table(index="N", columns="pitch_mm", values="sigma").dropna(how="all").dropna(axis=1, how="all")
    X, Y = np.meshgrid(np.log10(pv.columns.values), np.log10(pv.index.values))
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, np.log10(pv.values), cmap="viridis", lw=0.5, edgecolors="k", alpha=0.9)
    ax.view_init(elev=24, azim=-215)
    ax.set(xlabel=r"$\log_{10}\Delta x$ [mm]", ylabel=r"$\log_{10}N$", zlabel=r"$\log_{10}\sigma_\theta$ [deg]",
           title=rf"$\sigma_\theta(\Delta x, N)$: {method}")
    fig.colorbar(surf, ax=ax, orientation="horizontal", shrink=0.55, pad=0.08,
                 label=r"$\log_{10}\sigma_\theta$ [deg]")
    return fig


@figure("full360")
def full360(run, method="fnd_argmin", bin_deg=15.0):
    """Linearity (mean estimate vs truth) and isotropy (sigma in angular sectors)."""
    s = run.summary
    s = s[s.method == method]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})
    a1.plot([-180, 180], [-180, 180], "k--", lw=1, label="identity")
    edges = np.arange(-180, 180 + bin_deg, bin_deg)
    centers = 0.5 * (edges[1:] + edges[:-1])
    for p in _pitches(s):
        sub = s[s.pitch_mm == p].sort_values("theta_true")
        t, m = sub["theta_true"].values.astype(float), sub["mean"].values.astype(float)
        jump = np.where(np.abs(np.diff(m)) > 150)[0] + 1
        a1.plot(np.insert(t, jump, np.nan), np.insert(m, jump, np.nan), color=pcolor(p), lw=1.6, label=plabel(p))
        b = np.digitize(sub["theta_true"], edges) - 1
        g = sub.groupby(b)["sigma"]
        idx = g.mean().index
        a2.errorbar(centers[idx], g.mean(), yerr=g.std() / np.sqrt(g.count()), fmt="o-",
                    color=pcolor(p), ms=4.5, capsize=3, lw=1.3)
    a1.set(ylabel=r"mean $\hat\theta$ [deg]", title=f"Full-circle response: {method}")
    a1.legend(fontsize=9, loc="upper left")
    a2.axhline(1.0, color="gray", ls=":", lw=1.2)
    a2.set(xlabel=r"$\theta_{\rm true}$ [deg]", ylabel=rf"$\sigma_\theta$ [deg] ({bin_deg:g}° sectors)",
           xlim=(-185, 185))
    return fig


@figure("pair")
def pair(run, a="fnd_parab", b="fnd_sine", N=None):
    """Per-trial estimates of two methods on the same samples: scatter + |difference|."""
    t = run.trials
    if t is None:
        raise ValueError("pair figure needs per-trial data (trials.csv)")
    N = N or (10000 if (t.N == 10000).any() else t.N.max())
    t = t[t.N == N]
    keys = [c for c in ("pitch_mm", "N", "theta_true", "trial") if c in t]
    w = t.pivot_table(index=keys, columns="method", values="theta_hat").reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    lim = np.nanmax(np.abs(w[[a, b]].values)) * 1.1
    a1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    for p in _pitches(w):
        sub = w[w.pitch_mm == p]
        a1.scatter(sub[a], sub[b], s=22, color=pcolor(p), alpha=0.8, label=plabel(p))
        a2.hist(np.abs(wrap(sub[a] - sub[b])), bins=20, histtype="step", lw=1.8, color=pcolor(p), label=plabel(p))
    a1.set(xlabel=rf"$\hat\theta$ {a} [deg]", ylabel=rf"$\hat\theta$ {b} [deg]", title=f"same samples, N={N:,}")
    a2.set(xlabel=rf"$|\hat\theta_{{a}} - \hat\theta_{{b}}|$ [deg]", ylabel="trials", title=f"{a} vs {b}")
    a1.legend(fontsize=9)
    fig.tight_layout()
    return fig


@figure("landscape")
def landscape(run, methods=None):
    """Min-max normalised loss L(phi) (mean +/- std over trials), one panel per pitch."""
    L = run.landscape
    if L is None:
        raise ValueError("landscape figure needs landscape.csv (config kind: landscape)")
    ms = _methods(L, methods)
    ps = _pitches(L)
    fig, axes = plt.subplots(1, len(ps), figsize=(6.2 * len(ps), 4.8), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], ps):
        for i, m in enumerate(ms):
            g = L[(L.pitch_mm == p) & (L.method == m)].groupby("phi")["loss"]
            mu, sd = g.mean(), g.std()
            rng = (mu.max() - mu.min()) or 1.0
            y, e = (mu - mu.min()) / rng, sd / rng
            mk, ls = STYLES[i % len(STYLES)]
            ax.plot(mu.index, y, marker=mk, ls=ls, color=f"C{i}", ms=4, lw=1.8, label=m)
            ax.fill_between(mu.index, y - e, y + e, color=f"C{i}", alpha=0.15)
        N = int(L[L.pitch_mm == p]["N"].iloc[0])
        ax.set(title=f"{plabel(p)}, N={N:,}", xlabel=r"trial rotation $\phi$ [deg]")
    axes[0][0].set_ylabel(r"normalised loss $\tilde L(\phi)$")
    axes[0][0].legend(fontsize=9)
    fig.tight_layout()
    return fig


@figure("hits")
def hits(run=None, pitches=(150, 50, 5), n_events=40, width=450.0):
    """Detector face with cell grid, prompt-cell veto circle and example event vectors."""
    from .data import Geometry, load_events
    ev = load_events(max_events=n_events)
    fig, axes = plt.subplots(1, len(pitches), figsize=(5.3 * len(pitches), 5.5))
    for ax, p in zip(axes, pitches):
        g = Geometry(p, width)
        for e in g.edges:
            ax.axvline(e, color="gray", lw=0.5, alpha=0.6)
            ax.axhline(e, color="gray", lw=0.5, alpha=0.6)
        ax.add_patch(plt.Circle((0, 0), g.cut_radius, fill=False, ec="#6250d6", ls="--", lw=1.2))
        for v, c in zip(ev.vtx, ev.cap):
            ax.annotate("", xy=c[:2], xytext=v[:2], arrowprops=dict(arrowstyle="->", lw=0.7, alpha=0.6))
        ax.plot(ev.vtx[:, 0], ev.vtx[:, 1], ".", color="#e34948", ms=5)
        ax.plot(ev.cap[:, 0], ev.cap[:, 1], ".", color="#2a78d6", ms=5)
        ax.set(xlim=(-240, 240), ylim=(-240, 240), aspect="equal", xlabel="x [mm]",
               title=rf"{plabel(p)} (${g.grid_size}\times{g.grid_size}$)")
        ax.grid(False)
    axes[0].set_ylabel("y [mm]")
    fig.legend(handles=[Line2D([], [], marker=".", color="#e34948", ls="none", ms=8, label=r"prompt vertex ($e^+$)"),
                        Line2D([], [], marker=".", color="#2a78d6", ls="none", ms=8, label="capture ($n$)"),
                        Line2D([], [], color="k", lw=1, label="displacement"),
                        Line2D([], [], color="#6250d6", ls="--", label=r"veto $r \leq \Delta x/\sqrt{2}$")],
               loc="upper center", ncol=4, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ----------------------------------------------------------------------------- driver
def make_plots(run, only=None, **overrides):
    """Render the figures listed in run.config['plots'] (or just `only`) to run.plot_dir."""
    entries = list(run.config.get("plots", []))
    if only:
        entries = [e for e in entries if only in (e["figure"], e.get("name"))] or [{"figure": only}]
    run.plot_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for e in entries:
        e = dict(e)
        fn = FIGURES[e.pop("figure")]
        name = e.pop("name", fn.__name__)
        params = inspect.signature(fn).parameters
        e.update({k: v for k, v in overrides.items() if v is not None and k in params})
        try:
            fig = fn(run, **e)
        except Exception as exc:  # keep going; one bad figure should not block the rest
            print(f"[plot] {name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        fig.savefig(run.plot_dir / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
        out.append(run.plot_dir / f"{name}.png")
        print(f"[plot] {out[-1]}")
    return out
