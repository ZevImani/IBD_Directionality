"""Checks for the discrete (cell-hop) and large-template EMD code paths, plus timings.

    python -m dirreco.selftest_emd            (from the repo root; GPU used if present)

0. emd_sliced (numpy and torch paths) equals the old per-angle-sort loss, on and off the
   1-deg direction grid, for n_proj 32 and 36, mean and max.
1. emd_sliced large-template path: c = 1 equals the equal-size numpy path; c = 5 equals
   brute-force 1D W1 (scipy) averaged over the same directions.
2. HopLattice cloud-in-cell templates equal a Monte Carlo over random vertex cell phases.
3. Injection recovery on real MC (theta_true = 10 deg), discrete and continuous, with zoom.
4. Seconds per estimate for each emd_crlb method at the grid's largest N.
"""
import time

import numpy as np

from .data import Geometry, load_events
from .methods import build
from .methods.emd import HopLattice, _directions


def _ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg, flush=True)
    return bool(cond)


def _sliced_per_angle(x, y, xr, yr, phis, n_proj, reduce):
    """The pre-2026-11-26 emd_sliced loss: sorts the measurement again for every (phi, slice)."""
    a = _directions(n_proj)
    P_ref = np.sort(np.cos(a)[:, None] * xr + np.sin(a)[:, None] * yr, axis=1)
    A = a[None, :] - np.radians(phis)[:, None]
    P = np.sort(np.cos(A)[..., None] * x + np.sin(A)[..., None] * y, axis=-1)
    w1 = np.abs(P - P_ref[None]).mean(axis=2)
    return w1.mean(axis=1) if reduce == "mean" else w1.max(axis=1)


def check_sliced_direction_cache(ev, rng):
    print("0. emd_sliced shared measurement sorts == per-angle sorts")
    ok = True
    n = 400
    idx = rng.choice(len(ev), 2 * n, replace=False)
    meas, ref = ev.take(idx[:n]).rotated(23.0), ev.take(idx[n:])
    for n_proj in (32, 36):
        for reduce in ("mean", "max"):
            for scan in ([-180, 175, 5], [27, 37, 1], [-3, 3, 0.7]):    # coarse, zoom, off-grid
                m = build("s", {"type": "emd_sliced", "n_proj": n_proj, "reduce": reduce, "scan": scan})
                x, y, xr, yr = m.clouds(meas, ref, None)
                want = _sliced_per_angle(x, y, xr, yr, m.phis, n_proj, reduce)
                got, got_t = m.loss(meas, ref, None), m._loss_large_ref(x, y, xr, yr)
                d = max(np.abs(got - want).max(), np.abs(got_t - want).max())
                ok &= _ok(d < 1e-9, f"n_proj {n_proj}, {reduce}, scan {scan}: numpy and torch "
                                    f"(c=1) max diff {d:.1e} mm")
    return ok


def check_sliced_large_ref(ev, rng):
    from scipy.stats import wasserstein_distance
    print("1. emd_sliced large-template path")
    ok = True
    n = 300
    idx = rng.choice(len(ev), 6 * n, replace=False)
    meas, ref = ev.take(idx[:n]), ev.take(idx[n:])
    m = build("s", {"type": "emd_sliced", "scan": [-20, 20, 5]})
    x, y, xr, yr = m.clouds(meas, ev.take(idx[n:2 * n]), None)
    a = m.loss(meas, ev.take(idx[n:2 * n]), None)
    b = m._loss_large_ref(x, y, xr, yr)
    ok &= _ok(np.allclose(a, b, rtol=1e-10, atol=1e-9), f"c=1 torch == numpy (max diff {np.abs(a - b).max():.2e} mm)")
    x, y, xr, yr = m.clouds(meas, ref, None)                     # c = 5
    got = m._loss_large_ref(x, y, xr, yr)
    dirs = _directions(m.n_proj)
    want = []
    for phi in m.phis:
        A = dirs - np.radians(phi)
        want.append(np.mean([wasserstein_distance(np.cos(t) * x + np.sin(t) * y,
                                                  np.cos(d) * xr + np.sin(d) * yr)
                             for t, d in zip(A, dirs)]))
    want = np.array(want)
    ok &= _ok(np.allclose(got, want, rtol=1e-9, atol=1e-9), f"c=5 == scipy brute force (max diff {np.abs(got - want).max():.2e} mm)")
    return ok


def check_cic(ev, rng):
    print("2. cloud-in-cell template == Monte Carlo over vertex cell phases")
    ok = True
    for pitch in (150.0, 50.0):
        geom = Geometry(pitch)
        sel = ev.take(np.flatnonzero(geom.usable_mask(ev))[:20000])
        lat = HopLattice(sel, sel, geom)
        for psi in (0.0, 23.0):
            T = lat.templates([psi])[0]
            c, s = np.cos(np.radians(psi)), np.sin(np.radians(psi))
            r = sel.disp @ np.array([[c, s], [-s, c]])           # rows rotated CCW by psi
            acc = np.zeros_like(T)
            M = 50
            for _ in range(M):
                v = rng.uniform(0, pitch, size=r.shape)          # vertex position inside its cell
                hop = (np.floor((v + r) / pitch) - np.floor(v / pitch)).astype(np.int64)
                acc += np.bincount((hop[:, 0] + lat.H) * lat.S + (hop[:, 1] + lat.H),
                                   minlength=lat.S ** 2)
            mc = acc / acc.sum()
            tv = 0.5 * np.abs(mc - T).sum()
            noise = 0.5 * np.sqrt(T * (1 - T) / (M * len(r))).sum() * np.sqrt(2 / np.pi) * 2
            ok &= _ok(tv < 2 * noise + 1e-4, f"pitch {pitch:g} mm, psi {psi:g} deg: total variation "
                      f"{tv:.2e} (MC noise ~{noise:.1e}), sum T = {T.sum():.6f}")
    return ok


def check_recovery(ev, rng, th=10.0, K=6, N=2000):
    print(f"3. injection recovery, theta_true = {th:g} deg, N = {N}, K = {K}")
    specs = {
        "emd_sliced":          {"type": "emd_sliced", "scan": [-180, 175, 5], "zoom": [5, 1]},
        "emd_sliced_t5":       {"type": "emd_sliced", "scan": [-180, 175, 5], "zoom": [5, 1], "ref_mult": 5},
        "emd_sliced_disc":     {"type": "emd_sliced", "discrete": True, "scan": [-180, 175, 5], "zoom": [5, 1]},
        "emd_sliced_disc_t5":  {"type": "emd_sliced", "discrete": True, "scan": [-180, 175, 5], "zoom": [5, 1], "ref_mult": 5},
        "emd_exact_disc":      {"type": "emd_exact", "discrete": True, "scan": [-180, 175, 5], "zoom": [5, 1]},
        "centroid_disc":       {"type": "centroid", "discrete": True},
    }
    ms = {k: build(k, v) for k, v in specs.items()}
    ok = True
    for pitch in (150.0, 50.0):
        geom = Geometry(pitch)
        usable = np.flatnonzero(geom.usable_mask(ev))
        est = {k: [] for k in ms}
        for _ in range(K):
            idx = rng.choice(usable, N * 6, replace=False)
            meas = ev.take(idx[:N]).rotated(th)
            for k, m in ms.items():
                ref = ev.take(idx[N:N + N * m.ref_mult])
                est[k].append(m.estimate(meas, ref, geom))
        for k, v in est.items():
            v = np.array(v)
            se = v.std(ddof=1) / np.sqrt(K)
            ok &= _ok(abs(v.mean() - th) < max(4 * se, 0.5),
                      f"{pitch:5g} mm {k:20s} mean {v.mean():6.2f} +- {se:.2f}  (spread {v.std(ddof=1):.2f})")
    return ok


def check_zoom(ev, rng):
    print("4. zoom (5 deg full circle + 1 deg zoom) == full 1-deg circle scan")
    geom = Geometry(50.0)
    usable = np.flatnonzero(geom.usable_mask(ev))
    full = build("f", {"type": "emd_sliced", "discrete": True, "scan": [-180, 179, 1]})
    zoom = build("z", {"type": "emd_sliced", "discrete": True, "scan": [-180, 175, 5], "zoom": [5, 1]})
    d = []
    for _ in range(5):
        idx = rng.choice(usable, 1000, replace=False)
        meas, ref = ev.take(idx[:500]).rotated(37.0), ev.take(idx[500:])
        d.append(full.estimate(meas, ref, geom) - zoom.estimate(meas, ref, geom))
    return _ok(np.max(np.abs(d)) < 1e-9, f"max |difference| {np.max(np.abs(d)):.1e} deg")


def timings(ev, rng):
    import yaml
    from .data import REPO
    cfg = yaml.safe_load(open(REPO / "configs" / "emd_crlb.yaml"))
    Nmax = max(cfg["sweep"]["N"])
    print(f"5. seconds per estimate (pitch 50 mm, one estimate each)")
    geom = Geometry(50.0)
    usable = np.flatnonzero(geom.usable_mask(ev))
    for N in sorted({300, 1000, 3000, Nmax}):
        for label, spec in cfg["methods"].items():
            m = build(label, spec)
            if not m.applicable(N, geom):
                continue
            idx = rng.choice(usable, N * (1 + m.ref_mult), replace=False)
            meas, ref = ev.take(idx[:N]).rotated(10.0), ev.take(idx[N:])
            m.estimate(meas, ref, geom)                            # warm-up (CUDA init, caches)
            t0 = time.time()
            m.estimate(meas, ref, geom)
            print(f"  N={N:6d} {label:22s} {time.time() - t0:8.3f} s", flush=True)


def main():
    rng = np.random.default_rng(1)
    ev = load_events(max_events=3_000_000)
    res = [check_sliced_direction_cache(ev, rng), check_sliced_large_ref(ev, rng), check_cic(ev, rng), check_recovery(ev, rng), check_zoom(ev, rng)]
    print("ALL PASS" if all(res) else "SOME CHECKS FAILED", flush=True)
    timings(ev, rng)


if __name__ == "__main__":
    main()
