"""Data-driven Cramer-Rao bound on the direction, from the real MC pool (no generator needed).

    python -m dirreco.fisher                      # -> results/crlb/crlb.csv, crlb.txt
    python -m dirreco.fisher --selftest           # Gaussian check: CRLB must equal theta0

A rotation of the neutrino direction by theta rotates the displacement distribution p(r_xy). The
per-event Fisher information is I = E[(d/dtheta log p)^2], and no unbiased estimator using N
usable events can beat sigma_theta >= 1/sqrt(N I), i.e. Theta = sigma*sqrt(N) >= Theta_CRLB =
deg(1/sqrt(I)). This is the bound README "Open physics issues" 3 asks for: theta0 = std(r_y)/mean(r_x)
is only the centroid's Theta, and equals the CRLB only if p is Gaussian.

Two readouts, both restricted to usable events (|r_xy| > pitch/sqrt 2):
  continuous  r_xy measured exactly. Rotation shifts phi at fixed rho, so I = E[(d_phi log p(phi|rho))^2].
              p(phi|rho) is a Fourier series (order K) fitted per rho bin. The usable cut is a
              circle, so it does not depend on theta and only selects rho.
  discrete    only the cell hop (cell of capture - cell of vertex) is seen, as in centroid_disc /
              the detector. I = sum_k (dP_k/dtheta)^2 / P_k. With a uniform vertex cell phase,
              P_k is a cloud-in-cell deposit of the events and dP_k/dtheta is analytic (see
              discrete_info), so there are no finite differences. Must approach `continuous` as
              pitch -> 0, and can never be below it.
Squared derivatives are formed as products of two independent half-samples, so Poisson noise does
not bias them upward. The K columns (Fourier order) show the continuous estimate's convergence.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from .data import REPO, load_events

OUT = REPO / "results" / "crlb"


def _fourier_info(phi, rho, w_edges, K, n_grid=720):
    """Per-event Fisher information of phi | rho, Fourier order K, halves cross-multiplied."""
    g = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    k = np.arange(1, K + 1)
    ck, sk = np.cos(np.outer(k, g)), np.sin(np.outer(k, g))      # (K, G)
    half = np.arange(len(phi)) % 2
    which = np.digitize(rho, w_edges) - 1
    total, n_used = 0.0, 0
    for b in range(len(w_edges) - 1):
        m = which == b
        n = int(m.sum())
        if n < 200:
            continue
        coef = {}
        for h in (0, 1, None):
            mm = m if h is None else (m & (half == h))
            ph = phi[mm][:, None] * k[None, :]
            coef[h] = (np.cos(ph).mean(0), np.sin(ph).mean(0))      # a_k, b_k
        a, s = coef[None]
        p = (1 + 2 * (a @ ck + s @ sk)) / (2 * np.pi)
        p = np.maximum(p, 1e-3 / (2 * np.pi))
        dp = {h: 2 * ((k * coef[h][1]) @ ck - (k * coef[h][0]) @ sk) / (2 * np.pi) for h in (0, 1)}
        info = np.sum(dp[0] * dp[1] / p) * (2 * np.pi / n_grid)
        total += n * info
        n_used += n
    return total / max(n_used, 1)


def continuous_info(disp, cut, K=8, rho_bin=5.0):
    rho = np.hypot(disp[:, 0], disp[:, 1])
    m = rho > cut
    phi = np.arctan2(disp[m, 1], disp[m, 0])
    edges = np.arange(cut, rho[m].max() + rho_bin, rho_bin)
    return _fourier_info(phi, rho[m], edges, K)


def discrete_info(disp, pitch, cut, min_count=100):
    """Fisher info of the cell-hop distribution P_k(theta), exact in the vertex phase.

    With the vertex cell phase uniform (true here: vertices are uniform over a 40 m cube), the hop
    along x is floor(s_x) or floor(s_x)+1 with probabilities 1-f_x and f_x, where s = r/pitch and
    f = frac(s); x and y are independent. So P_k(theta) is a cloud-in-cell deposit of the
    rotated events, and dP_k/dtheta follows analytically from ds/dtheta = J s = (-s_y, s_x).
    No finite differences, no phase noise; the two half-samples only remove the bias of the
    squared derivative.
    """
    m = np.hypot(disp[:, 0], disp[:, 1]) > cut
    sx, sy = disp[m, 0] / pitch, disp[m, 1] / pitch
    nx, ny = np.floor(sx), np.floor(sy)
    fx, fy = sx - nx, sy - ny
    dfx, dfy = -sy, sx                                  # d f / d theta at theta = 0
    H = int(max(np.abs(nx).max(), np.abs(ny).max())) + 2
    S = 2 * H + 1
    ix, iy = (nx + H).astype(np.int64), (ny + H).astype(np.int64)
    half = np.arange(len(sx)) % 2
    P = np.zeros(S * S)
    dP = [np.zeros(S * S), np.zeros(S * S)]
    for a in (0, 1):
        wx, dwx = (fx, dfx) if a else (1 - fx, -dfx)
        for b in (0, 1):
            wy, dwy = (fy, dfy) if b else (1 - fy, -dfy)
            k = (ix + a) * S + (iy + b)
            P += np.bincount(k, wx * wy, minlength=S * S)
            dw = dwx * wy + wx * dwy
            for h in (0, 1):
                sel = half == h
                dP[h] += np.bincount(k[sel], dw[sel], minlength=S * S)
    n = len(sx)
    P /= n
    dP = [dP[h] / (half == h).sum() for h in (0, 1)]
    ok = P > 0
    contrib = np.zeros_like(P)
    contrib[ok] = dP[0][ok] * dP[1][ok] / P[ok]
    info = float(contrib.sum())
    # The ratio estimator is only unbiased when every bin that matters is well populated. At small
    # pitch (many sparse hop bins) it fails, even below the continuous bound (impossible), so
    # refuse when > 1% of the information sits in bins with < min_count effective events.
    sparse = np.abs(contrib[P * n < min_count]).sum() / max(np.abs(contrib).sum(), 1e-300)
    return info if (info > 0 and sparse < 0.01) else float("nan")


def theta_deg(info):
    return float(np.degrees(1.0 / np.sqrt(info))) if info > 0 else float("nan")


def run(disp, pitches, Ks, log):
    rows = []
    for p in pitches:
        cut = p / np.sqrt(2.0) if p > 0 else 0.0
        u = np.hypot(disp[:, 0], disp[:, 1]) > cut
        du = disp[u]
        th0 = float(np.degrees(np.std(du[:, 1]) / np.mean(du[:, 0])))
        t0 = time.time()
        cont = {K: theta_deg(continuous_info(disp, cut, K)) for K in Ks}
        disc = theta_deg(discrete_info(disp, p, cut)) if p > 0 else float("nan")
        row = dict(pitch_mm=p, eff=float(u.mean()), theta0=th0,
                   **{f"crlb_cont_K{K}": v for K, v in cont.items()})
        row["crlb_cont"] = cont[8] if 8 in Ks else cont[max(Ks)]   # K=12 is noisier, K=4 may truncate
        row["crlb_disc"] = disc
        row["eta_centroid_cont"] = (row["crlb_cont"] / th0) ** 2
        rows.append(row)
        log(f"pitch {p:6g}  eff {row['eff']:.4f}  theta0 {th0:7.2f} | CRLB cont "
            + " ".join(f"K{K}:{v:7.2f}" for K, v in cont.items())
            + f" | disc {disc:7.2f}"
            + f"  ({time.time() - t0:.0f} s)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitches", default="0,1,2,5,10,25,50,100,150", help="mm; 0 = no cut, continuous only")
    ap.add_argument("--K", default="4,8,12", help="Fourier orders (convergence check)")
    ap.add_argument("--max-events", type=int, help="default: the whole pool")
    ap.add_argument("--selftest", action="store_true",
                    help="Gaussian displacements with the real mean/width: CRLB_cont must equal theta0")
    a = ap.parse_args()
    pitches = [float(x) for x in a.pitches.split(",")]
    Ks = [int(x) for x in a.K.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "selftest" if a.selftest else "crlb"
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    ev = load_events(max_events=a.max_events)
    disp = ev.disp
    if a.selftest:
        rng = np.random.default_rng(0)
        sig = float(np.std(disp[:, 1]))
        disp = rng.normal(0.0, sig, size=disp.shape)
        disp[:, 0] += float(np.mean(ev.disp[:, 0]))
        log(f"SELFTEST: Gaussian r_xy, mean ({np.mean(disp[:, 0]):.2f}, 0) mm, sigma {sig:.2f} mm. "
            "Expect CRLB_cont == theta0 at pitch 0, and CRLB_disc -> CRLB_cont as pitch -> 0.")
    log(f"{len(disp):,} events. Theta values in deg*sqrt(N_usable); eta_centroid = (CRLB/theta0)^2")
    rows = run(disp, pitches, Ks, log)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{tag}.csv", index=False)
    log("\n" + df[["pitch_mm", "eff", "theta0", "crlb_cont", "crlb_disc", "eta_centroid_cont"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    Path(OUT / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / (tag + '.csv')}")


if __name__ == "__main__":
    main()
