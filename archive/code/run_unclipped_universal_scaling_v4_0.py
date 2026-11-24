#!/usr/bin/env python3
"""
run_unclipped_universal_scaling_v4_0.py
=======================================
Full unclipped 360-degree circle search with 3-point parabolic sub-degree refinement:
  - Eliminates artificial +/-15 deg clipping artifacts.
  - Sweeps dx in [150, 50, 25, 10, 5, 1] mm across N in [100, 500, 1k, 5k, 10k, 50k].
  - Evaluates Discrete Centroid, Continuous FND, and 2D Spatial EMD (W1).
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

PITCHES = [150.0, 50.0, 25.0, 10.0, 5.0, 1.0]
N_VALS = [100, 500, 1000, 5000, 10000, 50000]
K_TRIALS = 20
DETECTOR_WIDTH = 450.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/stats_unclipped_universal_scaling_{TS}.txt"

with open(OUT_FILE, "w") as f:
    f.write("# dx_mm  N  rho  trials  Cent_disc_std  FND_para_std  EMD_para_std  usable_eff\n")

print(f"[{datetime.now().strftime('%T')}] Ingesting master Monte Carlo truth coordinates...")
proc = DataProcessor(dx=5, gs=89, n=100000)
proc.readData()

raw_dx = proc.coords[:, 0]
raw_dy = proc.coords[:, 1]
r_sq = raw_dx**2 + raw_dy**2
n_pool = len(raw_dx)

# Full 360-degree search grid (no boundary clipping)
SEARCH_ANGLES = np.arange(-180, 180, 1)
RAD_GRID = np.radians(SEARCH_ANGLES)
COS_GRID = np.cos(RAD_GRID)
SIN_GRID = np.sin(RAD_GRID)

for pitch in PITCHES:
    R = pitch / np.sqrt(2.0)
    usable_mask = r_sq > (R**2)
    eff = np.sum(usable_mask) / float(n_pool)

    dx_use = raw_dx[usable_mask]
    dy_use = raw_dy[usable_mask]

    # Hardware discretized cell centers
    disc_px = np.floor(proc.vertices[usable_mask, 0] / pitch) * pitch + pitch / 2.0
    disc_py = np.floor(proc.vertices[usable_mask, 1] / pitch) * pitch + pitch / 2.0
    disc_nx = np.floor(proc.captures[usable_mask, 0] / pitch) * pitch + pitch / 2.0
    disc_ny = np.floor(proc.captures[usable_mask, 1] / pitch) * pitch + pitch / 2.0
    disc_dx = disc_nx - disc_px
    disc_dy = disc_ny - disc_py

    gs = int(np.round(DETECTOR_WIDTH / pitch))
    G2 = gs**2
    x_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)
    y_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)

    can_fnd = (pitch >= 1.0) and (gs <= 450)

    for N in N_VALS:
        if len(dx_use) < 2 * N:
            continue

        rho = N / float(G2)
        cent_list = []
        fnd_list  = []
        emd_list  = []

        for _ in range(K_TRIALS):
            idx = np.random.choice(len(dx_use), 2 * N, replace=False)
            idx_meas = idx[N:]
            idx_ref  = idx[:N]

            # 1. Discrete Hardware Centroid (exact unconstrained arctan2)
            th_c = np.degrees(np.arctan2(np.mean(disc_dy[idx_meas]), np.mean(disc_dx[idx_meas])))
            cent_list.append(th_c)

            xm, ym = dx_use[idx_meas], dy_use[idx_meas]
            xr, yr = dx_use[idx_ref], dy_use[idx_ref]

            # 2. Continuous FND across 360 deg with parabolic interpolation
            if can_fnd and (N <= 20000):
                ref_mat, _, _ = np.histogram2d(xr, yr, bins=[x_edges, y_edges])
                norms = []
                for ca, sa in zip(COS_GRID, SIN_GRID):
                    rx = xm * ca - ym * sa
                    ry = xm * sa + ym * ca
                    rot_mat, _, _ = np.histogram2d(rx, ry, bins=[x_edges, y_edges])
                    norms.append(np.linalg.norm(ref_mat - rot_mat))

                norms = np.array(norms)
                i_min = np.argmin(norms)
                # Circular boundary check for parabolic fit
                im = (i_min - 1) % len(norms)
                ip = (i_min + 1) % len(norms)
                dp, dz, dm = norms[ip], norms[i_min], norms[im]
                denom = dp - 2.0 * dz + dm
                sub_step = -0.5 * (dp - dm) / denom if denom > 0 else 0.0
                th_f = -(SEARCH_ANGLES[i_min] + sub_step)
                # Wrap to [-180, 180]
                th_f = ((th_f + 180.0) % 360.0) - 180.0
                fnd_list.append(th_f)

            # 3. 2D Spatial Optimal Transport (W1 chord proxy) across 360 deg
            if N <= 20000:
                emd_work = []
                for ca, sa in zip(COS_GRID, SIN_GRID):
                    rx = xm * ca - ym * sa
                    ry = xm * sa + ym * ca
                    disp = np.sqrt((rx - xr)**2 + (ry - yr)**2)
                    emd_work.append(np.mean(disp))

                emd_work = np.array(emd_work)
                i_emd = np.argmin(emd_work)
                im = (i_emd - 1) % len(emd_work)
                ip = (i_emd + 1) % len(emd_work)
                dp, dz, dm = emd_work[ip], emd_work[i_emd], emd_work[im]
                denom = dp - 2.0 * dz + dm
                sub_step = -0.5 * (dp - dm) / denom if denom > 0 else 0.0
                th_e = -(SEARCH_ANGLES[i_emd] + sub_step)
                th_e = ((th_e + 180.0) % 360.0) - 180.0
                emd_list.append(th_e)

        s_c = np.std(cent_list)
        s_f = np.std(fnd_list) if len(fnd_list) == K_TRIALS else np.nan
        s_e = np.std(emd_list) if len(emd_list) == K_TRIALS else np.nan

        f_str = f"{s_f:8.3f}" if not np.isnan(s_f) else "     NaN"
        e_str = f"{s_e:8.3f}" if not np.isnan(s_e) else "     NaN"

        line = f"{pitch:6.1f} {N:8d} {rho:10.4f} {K_TRIALS:4d} {s_c:10.3f} {f_str} {e_str} {eff:10.4f}\n"
        with open(OUT_FILE, "a") as f:
            f.write(line)

        print(f"[{datetime.now().strftime('%T')}] dx={pitch:5.1f}mm | N={N:6d} | rho={rho:7.3f} | Cent={s_c:5.2f} | FND={f_str.strip()} | EMD={e_str.strip()}")

print(f"[SUCCESS] Unclipped universal scaling complete: {OUT_FILE}")
