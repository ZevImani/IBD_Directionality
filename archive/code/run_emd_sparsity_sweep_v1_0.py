#!/usr/bin/env python3
"""
run_emd_sparsity_sweep_v1_0.py
==============================
Sweeps pitches dx in [150, 50, 5, 1] mm across N in [100, 500, 1000, 5000, 10000]
to evaluate whether 2D Spatial EMD preserves linear statistical efficiency
in the sparse-matrix regime (rho = N/G^2 < 1) where FND degrades.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

PITCHES = [150.0, 50.0, 5.0, 1.0]
N_VALS = [100, 500, 1000, 5000, 10000]
K_TRIALS = 20
DETECTOR_WIDTH = 450.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/stats_emd_sparsity_sweep_{TS}.txt"

with open(OUT_FILE, "w") as f:
    f.write("# dx_mm  N  rho  trials  Cent_disc_std  FND_para_std  EMD_std  usable_eff\n")

print(f"[{datetime.now().strftime('%T')}] Ingesting coordinate pool...")
proc = DataProcessor(dx=5, gs=89, n=100000)
proc.readData()

raw_dx = proc.coords[:, 0]
raw_dy = proc.coords[:, 1]
r_sq = raw_dx**2 + raw_dy**2
n_pool = len(raw_dx)
search_angles = np.arange(-15, 16, 1)

for pitch in PITCHES:
    R = pitch / np.sqrt(2.0)
    usable_mask = r_sq > (R**2)
    eff = np.sum(usable_mask) / float(n_pool)

    dx_use = raw_dx[usable_mask]
    dy_use = raw_dy[usable_mask]

    # Quantized hardware cell hopping
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

            # 1. Discrete Centroid
            th_c = np.degrees(np.arctan2(np.mean(disc_dy[idx_meas]), np.mean(disc_dx[idx_meas])))
            cent_list.append(th_c)

            # Continuous unbinned vectors
            xm, ym = dx_use[idx_meas], dy_use[idx_meas]
            xr, yr = dx_use[idx_ref], dy_use[idx_ref]

            # 2. Continuous FND with Parabolic Refinement
            ref_mat, _, _ = np.histogram2d(xr, yr, bins=[x_edges, y_edges])
            norms = []
            for phi in search_angles:
                rad = np.radians(phi)
                ca, sa = np.cos(rad), np.sin(rad)
                rx = xm * ca - ym * sa
                ry = xm * sa + ym * ca
                rot_mat, _, _ = np.histogram2d(rx, ry, bins=[x_edges, y_edges])
                norms.append(np.linalg.norm(ref_mat - rot_mat))

            norms = np.array(norms)
            i_min = np.argmin(norms)
            if 0 < i_min < len(norms) - 1:
                dp, dz, dm = norms[i_min + 1], norms[i_min], norms[i_min - 1]
                denom = dp - 2.0 * dz + dm
                sub_step = -0.5 * (dp - dm) / denom if denom > 0 else 0.0
                th_f = -(search_angles[i_min] + sub_step)
            else:
                th_f = -search_angles[i_min]
            fnd_list.append(th_f)

            # 3. 2D Spatial Optimal Transport (Continuous Chord Proxy W1)
            emd_work = []
            for phi in search_angles:
                rad = np.radians(phi)
                ca, sa = np.cos(rad), np.sin(rad)
                rx = xm * ca - ym * sa
                ry = xm * sa + ym * ca
                disp = np.sqrt((rx - xr)**2 + (ry - yr)**2)
                emd_work.append(np.mean(disp))

            emd_work = np.array(emd_work)
            i_emd = np.argmin(emd_work)
            if 0 < i_emd < len(emd_work) - 1:
                dp, dz, dm = emd_work[i_emd + 1], emd_work[i_emd], emd_work[i_emd - 1]
                denom = dp - 2.0 * dz + dm
                sub_step = -0.5 * (dp - dm) / denom if denom > 0 else 0.0
                th_e = -(search_angles[i_emd] + sub_step)
            else:
                th_e = -search_angles[i_emd]
            emd_list.append(th_e)

        s_c = np.std(cent_list)
        s_f = np.std(fnd_list)
        s_e = np.std(emd_list)

        line = f"{pitch:6.1f} {N:8d} {rho:10.4f} {K_TRIALS:4d} {s_c:10.3f} {s_f:10.3f} {s_e:10.3f} {eff:10.4f}\n"
        with open(OUT_FILE, "a") as f:
            f.write(line)

        print(f"[{datetime.now().strftime('%T')}] dx={pitch:5.1f}mm | N={N:6d} | rho={rho:7.3f} | Cent={s_c:5.2f} | FND={s_f:5.2f} | EMD={s_e:5.2f}")

print(f"[SUCCESS] EMD sparsity sweep complete: {OUT_FILE}")
