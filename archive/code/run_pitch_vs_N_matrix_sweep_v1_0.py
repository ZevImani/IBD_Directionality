#!/usr/bin/env python3
"""
run_pitch_vs_N_matrix_sweep_v1_0.py
===================================
Version: v1.0 (2026-10-07)
Author: Max A. A. Dornfest
Description: Sweeps pitch dx in [0.5, 1, 5, 10, 25, 50, 100, 150] mm
             across sample sizes N in [10^2, 10^3, 10^4, 10^5, 10^6].
             Computes:
               1. Centroid Continuous (theoretical limit)
               2. Centroid Discrete (experimental cell-center readout)
               3. Continuous FND (matrix pattern matching with split-half bootstrap)
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

PITCHES = [0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0]
N_VALS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
K_TRIALS = 5
THETA_TRUE = 10.0
DETECTOR_WIDTH = 450.0  # mm

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/stats_dx_vs_N_matrix_{TS}.txt"

with open(OUT_FILE, "w") as f:
    f.write("# dx_mm  N  trials  usable_eff  grid_size  Cent_cont_std  Cent_disc_std  FND_std\n")

print(f"[{datetime.now().strftime('%T')}] Loading master coordinate pool...")
proc_base = DataProcessor(dx=5, gs=89, n=10000)
proc_base.readData()

raw_dx = proc_base.coords[:, 0]
raw_dy = proc_base.coords[:, 1]
r_sq = raw_dx**2 + raw_dy**2
n_pool = len(raw_dx)
print(f"[{datetime.now().strftime('%T')}] Master pool ready: {n_pool:,} interaction pairs.")

for pitch in PITCHES:
    R = pitch / np.sqrt(2.0)
    usable_mask = r_sq > (R**2)
    eff = np.sum(usable_mask) / float(n_pool)

    # Continuous unbinned vectors
    dx_use = raw_dx[usable_mask]
    dy_use = raw_dy[usable_mask]

    # Hardware discretized cell vectors (voxel center hopping)
    disc_px = np.floor(proc_base.vertices[usable_mask, 0] / pitch) * pitch + pitch / 2.0
    disc_py = np.floor(proc_base.vertices[usable_mask, 1] / pitch) * pitch + pitch / 2.0
    disc_nx = np.floor(proc_base.captures[usable_mask, 0] / pitch) * pitch + pitch / 2.0
    disc_ny = np.floor(proc_base.captures[usable_mask, 1] / pitch) * pitch + pitch / 2.0
    disc_dx = disc_nx - disc_px
    disc_dy = disc_ny - disc_py

    gs = int(np.round(DETECTOR_WIDTH / pitch))
    can_fnd_base = (pitch >= 1.0) and (gs <= 450)
    
    if can_fnd_base:
        x_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)
        y_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)
        search_angles = np.arange(-180, 180, 1)

    for N in N_VALS:
        if len(dx_use) < 2 * N:
            print(f"[SKIP] dx = {pitch:5.1f} mm, N = {N:,}: Insufficient usable events ({len(dx_use):,} < {2*N:,})")
            continue

        cent_cont_list = []
        cent_disc_list = []
        fnd_list = []
        can_fnd_n = can_fnd_base and (N <= 100000)

        for _ in range(K_TRIALS):
            idx = np.random.choice(len(dx_use), 2 * N, replace=False)
            idx_meas = idx[N:]
            idx_ref = idx[:N]

            # 1. Continuous Centroid
            th_c_cont = np.degrees(np.arctan2(np.mean(dy_use[idx_meas]), np.mean(dx_use[idx_meas])))
            cent_cont_list.append(th_c_cont)

            # 2. Hardware Discretized Centroid
            th_c_disc = np.degrees(np.arctan2(np.mean(disc_dy[idx_meas]), np.mean(disc_dx[idx_meas])))
            cent_disc_list.append(th_c_disc)

            # 3. Continuous FND (Split-half independent reference matching)
            if can_fnd_n:
                xm, ym = dx_use[idx_meas], dy_use[idx_meas]
                xr, yr = dx_use[idx_ref], dy_use[idx_ref]
                ref_mat, _, _ = np.histogram2d(xr, yr, bins=[x_edges, y_edges])
                norms = []
                for phi in search_angles:
                    rad = np.radians(phi)
                    ca, sa = np.cos(rad), np.sin(rad)
                    rx = xm * ca - ym * sa
                    ry = xm * sa + ym * ca
                    rot_mat, _, _ = np.histogram2d(rx, ry, bins=[x_edges, y_edges])
                    norms.append(np.linalg.norm(ref_mat - rot_mat))
                th_f = -search_angles[np.argmin(norms)]
                fnd_list.append(th_f)

        s_c_cont = np.std(cent_cont_list)
        s_c_disc = np.std(cent_disc_list)
        s_fnd = np.std(fnd_list) if (can_fnd_n and len(fnd_list) == K_TRIALS) else np.nan

        fnd_str = f"{s_fnd:6.2f}" if not np.isnan(s_fnd) else "   NaN"
        line = f"{pitch:6.2f} {N:8d} {K_TRIALS:4d} {eff:10.4f} {gs:6d} {s_c_cont:10.3f} {s_c_disc:10.3f} {fnd_str:>10}\n"
        with open(OUT_FILE, "a") as f:
            f.write(line)

        print(f"[{datetime.now().strftime('%T')}] dx={pitch:5.1f}mm | N={N:7d} | Eff={eff*100:4.1f}% | C_cont={s_c_cont:5.2f} | C_disc={s_c_disc:5.2f} | FND={fnd_str}")

print(f"[SUCCESS] Matrix phase space sweep complete: {OUT_FILE}")
