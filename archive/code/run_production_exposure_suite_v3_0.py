#!/usr/bin/env python3
"""
run_production_exposure_suite_v3_0.py
=====================================
Production-grade sweep generating the final journal scaling datasets:
  - Algorithmic Capacity: sigma_theta vs N_usable
  - Operational Exposure: sigma_theta vs n_total = N_usable / epsilon
Sweeps n_total in [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
with K = 25 trials using parabolic sub-degree refinement across all estimators.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

PITCHES = [150.0, 50.0, 5.0]
N_TOTAL_VALS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
K_TRIALS = 25
DETECTOR_WIDTH = 450.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/stats_production_exposure_v3_{TS}.txt"

with open(OUT_FILE, "w") as f:
    f.write("# dx_mm  n_total  N_usable  usable_eff  trials  Cent_cont_std  Cent_disc_std  FND_para_std  EMD_std\n")

print(f"[{datetime.now().strftime('%T')}] Ingesting master coordinate pool...")
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

    # Discrete hardware hopping
    disc_px = np.floor(proc.vertices[usable_mask, 0] / pitch) * pitch + pitch / 2.0
    disc_py = np.floor(proc.vertices[usable_mask, 1] / pitch) * pitch + pitch / 2.0
    disc_nx = np.floor(proc.captures[usable_mask, 0] / pitch) * pitch + pitch / 2.0
    disc_ny = np.floor(proc.captures[usable_mask, 1] / pitch) * pitch + pitch / 2.0
    disc_dx = disc_nx - disc_px
    disc_dy = disc_ny - disc_py

    gs = int(np.round(DETECTOR_WIDTH / pitch))
    x_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)
    y_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)

    print(f"[{datetime.now().strftime('%T')}] Running production suite for dx = {pitch:5.1f} mm (eff = {eff*100:4.1f}%)...")

    for n_tot in N_TOTAL_VALS:
        N_usable = int(np.round(n_tot * eff))
        if N_usable < 5:
            continue
        if len(dx_use) < 2 * N_usable:
            continue

        c_cont_list = []
        c_disc_list = []
        fnd_list    = []
        emd_list    = []

        can_fnd = (N_usable <= 100000)

        for _ in range(K_TRIALS):
            idx = np.random.choice(len(dx_use), 2 * N_usable, replace=False)
            idx_meas = idx[N_usable:]
            idx_ref  = idx[:N_usable]

            # 1. Continuous Centroid
            th_c_cont = np.degrees(np.arctan2(np.mean(dy_use[idx_meas]), np.mean(dx_use[idx_meas])))
            c_cont_list.append(th_c_cont)

            # 2. Discrete Centroid
            th_c_disc = np.degrees(np.arctan2(np.mean(disc_dy[idx_meas]), np.mean(disc_dx[idx_meas])))
            c_disc_list.append(th_c_disc)

            xm, ym = dx_use[idx_meas], dy_use[idx_meas]
            xr, yr = dx_use[idx_ref], dy_use[idx_ref]

            # 3. Continuous FND with Parabolic Refinement
            if can_fnd:
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

            # 4. Continuous Spatial EMD
            if can_fnd:
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

        s_cc = np.std(c_cont_list)
        s_cd = np.std(c_disc_list)
        s_f  = np.std(fnd_list) if (can_fnd and len(fnd_list) == K_TRIALS) else np.nan
        s_e  = np.std(emd_list) if (can_fnd and len(emd_list) == K_TRIALS) else np.nan

        f_str = f"{s_f:8.3f}" if not np.isnan(s_f) else "     NaN"
        e_str = f"{s_e:8.3f}" if not np.isnan(s_e) else "     NaN"

        line = f"{pitch:6.1f} {n_tot:9d} {N_usable:9d} {eff:10.4f} {K_TRIALS:4d} {s_cc:10.3f} {s_cd:10.3f} {f_str} {e_str}\n"
        with open(OUT_FILE, "a") as f:
            f.write(line)

        print(f"  n_tot={n_tot:8d} | N_use={N_usable:7d} | Cent_disc={s_cd:5.2f} | FND_para={f_str.strip()} | EMD={e_str.strip()}")

print(f"[SUCCESS] Production exposure suite complete: {OUT_FILE}")
