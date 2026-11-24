#!/usr/bin/env python3
"""
run_smearing_sensitivity_suite_v1_0.py
======================================
Evaluates the robustness of Centroid, Parabolic FND, and 2D Spatial EMD (W1)
against physical optical readout smearing sigma_smear in [0, 5, 10, 20, 30, 50] mm.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

PITCHES = [50.0, 5.0]
SMEAR_VALS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]  # mm isotropic Gaussian sigma
N_EVENTS = 10000
K_TRIALS = 15
DETECTOR_WIDTH = 450.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/stats_smearing_sensitivity_{TS}.txt"

with open(OUT_FILE, "w") as f:
    f.write("# dx_mm  sigma_smear_mm  N  trials  Cent_std  FND_std  EMD_std\n")

print(f"[{datetime.now().strftime('%T')}] Loading unbinned coordinates...")
proc = DataProcessor(dx=5, gs=89, n=N_EVENTS)
proc.readData()

raw_vx = proc.vertices[:, 0]
raw_vy = proc.vertices[:, 1]
raw_cx = proc.captures[:, 0]
raw_cy = proc.captures[:, 1]
n_pool = len(raw_vx)

search_angles = np.arange(-15, 16, 1)

for pitch in PITCHES:
    gs = int(np.round(DETECTOR_WIDTH / pitch))
    x_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)
    y_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)

    for smear in SMEAR_VALS:
        cent_list = []
        fnd_list  = []
        emd_list  = []

        for _ in range(K_TRIALS):
            # Draw paired sample
            idx = np.random.choice(n_pool, 2 * N_EVENTS, replace=False)
            idx_meas = idx[N_EVENTS:]
            idx_ref  = idx[:N_EVENTS]

            # Apply isotropic Gaussian optical reconstruction smearing
            smear_vx = raw_vx[idx_meas] + (np.random.normal(0, smear, N_EVENTS) if smear > 0 else 0.0)
            smear_vy = raw_vy[idx_meas] + (np.random.normal(0, smear, N_EVENTS) if smear > 0 else 0.0)
            smear_cx = raw_cx[idx_meas] + (np.random.normal(0, smear, N_EVENTS) if smear > 0 else 0.0)
            smear_cy = raw_cy[idx_meas] + (np.random.normal(0, smear, N_EVENTS) if smear > 0 else 0.0)

            # Reconstruct interaction displacement
            dx_meas = smear_cx - smear_vx
            dy_meas = smear_cy - smear_vy

            # Reference template coordinates (un-smeared canonical baseline)
            dx_ref = raw_cx[idx_ref] - raw_vx[idx_ref]
            dy_ref = raw_cy[idx_ref] - raw_vy[idx_ref]

            # Fiducial / containment cut
            r_sq_meas = dx_meas**2 + dy_meas**2
            cut_r = pitch / np.sqrt(2.0)
            valid = r_sq_meas > (cut_r**2)

            xm, ym = dx_meas[valid], dy_meas[valid]
            xr, yr = dx_ref[:len(xm)], dy_ref[:len(xm)]

            # 1. Discrete Hardware Centroid (quantized to segment centers)
            disc_xm = np.floor(xm / pitch) * pitch + pitch / 2.0
            disc_ym = np.floor(ym / pitch) * pitch + pitch / 2.0
            th_c = np.degrees(np.arctan2(np.mean(disc_ym), np.mean(disc_xm)))
            cent_list.append(th_c)

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
                dp = norms[i_min + 1]
                dz = norms[i_min]
                dm = norms[i_min - 1]
                denom = dp - 2.0 * dz + dm
                sub_step = -0.5 * (dp - dm) / denom if denom > 0 else 0.0
                th_f = -(search_angles[i_min] + sub_step)
            else:
                th_f = -search_angles[i_min]
            fnd_list.append(th_f)

            # 3. 2D Spatial Optimal Transport (EMD / W1 continuous chord proxy)
            # Evaluates physical transport work across candidate angles
            emd_work = []
            for phi in search_angles:
                rad = np.radians(phi)
                ca, sa = np.cos(rad), np.sin(rad)
                rx = xm * ca - ym * sa
                ry = xm * sa + ym * ca
                disp = np.sqrt((rx - xr)**2 + (ry - yr)**2)
                emd_work.append(np.mean(disp))

            emd_work = np.array(emd_work)
            i_emd_min = np.argmin(emd_work)
            if 0 < i_emd_min < len(emd_work) - 1:
                dp = emd_work[i_emd_min + 1]
                dz = emd_work[i_emd_min]
                dm = emd_work[i_emd_min - 1]
                denom = dp - 2.0 * dz + dm
                sub_step = -0.5 * (dp - dm) / denom if denom > 0 else 0.0
                th_e = -(search_angles[i_emd_min] + sub_step)
            else:
                th_e = -search_angles[i_emd_min]
            emd_list.append(th_e)

        s_c = np.std(cent_list)
        s_f = np.std(fnd_list)
        s_e = np.std(emd_list)

        line = f"{pitch:6.1f} {smear:8.1f} {N_EVENTS:8d} {K_TRIALS:4d} {s_c:10.3f} {s_f:10.3f} {s_e:10.3f}\n"
        with open(OUT_FILE, "a") as f:
            f.write(line)

        print(f"[{datetime.now().strftime('%T')}] dx={pitch:4.1f}mm | smear={smear:4.1f}mm | Cent={s_c:5.2f} deg | FND={s_f:5.2f} deg | EMD={s_e:5.2f} deg")

print(f"[SUCCESS] Smearing sensitivity sweep complete: {OUT_FILE}")
