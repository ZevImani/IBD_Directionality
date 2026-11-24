#!/usr/bin/env python3
"""
run_dense_kink_sweep_v1_0.py
============================
High-density N sweep across the 1k -> 15k kink zone for dx = [5, 50, 150] mm.
Evaluates K = 25 independent split-half trials to isolate true scaling behavior.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

N_VALUES = [1000, 2000, 3000, 4000, 4500, 4750, 5000, 5250, 5500, 6000, 7500, 10000, 15000, 20000, 50000]
GEOMS = [5.0, 50.0, 150.0]
K_TRIALS = 25
DETECTOR_WIDTH = 450.0

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = f"{OUT_DIR}/stats_dense_kink_audit_{TS}.txt"

with open(OUT_FILE, "w") as f:
    f.write("# dx_mm  N  trials  Cent_disc_std  FND_std  usable_eff\n")

print(f"[{datetime.now().strftime('%T')}] Loading coordinates...")
proc_base = DataProcessor(dx=5, gs=89, n=10000)
proc_base.readData()

raw_dx = proc_base.coords[:, 0]
raw_dy = proc_base.coords[:, 1]
r_sq = raw_dx**2 + raw_dy**2
n_pool = len(raw_dx)
search_angles = np.arange(-180, 180, 1)

for dx in GEOMS:
    R = dx / np.sqrt(2.0)
    usable_mask = r_sq > (R**2)
    eff = np.sum(usable_mask) / float(n_pool)

    dx_use = raw_dx[usable_mask]
    dy_use = raw_dy[usable_mask]

    # Hardware discretized coordinates
    disc_px = np.floor(proc_base.vertices[usable_mask, 0] / dx) * dx + dx / 2.0
    disc_py = np.floor(proc_base.vertices[usable_mask, 1] / dx) * dx + dx / 2.0
    disc_nx = np.floor(proc_base.captures[usable_mask, 0] / dx) * dx + dx / 2.0
    disc_ny = np.floor(proc_base.captures[usable_mask, 1] / dx) * dx + dx / 2.0
    disc_dx = disc_nx - disc_px
    disc_dy = disc_ny - disc_py

    gs = int(np.round(DETECTOR_WIDTH / dx))
    x_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)
    y_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)

    print(f"[{datetime.now().strftime('%T')}] Starting sweep for dx = {dx} mm (gs={gs})...")

    for N in N_VALUES:
        if len(dx_use) < 2 * N:
            continue

        cent_list = []
        fnd_list = []

        for _ in range(K_TRIALS):
            idx = np.random.choice(len(dx_use), 2 * N, replace=False)
            idx_meas = idx[N:]
            idx_ref = idx[:N]

            # 1. Discrete Centroid
            th_c = np.degrees(np.arctan2(np.mean(disc_dy[idx_meas]), np.mean(disc_dx[idx_meas])))
            cent_list.append(th_c)

            # 2. Continuous FND (Split-half bootstrap matching)
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

        s_c = np.std(cent_list)
        s_f = np.std(fnd_list)

        line = f"{dx:6.1f} {N:8d} {K_TRIALS:4d} {s_c:10.3f} {s_f:10.3f} {eff:10.4f}\n"
        with open(OUT_FILE, "a") as f:
            f.write(line)

        print(f"  dx={dx:5.1f} mm | N={N:6d} | Cent_std={s_c:5.2f} deg | FND_std={s_f:5.2f} deg")

print(f"[SUCCESS] Dense kink sweep complete: {OUT_FILE}")
