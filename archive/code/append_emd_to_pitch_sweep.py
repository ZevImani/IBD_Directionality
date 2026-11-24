#!/usr/bin/env python3
"""
append_emd_to_pitch_sweep.py
============================
Version: v1.0 (2026-10-07)
Author: Max A. A. Dornfest
Description: Computes 2D spatial Wasserstein transport work W1 (in mm) between
             the 10-deg signal and 0-deg template across all viable pitches,
             merging it into the heavy pitch sweep data table.
"""

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
IN_FILES = sorted(glob.glob(f"{OUT_DIR}/stats_heavy_pitch_sweep_n10k_*.txt"))
if not IN_FILES:
    raise FileNotFoundError("No heavy pitch sweep tables found.")
in_file = IN_FILES[-1]
print(f"[INFO] Ingesting base pitch table: {in_file}")

with open(in_file) as f:
    cols = f.readline().lstrip('#').strip().split()
df = pd.read_csv(in_file, sep=r'\s+', comment='#', names=cols)

# Load master unbinned coordinates
proc_base = DataProcessor(dx=5, gs=89, n=10000)
proc_base.readData()
raw_dx = proc_base.coords[:, 0]
raw_dy = proc_base.coords[:, 1]
r_sq = raw_dx**2 + raw_dy**2

N_SAMPLE = 10000
K_TRIALS = 5
THETA_TRUE_RAD = np.radians(10.0)

# Rotate baseline coordinate pool to 0 deg for transport work comparison
cos_t, sin_t = np.cos(-THETA_TRUE_RAD), np.sin(-THETA_TRUE_RAD)
raw_dx_0 = raw_dx * cos_t - raw_dy * sin_t
raw_dy_0 = raw_dx * sin_t + raw_dy * cos_t

emd_means = []
emd_stds = []

print(f"[{datetime.now().strftime('%T')}] Computing 2D Spatial Transport Work across pitches...")
for idx, row in df.iterrows():
    pitch = row['dx_mm']
    R = pitch / np.sqrt(2.0)
    mask = r_sq > (R**2)
    
    dx_meas = raw_dx[mask]
    dy_meas = raw_dy[mask]
    dx_ref = raw_dx_0[mask]
    dy_ref = raw_dy_0[mask]
    
    if len(dx_meas) < N_SAMPLE:
        emd_means.append(np.nan)
        emd_stds.append(np.nan)
        continue
        
    trial_w1 = []
    for _ in range(K_TRIALS):
        sub_m = np.random.choice(len(dx_meas), N_SAMPLE, replace=False)
        sub_r = np.random.choice(len(dx_ref), N_SAMPLE, replace=False)
        
        # 2D Spatial coordinate transport work (in mm)
        w1_x = wasserstein_distance(dx_meas[sub_m], dx_ref[sub_r])
        w1_y = wasserstein_distance(dy_meas[sub_m], dy_ref[sub_r])
        w1_2d = np.sqrt(w1_x**2 + w1_y**2)
        trial_w1.append(w1_2d)
        
    m_val = np.mean(trial_w1)
    s_val = np.std(trial_w1)
    emd_means.append(m_val)
    emd_stds.append(s_val)
    print(f"  dx = {pitch:8.3f} mm: W1 = {m_val:7.3f} +/- {s_val:5.3f} mm")

df['EMD_W1_mm'] = emd_means
df['EMD_W1_std'] = emd_stds

out_file = f"{OUT_DIR}/stats_heavy_pitch_sweep_with_emd_n10k_{TS}.txt"
with open(out_file, "w") as f:
    f.write("# " + "  ".join(df.columns) + "\n")
    df.to_csv(f, sep=" ", index=False, header=False, na_rep="NaN")

print(f"[SUCCESS] Saved EMD-augmented table: {out_file}")
