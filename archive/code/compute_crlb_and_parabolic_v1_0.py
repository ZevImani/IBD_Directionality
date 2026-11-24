#!/usr/bin/env python3
"""
compute_crlb_and_parabolic_v1_0.py
==================================
Extracts analytical CRLB constants from the unbinned MC truth pool
and tests 3-point parabolic interpolation on high-N FND.
"""

import numpy as np
from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

print("[INFO] Loading Monte Carlo coordinate pool...")
proc = DataProcessor(dx=5, gs=89, n=100000)
proc.readData()

dx_raw = proc.coords[:, 0]
dy_raw = proc.coords[:, 1]
r_sq = dx_raw**2 + dy_raw**2

# Apply nominal fiducial cut (dx = 50 mm equivalent)
R = 50.0 / np.sqrt(2.0)
usable = r_sq > (R**2)
x_use = dx_raw[usable]
y_use = dy_raw[usable]

# 1. Calculate Analytical CRLB Parameters
# True injection was along theta = 0 (X-axis)
r_forward = np.mean(x_use)
sigma_x = np.std(x_use)
sigma_y = np.std(y_use)
sigma_trans = np.sqrt(0.5 * (sigma_x**2 + sigma_y**2))

# Asymptotic CRLB slope: sigma_theta * sqrt(N) in degrees*sqrt(events)
crlb_constant_rad = sigma_trans / r_forward
crlb_constant_deg = np.degrees(crlb_constant_rad)

print("=" * 60)
print(f"ANALYTICAL CRLB DERIVATION:")
print(f"  Forward Dipole Displacement <r_forward> : {r_forward:6.2f} mm")
print(f"  Transverse Spatial Dispersion sigma_trans: {sigma_trans:6.2f} mm")
print(f"  CRLB Constant (Theta_min = sigma * sqrt(N)): {crlb_constant_deg:6.2f} deg*sqrt(events)")
print("=" * 60)

# 2. Test 3-Point Parabolic Interpolator
# Simulate a discrete Frobenius norm loss vector around a shallow minimum
search_angles = np.arange(-10, 11, 1)
true_offset = 0.37  # True minimum sits at non-integer sub-degree angle
loss_profile = (search_angles - true_offset)**2 + 0.05 * np.random.randn(len(search_angles))

idx_min = np.argmin(loss_profile)
phi_0 = search_angles[idx_min]

d_plus = loss_profile[idx_min + 1]
d_zero = loss_profile[idx_min]
d_minus = loss_profile[idx_min - 1]

delta_phi = -0.5 * (d_plus - d_minus) / (d_plus - 2.0 * d_zero + d_minus)
phi_refined = phi_0 + delta_phi

print(f"PARABOLIC INTERPOLATION TEST:")
print(f"  True Underlying Minimum   : {true_offset:6.3f} deg")
print(f"  Raw Integer Argmin Angle  : {phi_0:6.3f} deg")
print(f"  Parabolic Refined Angle   : {phi_refined:6.3f} deg")
print(f"  Residual Truncation Error : {abs(phi_refined - true_offset):6.3f} deg")
print("=" * 60)
