#!/usr/bin/env python3
"""
plot_unclipped_universal_scaling_v6_0.py
========================================
Journal-ready Figure 5: Universal Scaling Collapse
  - LHS panel (a): Legend moved OUTSIDE beneath panel (a) in 3 clean columns;
    zero data occlusion below 10^0 deg.
  - RHS panel (b): Legend structured in 2 compact columns beneath panel (b)
    with safe right-margin clearance; no text truncation.
  - Uses strictly \geq in all mathtext strings.
"""

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.linewidth": 1.2,
    "xtick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.major.size": 5,
    "ytick.minor.size": 2.5,
})

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
PLOT_DIR = "../results/plots"  # figures; tables stay in OUT_DIR (-> ../results/tables)
os.makedirs(PLOT_DIR, exist_ok=True)
files = sorted(glob.glob(f"{OUT_DIR}/stats_unclipped_universal_scaling_*.txt"))
if not files:
    raise FileNotFoundError("No unclipped universal scaling dataset found.")
data_file = files[-1]
print(f"[INFO] Ingesting table: {data_file}")

with open(data_file) as f:
    cols = f.readline().lstrip('#').strip().split()
df = pd.read_csv(data_file, sep=r'\s+', comment='#', names=cols)

df['Theta_cent'] = df['Cent_disc_std'] * np.sqrt(df['N'])
df['Theta_fnd']  = df['FND_para_std'] * np.sqrt(df['N'])
df['Theta_emd']  = df['EMD_para_std'] * np.sqrt(df['N'])

CRLB_CONST = 129.58

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.2), dpi=300)

PITCH_COLORS = {150.0: '#1f77b4', 50.0: '#ff7f0e', 25.0: '#9467bd', 10.0: '#8c564b', 5.0: '#2ca02c', 1.0: '#d62728'}
MARKERS = {'Cent': 's', 'FND': 'o', 'EMD': '^'}

# -------------------------------------------------------------------------
# Left Panel: Raw Resolution sigma_theta vs rho
# -------------------------------------------------------------------------
for dx in sorted(df['dx_mm'].unique(), reverse=True):
    sub = df[df['dx_mm'] == dx].sort_values('rho')
    col = PITCH_COLORS.get(dx, 'gray')
    ax1.plot(sub['rho'], sub['Cent_disc_std'], f"{MARKERS['Cent']}:", color=col, lw=1.2, ms=4)
    valid_fnd = sub.dropna(subset=['FND_para_std'])
    if not valid_fnd.empty:
        ax1.plot(valid_fnd['rho'], valid_fnd['FND_para_std'], f"{MARKERS['FND']}--", color=col, lw=1.6, ms=5)
    valid_emd = sub.dropna(subset=['EMD_para_std'])
    if not valid_emd.empty:
        ax1.plot(valid_emd['rho'], valid_emd['EMD_para_std'], f"{MARKERS['EMD']}-", color=col, lw=1.8, ms=5)

ax1.axvline(1.0, color='black', ls='--', lw=1.2)
ax1.axhline(1.0, color='gray', ls=':', lw=1.2)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Dimensionless Voxel Occupancy Ratio $\rho \equiv N / G^2$', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Angular Uncertainty $\sigma_\theta$ [degrees]', fontsize=12, fontweight='bold')
ax1.set_title(r'(a) Estimator Resolution vs. Voxel Occupancy', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlim(df['rho'].min() * 0.5, df['rho'].max() * 2.0)
ax1.set_ylim(0.08, 100.0)
ax1.grid(True, which='both', ls='--', alpha=0.45)

# LHS External Bottom Legend (3 Columns, clean below ax1)
legend_elements_ax1 = [
    Line2D([0], [0], color=PITCH_COLORS[150.0], lw=2.0, marker='o', ms=4, label=r'$dx=150\,\mathrm{mm}$'),
    Line2D([0], [0], color=PITCH_COLORS[10.0],  lw=2.0, marker='o', ms=4, label=r'$dx=10\,\mathrm{mm}$'),
    Line2D([0], [0], color='black', lw=1.6, ls='--', marker='o', ms=4, label='FND (dashed)'),
    Line2D([0], [0], color=PITCH_COLORS[50.0],  lw=2.0, marker='o', ms=4, label=r'$dx=50\,\mathrm{mm}$'),
    Line2D([0], [0], color=PITCH_COLORS[5.0],   lw=2.0, marker='o', ms=4, label=r'$dx=5\,\mathrm{mm}$'),
    Line2D([0], [0], color='black', lw=1.8, ls='-', marker='^', ms=4, label=r'EMD $W_1$ (solid)'),
    Line2D([0], [0], color=PITCH_COLORS[25.0],  lw=2.0, marker='o', ms=4, label=r'$dx=25\,\mathrm{mm}$'),
    Line2D([0], [0], color=PITCH_COLORS[1.0],   lw=2.0, marker='o', ms=4, label=r'$dx=1\,\mathrm{mm}$'),
    Line2D([0], [0], color='black', lw=1.2, ls=':', marker='s', ms=4, label='Centroid (dotted)'),
]
ax1.legend(handles=legend_elements_ax1, loc='upper center', bbox_to_anchor=(0.50, -0.18),
           ncol=3, fontsize=8.4, frameon=True, framealpha=0.95)

# -------------------------------------------------------------------------
# Right Panel: Normalized Work Theta vs rho
# -------------------------------------------------------------------------
ax2.axvspan(df['rho'].min() * 0.5, 1.0, color='red', alpha=0.06)
ax2.axvspan(1.0, df['rho'].max() * 2.0, color='green', alpha=0.05)

# Centroid scatter baseline
ax2.scatter(df['rho'], df['Theta_cent'], color='#1f77b4', marker='s', s=28, alpha=0.6)

for dx in sorted(df['dx_mm'].unique(), reverse=True):
    sub = df[df['dx_mm'] == dx].sort_values('rho')
    col = PITCH_COLORS.get(dx, 'gray')
    valid_fnd = sub.dropna(subset=['Theta_fnd'])
    if not valid_fnd.empty:
        ax2.scatter(valid_fnd['rho'], valid_fnd['Theta_fnd'], color=col, marker='o', s=35, alpha=0.85)
    valid_emd = sub.dropna(subset=['Theta_emd'])
    if not valid_emd.empty:
        ax2.scatter(valid_emd['rho'], valid_emd['Theta_emd'], color=col, marker='^', s=40, alpha=0.85)

ax2.axhline(CRLB_CONST, color='black', ls='-', lw=1.5)
ax2.axvline(1.0, color='black', ls='--', lw=1.2)

# Text annotations for physical regimes
ax2.text(1.5e-3, 1100, 'Sparsity Trap\n' + r'($\rho < 1.0$)', fontsize=10, fontweight='bold', color='#a50f15', ha='center')
ax2.text(1e2, 1100, 'Dense Template Regime\n' + r'($\rho \geq 1.0$)', fontsize=10, fontweight='bold', color='#006d2c', ha='center')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Dimensionless Voxel Occupancy Ratio $\rho \equiv N / G^2$', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'Normalized Work $\Theta \equiv \sigma_\theta \sqrt{N}$ [$\mathrm{deg}\cdot\sqrt{\mathrm{events}}$]',
              fontsize=12, fontweight='bold')
ax2.set_title(r'(b) Universal Phase Transition at $\rho_c = 1.0$', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlim(df['rho'].min() * 0.5, df['rho'].max() * 2.0)
ax2.set_ylim(40.0, 1800.0)
ax2.grid(True, which='both', ls='--', alpha=0.45)

# RHS External Bottom Legend (2 Columns, safe horizontal margin buffer)
legend_elements_ax2 = [
    Line2D([0], [0], color='#1f77b4', marker='s', ls='none', ms=6, label='Discrete Centroid (All Pitches)'),
    Line2D([0], [0], color='black', ls='-', lw=1.5, label=rf'CRLB Limit ($\Theta_0 = {CRLB_CONST:.1f}^\circ\sqrt{{N}}$)'),
    Line2D([0], [0], color='gray', marker='o', ls='none', ms=6, label='Continuous FND'),
    Line2D([0], [0], color='black', ls='--', lw=1.2, label=r'Critical Occupancy $\rho_c = 1.0$'),
    Line2D([0], [0], color='gray', marker='^', ls='none', ms=6, label=r'2D Spatial EMD ($W_1$)'),
]

ax2.legend(handles=legend_elements_ax2, loc='upper center', bbox_to_anchor=(0.48, -0.18),
           ncol=2, fontsize=8.6, frameon=True, framealpha=0.95)

# Reserve lower canvas margin for the bottom legends
fig.subplots_adjust(left=0.07, right=0.96, top=0.91, bottom=0.25, wspace=0.22)

out_base = f"{PLOT_DIR}/fig5_unclipped_universal_scaling_v7_{TS}"
fig.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{out_base}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[SUCCESS] Saved perfected Figure 5: {out_base}.[png|pdf]")
