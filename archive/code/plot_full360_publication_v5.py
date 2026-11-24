#!/usr/bin/env python3
"""
plot_full360_publication_v5.py
==============================
Publication-quality 360-degree linearity and isotropy verification:
  - Top: Clean, unwrapped reconstructed vs true angle (no periodic boundary chords).
  - Bottom: Angular uncertainty evaluated in 15-deg sectors with error bars
            and smooth guide baselines, eliminating solid barcode artifacts.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.linewidth": 1.1})

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
PLOT_DIR = "../results/plots"  # figures; tables stay in OUT_DIR (-> ../results/tables)
os.makedirs(PLOT_DIR, exist_ok=True)
DATA_FILE = f"{OUT_DIR}/stats_angle_sweep_full360_0p5deg_20261006.txt"

with open(DATA_FILE) as f:
    cols = f.readline().lstrip('#').strip().split()
df = pd.read_csv(DATA_FILE, sep=r'\s+', comment='#', names=cols)

COLOR_MAP = {150: '#1f77b4', 50: '#ff7f0e', 5: '#2ca02c'}
MARKER_MAP = {150: 'o', 50: 's', 5: '^'}
LABELS = {150: r'$dx = 150\,\mathrm{mm}\ (3\times 3)$',
          50:  r'$dx = 50\,\mathrm{mm}\ (9\times 9)$',
          5:   r'$dx = 5\,\mathrm{mm}\ (89\times 89)$'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, dpi=300,
                               gridspec_kw={'height_ratios': [1.2, 1.0], 'hspace': 0.08})

# -------------------------------------------------------------------------
# Panel 1: Linearity (Clean Unwrapped Trace)
# -------------------------------------------------------------------------
ax1.plot([-180, 180], [-180, 180], 'k--', lw=1.2, label='Ideal Identity')

for dx in [150, 50, 5]:
    sub = df[df['dx_mm'] == dx].sort_values('th_true').copy()
    th_t = sub['th_true'].values
    th_r = sub['FND_mean'].values
    
    # Unbroken line by inserting NaN at branch-cut jumps
    jump = np.where(np.abs(np.diff(th_r)) > 150.0)[0]
    if len(jump) > 0:
        th_t = np.insert(th_t, jump + 1, np.nan)
        th_r = np.insert(th_r, jump + 1, np.nan)

    ax1.plot(th_t, th_r, '-', color=COLOR_MAP[dx], lw=1.8, label=LABELS[dx])

ax1.set_ylabel(r'Reconstructed Angle $\hat{\theta}$ [$^\circ$]', fontsize=11, fontweight='bold')
ax1.set_title(r'Full $360^\circ$ Directional Linearity ($N = 10\mathrm{k}$)', fontsize=12, fontweight='bold')
ax1.grid(True, which='both', ls='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

# -------------------------------------------------------------------------
# Panel 2: Isotropic Resolution (15-deg Binned Points + Guide Lines)
# -------------------------------------------------------------------------
BIN_WIDTH = 15.0  # degrees
bin_edges = np.arange(-180.0, 180.0 + BIN_WIDTH, BIN_WIDTH)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for dx in [150, 50, 5]:
    sub = df[df['dx_mm'] == dx].copy()
    sub['bin'] = pd.cut(sub['th_true'], bins=bin_edges, labels=bin_centers)
    
    grouped = sub.groupby('bin')['FND_std']
    means = grouped.mean().values
    sems = (grouped.std() / np.sqrt(grouped.count())).values

    col = COLOR_MAP[dx]
    mark = MARKER_MAP[dx]

    # Guide line connecting sector means
    ax2.plot(bin_centers, means, '-', color=col, lw=1.5, alpha=0.7)
    
    # Discrete sector markers with error bars
    ax2.errorbar(bin_centers, means, yerr=sems, fmt=mark, color=col, ecolor=col,
                 elinewidth=1.2, capsize=3, capthick=1.2, ms=5.5, label=LABELS[dx])

ax2.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'Sub-Degree Target ($1.0^\circ$)')
ax2.set_xlabel(r'True Source Injection Angle $\theta_{\mathrm{true}}$ [$^\circ$]', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Uncertainty $\sigma_\theta$ [$^\circ$]', fontsize=11, fontweight='bold')
ax2.set_xlim(-185, 185)
ax2.set_ylim(0.0, 8.5)
ax2.grid(True, which='both', ls='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

fig.tight_layout()
out_base = f"{PLOT_DIR}/fig_full360_sweep_clean_publication_v5_{TS}"
fig.savefig(f"{out_base}.png", bbox_inches='tight')
fig.savefig(f"{out_base}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[SUCCESS] Saved clean binned 360-degree figure: {out_base}.[png|pdf]")
