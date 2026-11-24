#!/usr/bin/env python3
"""
plot_smearing_sensitivity_v1_0.py
=================================
Plots angular resolution vs position smearing for Centroid, FND, and EMD.
"""
import glob, os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.linewidth": 1.2})

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
PLOT_DIR = "../results/plots"  # figures; tables stay in OUT_DIR (-> ../results/tables)
os.makedirs(PLOT_DIR, exist_ok=True)
files = sorted(glob.glob(f"{OUT_DIR}/stats_smearing_sensitivity_*.txt"))
df = pd.read_csv(files[-1], sep=r'\s+', comment='#', 
                 names=['dx_mm','sigma_smear_mm','N','trials','Cent_std','FND_std','EMD_std'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True, dpi=300)

for ax, dx, title in [(ax1, 50.0, r'Coarse Segments ($dx = 50\,\mathrm{mm}$)'),
                      (ax2, 5.0,  r'Fine Segments ($dx = 5\,\mathrm{mm}$)')]:
    sub = df[df['dx_mm'] == dx].sort_values('sigma_smear_mm')
    if sub.empty:
        continue

    ax.plot(sub['sigma_smear_mm'], sub['Cent_std'], 's-', color='#1f77b4', lw=2.0, ms=6, label='Discrete Centroid')
    ax.plot(sub['sigma_smear_mm'], sub['FND_std'],  'o-', color='#e41a1c', lw=2.0, ms=6, label='Continuous FND (Parabolic)')
    ax.plot(sub['sigma_smear_mm'], sub['EMD_std'],  '^-', color='#2ca02c', lw=2.0, ms=6, label=r'2D Spatial EMD ($W_1$)')

    ax.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'Sub-Degree Target ($1.0^\circ$)')
    ax.set_xlabel(r'Optical Position Smearing $\sigma_{\mathrm{smear}}$ [mm]', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)

ax1.set_ylabel(r'Angular Uncertainty $\sigma_\theta$ [degrees]', fontsize=12, fontweight='bold')
fig.tight_layout()
out_fig = f"{PLOT_DIR}/fig_smearing_sensitivity_centroid_fnd_emd_v1_{TS}"
fig.savefig(f"{out_fig}.png", bbox_inches='tight')
fig.savefig(f"{out_fig}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[SUCCESS] Saved smearing sensitivity figure: {out_fig}.[png|pdf]")
