#!/usr/bin/env python3
"""
plot_final_paper_figures_v3_0.py
================================
Generates publication-quality figures from production exposure suite v3:
  - Fig 3: Algorithmic Capacity (sigma_theta vs N_usable) with exact CRLB
  - Fig 4: Operational Exposure (sigma_theta vs n_total) with containment penalty
"""

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
files = sorted(glob.glob(f"{OUT_DIR}/stats_production_exposure_v3_*.txt"))
if not files:
    raise FileNotFoundError("No production exposure v3 data files found.")
data_file = files[-1]
print(f"[INFO] Ingesting production dataset: {data_file}")

with open(data_file) as f:
    cols = f.readline().lstrip('#').strip().split()
df = pd.read_csv(data_file, sep=r'\s+', comment='#', names=cols)

COLOR_MAP = {150.0: '#1f77b4', 50.0: '#ff7f0e', 5.0: '#2ca02c'}
MARKER_MAP = {150.0: 'o', 50.0: 's', 5.0: '^'}
LABELS = {150.0: r'$dx = 150\,\mathrm{mm}\ (3\times 3)$',
          50.0:  r'$dx = 50\,\mathrm{mm}\ (9\times 9)$',
          5.0:   r'$dx = 5\,\mathrm{mm}\ (89\times 89)$'}

CRLB_CONST = 129.58  # deg * sqrt(events) derived from master MC truth pool

# =============================================================================
# FIGURE 3: Algorithmic Capacity (sigma_theta vs N_usable)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(7.5, 5.5), dpi=300)

for dx in [150.0, 50.0, 5.0]:
    sub = df[df['dx_mm'] == dx].sort_values('N_usable').dropna(subset=['FND_para_std'])
    # Filter out boundary-clipped low-N points (N_usable < 50 for 150mm)
    sub = sub[sub['N_usable'] >= 30]
    if sub.empty:
        continue
    ax1.plot(sub['N_usable'], sub['FND_para_std'], f"{MARKER_MAP[dx]}-", color=COLOR_MAP[dx],
             lw=1.8, ms=5.5, label=f"Continuous FND: {LABELS[dx]}")
    ax1.plot(sub['N_usable'], sub['Cent_disc_std'], f"{MARKER_MAP[dx]}:", color=COLOR_MAP[dx],
             lw=1.2, ms=4.0, alpha=0.7)

n_axis = np.logspace(1.5, 6.0, 100)
ax1.plot(n_axis, CRLB_CONST / np.sqrt(n_axis), 'k--', lw=1.3,
         label=rf'Analytical CRLB Limit ($\Theta = {CRLB_CONST:.1f}^\circ\sqrt{{N}}$)')
ax1.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'Sub-Degree Target ($1.0^\circ$)')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Usable IBD Events ($N_{\mathrm{usable}}$)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Angular Uncertainty $\sigma_\theta$ [degrees]', fontsize=12, fontweight='bold')
ax1.set_title(r'Algorithmic Capacity: Resolution vs. Usable Statistics', fontsize=12, fontweight='bold')
ax1.set_xlim(25, 2e6)
ax1.set_ylim(0.08, 30.0)
ax1.grid(True, which='both', ls='--', alpha=0.45)
ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

fig1.tight_layout()
out_fig1 = f"{PLOT_DIR}/fig3_algorithmic_capacity_crlb_{TS}"
fig1.savefig(f"{out_fig1}.png", bbox_inches='tight')
fig1.savefig(f"{out_fig1}.pdf", bbox_inches='tight')
plt.close(fig1)
print(f"[SUCCESS] Saved Fig 3 (Algorithmic Capacity): {out_fig1}.[png|pdf]")

# =============================================================================
# FIGURE 4: Operational Exposure (sigma_theta vs n_total = N / epsilon)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(7.5, 5.5), dpi=300)

for dx in [150.0, 50.0, 5.0]:
    sub = df[df['dx_mm'] == dx].sort_values('n_total').dropna(subset=['FND_para_std'])
    sub = sub[sub['N_usable'] >= 30]
    if sub.empty:
        continue
    ax2.plot(sub['n_total'], sub['FND_para_std'], f"{MARKER_MAP[dx]}-", color=COLOR_MAP[dx],
             lw=1.8, ms=5.5, label=f"Continuous FND: {LABELS[dx]}")
    ax2.plot(sub['n_total'], sub['Cent_disc_std'], f"{MARKER_MAP[dx]}:", color=COLOR_MAP[dx],
             lw=1.2, ms=4.0, alpha=0.7)

n_tot_axis = np.logspace(2.0, 6.0, 100)
ax2.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'Sub-Degree Target ($1.0^\circ$)')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Total Detector Exposure ($n_{\mathrm{total}} = N_{\mathrm{usable}} / \epsilon$)', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'Angular Uncertainty $\sigma_\theta$ [degrees]', fontsize=12, fontweight='bold')
ax2.set_title(r'Operational Performance: Containment Penalty Inversion', fontsize=12, fontweight='bold')
ax2.set_xlim(80, 2e6)
ax2.set_ylim(0.08, 30.0)
ax2.grid(True, which='both', ls='--', alpha=0.45)
ax2.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

fig2.tight_layout()
out_fig2 = f"{PLOT_DIR}/fig4_operational_exposure_containment_{TS}"
fig2.savefig(f"{out_fig2}.png", bbox_inches='tight')
fig2.savefig(f"{out_fig2}.pdf", bbox_inches='tight')
plt.close(fig2)
print(f"[SUCCESS] Saved Fig 4 (Operational Exposure): {out_fig2}.[png|pdf]")
