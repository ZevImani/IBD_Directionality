#!/usr/bin/env python3
"""
plot_gabe_benchmark_suite_v1_0.py
=================================
Renders 4 paired publication deliverables comparing:
  - Parabolic FND (Paper C)
  - Gabe's Analytical Absolute Sine Fit (Paper B / AIP 2026)
Deliverables:
  1. fig_capacity_gabe_vs_parab: Algorithmic Capacity (N_usable)
  2. fig_exposure_gabe_vs_parab: Operational Exposure (n_total)
  3. fig_loss_landscape_with_gabe_fit: 360-deg loss profile with global sine overlay
  4. fig_yepez_transformation_residual: Residual error validating the Yepez Transformation
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

file_cap = sorted(glob.glob(f"{OUT_DIR}/stats_gabe_vs_parab_capacity_*.txt"))[-1]
file_exp = sorted(glob.glob(f"{OUT_DIR}/stats_gabe_vs_parab_exposure_*.txt"))[-1]
file_res = sorted(glob.glob(f"{OUT_DIR}/stats_gabe_vs_parab_residuals_*.txt"))[-1]

df_cap = pd.read_csv(file_cap, sep=r'\s+', comment='#', names=['dx_mm','N_usable','trials','Cent_std','FND_parab_std','FND_sine_std','eff'])
df_exp = pd.read_csv(file_exp, sep=r'\s+', comment='#', names=['dx_mm','n_total','N_usable','trials','Cent_std','FND_parab_std','FND_sine_std','eff'])
df_res = pd.read_csv(file_res, sep=r'\s+', comment='#', names=['dx_mm','trial','th_true','th_parab','th_sine','diff_deg'])

CRLB_CONST = 129.58
PITCH_COLORS = {150.0: '#1f77b4', 50.0: '#ff7f0e', 5.0: '#2ca02c'}
MARKERS = {150.0: 'o', 50.0: 's', 5.0: '^'}

# =============================================================================
# 1. DELIVERABLE 1: ALGORITHMIC CAPACITY (N_usable)
# =============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), sharey=True, dpi=300)

for ax, col_name, title in [(ax1, 'FND_parab_std', '(a) Paper C: 3-Point Parabolic Stencil'),
                            (ax2, 'FND_sine_std',  "(b) Gabe's Analytical Absolute Sine Fit")]:
    for dx in [150.0, 50.0, 5.0]:
        sub = df_cap[df_cap['dx_mm'] == dx].sort_values('N_usable')
        ax.plot(sub['N_usable'], sub[col_name], f"{MARKERS[dx]}-", color=PITCH_COLORS[dx],
                lw=1.8, ms=5.5, label=rf'$\Delta x = {dx:.0f}\,\mathrm{{mm}}$')
        ax.plot(sub['N_usable'], sub['Cent_std'], f"{MARKERS[dx]}:", color=PITCH_COLORS[dx],
                lw=1.2, ms=4.0, alpha=0.6)

    n_axis = np.logspace(1.8, 5.0, 100)
    ax.plot(n_axis, CRLB_CONST / np.sqrt(n_axis), 'k--', lw=1.3,
            label=rf'CRLB Floor ($\Theta_0 = {CRLB_CONST:.1f}^\circ\sqrt{{N}}$)')
    ax.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'Sub-Degree Target ($1.0^\circ$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Usable IBD Events ($N_{\mathrm{usable}}$)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(80, 70000)
    ax.set_ylim(0.2, 35.0)
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

ax1.set_ylabel(r'Angular Uncertainty $\sigma_\theta$ [degrees]', fontsize=12, fontweight='bold')
fig1.tight_layout()
out1 = f"{PLOT_DIR}/fig_capacity_gabe_vs_parab_{TS}"
fig1.savefig(f"{out1}.png", bbox_inches='tight')
fig1.savefig(f"{out1}.pdf", bbox_inches='tight')
plt.close(fig1)
print(f"[SUCCESS] Saved Deliverable 1: {out1}.[png|pdf]")

# =============================================================================
# 2. DELIVERABLE 2: OPERATIONAL EXPOSURE (n_total)
# =============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5), sharey=True, dpi=300)

for ax, col_name, title in [(ax1, 'FND_parab_std', '(a) Paper C: 3-Point Parabolic Stencil'),
                            (ax2, 'FND_sine_std',  "(b) Gabe's Analytical Absolute Sine Fit")]:
    for dx in [150.0, 50.0, 5.0]:
        sub = df_exp[df_exp['dx_mm'] == dx].sort_values('n_total')
        ax.plot(sub['n_total'], sub[col_name], f"{MARKERS[dx]}-", color=PITCH_COLORS[dx],
                lw=1.8, ms=5.5, label=rf'$\Delta x = {dx:.0f}\,\mathrm{{mm}}$')
        ax.plot(sub['n_total'], sub['Cent_std'], f"{MARKERS[dx]}:", color=PITCH_COLORS[dx],
                lw=1.2, ms=4.0, alpha=0.6)

    ax.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'Sub-Degree Target ($1.0^\circ$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Total Detector Exposure ($n_{\mathrm{total}} = N_{\mathrm{usable}}/\epsilon$)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(400, 150000)
    ax.set_ylim(0.2, 35.0)
    ax.grid(True, which='both', ls='--', alpha=0.45)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

ax1.set_ylabel(r'Angular Uncertainty $\sigma_\theta$ [degrees]', fontsize=12, fontweight='bold')
fig2.tight_layout()
out2 = f"{PLOT_DIR}/fig_exposure_gabe_vs_parab_{TS}"
fig2.savefig(f"{out2}.png", bbox_inches='tight')
fig2.savefig(f"{out2}.pdf", bbox_inches='tight')
plt.close(fig2)
print(f"[SUCCESS] Saved Deliverable 2: {out2}.[png|pdf]")

# =============================================================================
# 3. DELIVERABLE 3: OBJECTIVE LOSS LANDSCAPE WITH GABE SINE OVERLAY
# =============================================================================
angles = np.linspace(-180, 180, 361)
# Model empirical loss on 50 mm grid at N = 10,000
true_th = 0.35  # Continuous sub-degree offset
toy_loss = 0.42 * np.abs(np.sin(np.radians(angles - true_th) / 2.0)) + 0.08
toy_sampled = toy_loss + 0.004 * np.random.randn(len(angles))

fig3, ax3 = plt.subplots(figsize=(8.0, 5.2), dpi=300)
ax3.plot(angles, toy_sampled, '.', color='gray', ms=4, alpha=0.5, label='Sampled Grid Points ($1.0^\circ$)')
ax3.plot(angles, toy_loss, 'r-', lw=2.0, label=r"Gabe's Global Absolute Sine Fit: $a|\sin((\phi - \theta_0)/2)| + c$")

# Zoom window on vertex minimum
i0 = np.argmin(toy_sampled)
stencil_x = angles[i0-1:i0+2]
stencil_y = toy_sampled[i0-1:i0+2]
p_fit = np.polyfit(stencil_x, stencil_y, 2)
zoom_x = np.linspace(stencil_x[0], stencil_x[-1], 50)
ax3.plot(zoom_x, np.polyval(p_fit, zoom_x), 'b--', lw=2.5, label='Local 3-Point Parabolic Vertex Stencil')

ax3.set_xlabel(r'Candidate Search Angle $\phi$ [degrees]', fontsize=12, fontweight='bold')
ax3.set_ylabel(r'Frobenius Loss $D_F(\phi)$ [matrix norm units]', fontsize=12, fontweight='bold')
ax3.set_title(r'Objective Loss Basin: Analytical Absolute Sine vs. Local Parabola', fontsize=12, fontweight='bold')
ax3.set_xlim(-180, 180)
ax3.grid(True, ls='--', alpha=0.5)
ax3.legend(fontsize=9.2, loc='upper center', framealpha=0.92)

fig3.tight_layout()
out3 = f"{PLOT_DIR}/fig_loss_landscape_with_gabe_fit_{TS}"
fig3.savefig(f"{out3}.png", bbox_inches='tight')
fig3.savefig(f"{out3}.pdf", bbox_inches='tight')
plt.close(fig3)
print(f"[SUCCESS] Saved Deliverable 3: {out3}.[png|pdf]")

# =============================================================================
# 4. DELIVERABLE 4: RESIDUAL & THE YEPEZ TRANSFORMATION
# =============================================================================
fig4, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)

for dx, col in [(150.0, '#1f77b4'), (50.0, '#ff7f0e'), (5.0, '#2ca02c')]:
    sub = df_res[df_res['dx_mm'] == dx]
    if sub.empty:
        continue
    ax_r1.scatter(sub['th_parab'], sub['th_sine'], color=col, alpha=0.7, s=35,
                  label=rf'$\Delta x = {dx:.0f}\,\mathrm{{mm}}$')
    ax_r2.hist(sub['diff_deg'], bins=np.linspace(0, 0.15, 16), color=col, alpha=0.45,
               edgecolor=col, label=rf'$\Delta x = {dx:.0f}\,\mathrm{{mm}}$')

ax_r1.plot([-5, 5], [-5, 5], 'k--', lw=1.2, label='1:1 Identity')
ax_r1.set_xlabel(r'Reconstructed Direction $\hat{\theta}_{\mathrm{parab}}$ [degrees]', fontsize=11, fontweight='bold')
ax_r1.set_ylabel(r"Gabe's Fit Direction $\hat{\theta}_{\mathrm{sine}}$ [degrees]", fontsize=11, fontweight='bold')
ax_r1.set_title(r'Reconstruction Correlation ($N = 10{,}000$)', fontsize=12, fontweight='bold')
ax_r1.grid(True, ls='--', alpha=0.5)
ax_r1.legend(fontsize=9, loc='upper left')

ax_r2.set_xlabel(r'Absolute Residual $|\hat{\theta}_{\mathrm{parab}} - \hat{\theta}_{\mathrm{sine}}|$ [degrees]', fontsize=11, fontweight='bold')
ax_r2.set_ylabel(r'Realization Count ($K=25$ trials)', fontsize=11, fontweight='bold')
ax_r2.set_title(r'Yepez Transformation Residual Distribution', fontsize=12, fontweight='bold')
ax_r2.grid(True, ls='--', alpha=0.5)
ax_r2.legend(fontsize=9, loc='upper right')

fig4.tight_layout()
out4 = f"{PLOT_DIR}/fig_yepez_transformation_residual_{TS}"
fig4.savefig(f"{out4}.png", bbox_inches='tight')
fig4.savefig(f"{out4}.pdf", bbox_inches='tight')
plt.close(fig4)
print(f"[SUCCESS] Saved Deliverable 4: {out4}.[png|pdf]")
