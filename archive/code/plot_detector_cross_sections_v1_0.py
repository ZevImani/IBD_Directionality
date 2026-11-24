#!/usr/bin/env python3
"""
plot_detector_cross_sections_v1_0.py
====================================
Renders 2D detector cross-section hit maps for Gabe:
  - Plots physical cell boundaries across 450 x 450 mm detector face
  - Compares Delta x = 150 mm (3x3), 50 mm (9x9), and 5 mm (89x89)
  - Overlays prompt positron vertices, delayed capture sites, and displacement vectors
  - Strictly uses \leq in mathtext to prevent parser crashes
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from paper_c_stats_package_v0_6_6_4_2_0 import DataProcessor

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
PLOT_DIR = "../results/plots"  # figures; tables stay in OUT_DIR (-> ../results/tables)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
DETECTOR_WIDTH = 450.0  # mm

print(f"[{datetime.now().strftime('%T')}] Ingesting coordinate sample for hit visualization...")
proc = DataProcessor(dx=5, gs=89, n=500)
proc.readData()

vx = proc.vertices[:500, 0]
vy = proc.vertices[:500, 1]
cx = proc.captures[:500, 0]
cy = proc.captures[:500, 1]

fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), dpi=300)
pitches = [150.0, 50.0, 5.0]

for ax, dx in zip(axes, pitches):
    gs = int(np.round(DETECTOR_WIDTH / dx))
    grid_edges = np.linspace(-DETECTOR_WIDTH / 2.0, DETECTOR_WIDTH / 2.0, gs + 1)

    ax.set_xlim(-240, 240)
    ax.set_ylim(-240, 240)
    ax.set_aspect('equal')

    # Draw grid cells
    for edge in grid_edges:
        ax.axvline(edge, color='gray', lw=0.6, alpha=0.6, ls='-')
        ax.axhline(edge, color='gray', lw=0.6, alpha=0.6, ls='-')

    # Chooz cut circle around origin for illustration
    cut_circle = patches.Circle((0, 0), dx / np.sqrt(2.0), fill=False, edgecolor='purple',
                                ls='--', lw=1.2)
    ax.add_patch(cut_circle)

    # Plot sample interaction vectors (first 40 events)
    for i in range(40):
        ax.plot(vx[i], vy[i], 'r.', ms=5, alpha=0.7)
        ax.plot(cx[i], cy[i], 'b.', ms=5, alpha=0.7)
        ax.annotate('', xy=(cx[i], cy[i]), xytext=(vx[i], vy[i]),
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.7, alpha=0.6))

    ax.set_xlabel('Transverse $X$ [mm]', fontsize=11, fontweight='bold')
    if ax == axes[0]:
        ax.set_ylabel('Transverse $Y$ [mm]', fontsize=11, fontweight='bold')
    ax.set_title(rf'$\Delta x = {dx:.0f}\,\mathrm{{mm}}\ ({gs}\times{gs}\ \mathrm{{cells}})$', fontsize=12, fontweight='bold')
    ax.grid(False)

# Custom legend with \leq
custom_handles = [
    plt.Line2D([0], [0], marker='.', color='r', ls='none', ms=8, label=r'Prompt Vertex ($e^+$)'),
    plt.Line2D([0], [0], marker='.', color='b', ls='none', ms=8, label=r'Delayed Capture ($n$)'),
    plt.Line2D([0], [0], color='black', lw=1.2, label=r'Displacement Vector $\vec{r}$'),
    plt.Line2D([0], [0], color='purple', ls='--', lw=1.2, label=r'Single-Cell Exclusion $r \leq \Delta x/\sqrt{2}$'),
]
fig.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.51, 1.02),
           ncol=4, fontsize=10, frameon=True, framealpha=0.95)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
out_fig = f"{PLOT_DIR}/fig_detector_cross_section_hits_{TS}"
fig.savefig(f"{out_fig}.png", bbox_inches='tight')
fig.savefig(f"{out_fig}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[SUCCESS] Saved detector hit cross-sections: {out_fig}.[png|pdf]")
