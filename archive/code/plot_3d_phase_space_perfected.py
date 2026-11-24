#!/usr/bin/env python3
"""
plot_3d_phase_space_perfected.py
================================
Restores the broad face-on viewing perspective (elev=24, azim=-215 / 145)
and relocates the colorbar horizontally to the bottom margin to eliminate
all label collisions.
"""

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.linewidth": 1.1})

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "review_deliverables"
PLOT_DIR = "../results/plots"  # figures; tables stay in OUT_DIR (-> ../results/tables)
os.makedirs(PLOT_DIR, exist_ok=True)
files = sorted(glob.glob(f"{OUT_DIR}/stats_dx_vs_N_matrix_*.txt"))
df = pd.read_csv(files[-1], sep=r'\s+', comment='#', 
                 names=['dx_mm','N','trials','usable_eff','grid_size','Cent_cont_std','Cent_disc_std','FND_std'])

pivot_c = df.pivot(index='N', columns='dx_mm', values='Cent_disc_std').dropna(axis=0, how='all').dropna(axis=1, how='all')
X_dx = pivot_c.columns.values
Y_n = pivot_c.index.values
Z_sig = pivot_c.values

fig = plt.figure(figsize=(9.5, 8.2), dpi=300)
ax3d = fig.add_subplot(111, projection='3d')
X_mesh, Y_mesh = np.meshgrid(np.log10(X_dx), np.log10(Y_n))
surf = ax3d.plot_surface(X_mesh, Y_mesh, np.log10(Z_sig), cmap='plasma',
                         linewidth=0.5, edgecolors='k', alpha=0.90)

# Verified broad face-on orientation
ax3d.view_init(elev=24.0, azim=-215.0)

ax3d.set_xlabel(r'$\log_{10}(dx\ [\mathrm{mm}])$', fontsize=11, labelpad=10)
ax3d.set_ylabel(r'$\log_{10}(N)$', fontsize=11, labelpad=10)
ax3d.set_zlabel(r'$\log_{10}(\sigma_\theta\ [^\circ])$', fontsize=11, labelpad=10)
ax3d.set_title(r'3D Resolution Phase Surface: $\sigma_\theta(dx, N)$ (Face-On)',
               fontsize=13, fontweight='bold', pad=15)

# Horizontal colorbar at bottom eliminates z-axis label overlap
cbar = fig.colorbar(surf, ax=ax3d, orientation='horizontal', shrink=0.55, aspect=18, pad=0.08)
cbar.set_label(r'$\log_{10}(\sigma_\theta\ [^\circ])$', fontsize=10, fontweight='bold')

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12)
out_3d = f"{PLOT_DIR}/fig_phase_space_3d_perfected_{TS}"
fig.savefig(f"{out_3d}.png", bbox_inches='tight')
fig.savefig(f"{out_3d}.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[SUCCESS] Saved perfected 3D phase space: {out_3d}.[png|pdf]")
