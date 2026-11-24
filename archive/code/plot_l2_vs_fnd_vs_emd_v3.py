import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.ndimage import rotate as ndimage_rotate

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

from paper_c_stats_package_v0_6_6_3_all_angles_N10000 import DataProcessor

proc = DataProcessor(sys.argv)
n_sample = 10000
test_angles = np.array([180.0, 160.0, 135.0, 112.5, 90.0, 67.5, 45.0, 22.5, 0.0])
k_eval = 15

dx, gs = 150, 3
proc.set_geometry(dx, gs)

# Physical 1D coordinate axes in mm: [-150, 0, 150]
coords_1d = np.linspace(-(gs//2)*dx, (gs//2)*dx, gs)

# Reference template at theta = 0 deg (Batch 0)
ref_sample = proc.get_disjoint_sample(n_sample, batch_idx=0)
ref_grid = proc.binEvents(ref_sample[:, 0], ref_sample[:, 1])
norm_ref = ref_grid / np.sum(ref_grid)
ref_proj_x = np.sum(norm_ref, axis=0)
ref_proj_y = np.sum(norm_ref, axis=1)

img_l2_m, img_l2_s = [], []
fnd_l2_m, fnd_l2_s = [], []
emd_2d_m, emd_2d_s = [], []

for th in test_angles:
    img_trials, fnd_trials, emd_trials = [], [], []
    for k in range(1, k_eval + 1):
        eval_sample = proc.get_disjoint_sample(n_sample, batch_idx=k)

        # 1. Standard Image-Space L2 (Rotated discrete matrix)
        eval_grid_0 = proc.binEvents(eval_sample[:, 0], eval_sample[:, 1])
        norm_grid_0 = eval_grid_0 / np.sum(eval_grid_0)
        rot_img = ndimage_rotate(norm_grid_0, th, reshape=False, order=1, mode='constant', cval=0.0)
        img_trials.append(np.linalg.norm(norm_ref - rot_img))

        # 2. FND Frobenius Loss (Continuous event vector rotation + fresh binning)
        xr, yr = proc.rotateCoords(eval_sample[:, 0], eval_sample[:, 1], th)
        rot_grid = proc.binEvents(xr, yr)
        norm_rot = rot_grid / np.sum(rot_grid)
        fnd_trials.append(np.linalg.norm(norm_ref - norm_rot))

        # 3. True 2D Spatial EMD (Marginal Coordinate Transport Work in mm)
        rot_proj_x = np.sum(norm_rot, axis=0)
        rot_proj_y = np.sum(norm_rot, axis=1)
        w_x = wasserstein_distance(coords_1d, coords_1d, u_weights=rot_proj_x, v_weights=ref_proj_x)
        w_y = wasserstein_distance(coords_1d, coords_1d, u_weights=rot_proj_y, v_weights=ref_proj_y)
        emd_trials.append(np.sqrt(w_x**2 + w_y**2))

    img_l2_m.append(np.mean(img_trials))
    img_l2_s.append(np.std(img_trials))
    fnd_l2_m.append(np.mean(fnd_trials))
    fnd_l2_s.append(np.std(fnd_trials))
    emd_2d_m.append(np.mean(emd_trials))
    emd_2d_s.append(np.std(emd_trials))

def min_max_norm(m, s):
    mi, ma = np.min(m), np.max(m)
    rng = ma - mi + 1e-12
    return (m - mi) / rng, s / rng

norm_img_m, norm_img_s = min_max_norm(np.array(img_l2_m), np.array(img_l2_s))
norm_fnd_m, norm_fnd_s = min_max_norm(np.array(fnd_l2_m), np.array(fnd_l2_s))
norm_emd_m, norm_emd_s = min_max_norm(np.array(emd_2d_m), np.array(emd_2d_s))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=300)

# PANEL 1: Normalized Recreation (Zev Poster Benchmark)
ax1.plot(test_angles, norm_img_m, "-o", color="#d62728", lw=2.2, ms=6, label=r"Standard Image $L_2$ (Pixel Rotated)")
ax1.fill_between(test_angles, norm_img_m - norm_img_s, norm_img_m + norm_img_s, color="#d62728", alpha=0.15)

ax1.plot(test_angles, norm_fnd_m, "-^", color="#2ca02c", lw=2.2, ms=6, label=r"FND Loss (Continuous Event Rotation)")
ax1.fill_between(test_angles, norm_fnd_m - norm_fnd_s, norm_fnd_m + norm_fnd_s, color="#2ca02c", alpha=0.15)

ax1.plot(test_angles, norm_emd_m, "-s", color="#1f77b4", lw=2.2, ms=6, label=r"Spatial Earth Mover's Distance (EMD / $W_1$)")
ax1.fill_between(test_angles, norm_emd_m - norm_emd_s, norm_emd_m + norm_emd_s, color="#1f77b4", alpha=0.15)

ax1.set_xlim(185, -5)
ax1.set_ylim(-0.05, 1.15)
ax1.set_xlabel(r"Reconstruction Angle Approaching Truth $\theta$ [degrees]")
ax1.set_ylabel(r"Normalized Distance to Truth $\tilde{D}(\theta) \in [0, 1]$ [dimensionless]")
ax1.set_title(r"Zev Benchmark Recreation: Standard $L_2$ vs. FND vs. Spatial EMD" "\n" r"($dx = 150\,\mathrm{mm},\ N = 10{,}000,\ K = 15$)")
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(frameon=True, loc="lower left")

# PANEL 2: Unnormalized Physical Metrics
ax2_twin = ax2.twinx()

p1 = ax2.plot(test_angles, img_l2_m, "--o", color="#d62728", lw=1.8, ms=5, label=r"Raw Image $L_2$ Norm")
p2 = ax2.plot(test_angles, fnd_l2_m, "-^", color="#2ca02c", lw=2.2, ms=6, label=r"Raw FND Frobenius Norm")
p3 = ax2_twin.plot(test_angles, emd_2d_m, "-s", color="#1f77b4", lw=2.2, ms=6, label=r"Raw Spatial 2D EMD")
ax2_twin.fill_between(test_angles, np.array(emd_2d_m) - np.array(emd_2d_s), np.array(emd_2d_m) + np.array(emd_2d_s), color="#1f77b4", alpha=0.15)

ax2.set_xlim(185, -5)
ax2.set_xlabel(r"Reconstruction Angle Approaching Truth $\theta$ [degrees]")
ax2.set_ylabel(r"Frobenius Distance $\|M(\theta) - M_0\|_F$ [prob. units]", color="#2ca02c")
ax2_twin.set_ylabel(r"Optimal Spatial Transport Work [mm]", color="#1f77b4")
ax2.set_title(r"Physical Metric Space (Raw Loss Landscapes)" "\n" r"True Separation Gradients and Baseline Offsets")
ax2.grid(True, linestyle="--", alpha=0.45)

lines = p1 + p2 + p3
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, frameon=True, loc="lower left")

plt.tight_layout()
PLOT_DIR = "../results/plots"  # figures; tables live in review_deliverables (-> ../results/tables)
os.makedirs(PLOT_DIR, exist_ok=True)
out_fig = f"{PLOT_DIR}/fig5c_zev_recreation_l2_vs_fnd_vs_emd_v3.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"Successfully generated: {out_fig}")
