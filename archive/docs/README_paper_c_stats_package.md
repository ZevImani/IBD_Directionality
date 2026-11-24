# Paper C Analysis & Statistical Verification Suite (`v0_6_6_4_2_0`)

**Directional Reconstruction & Optimal Spatial Transport for Segmented Antineutrino Detectors**  
*Department of Physics and Astronomy, University of Hawaiʻi at Mānoa (FROST / Paper C Collaboration)*

---

## 1. Overview & Purpose

This repository contains the post-processing, statistical analysis, and verification suite supporting **Paper C**, which evaluates Inverse Beta Decay (IBD) directional reconstruction across segmented scintillator lattices ($\Delta x \in \{5, 50, 150\}\,\text{mm}$ within a $450 \times 450\,\text{mm}^2$ active fiducial cross-section).

The codebase benchmarks three primary directional estimators against theoretical Cramér–Rao Lower Bounds (CRLB) and discrete cell quantization limits:
1. **Vector Centroid Estimators:** Both continuous unbinned displacement centroids and hardware-quantized (cell-center hopping) centroids.
2. **Frobenius Norm of Difference (FND) Template Matching:** Continuous event-vector rotation followed by 2D lattice histogramming, evaluated via both **3-point local parabolic vertex interpolation** and **global analytical $\vert{}\sin\vert{}$ CFND curve fitting**.
3. **2D Optimal Spatial Transport ($W_1$ / Earth Mover's Distance):** Evaluated directly in physical millimeter coordinates ($\mathbb{R}^2$) to eliminate grid-unrolling sawtooth artifacts and fine-mesh sparsity flattening.

---

## 2. Repository Structure

```text
mtc-b_scripts/
├── README_paper_c_stats_package.md                    # This documentation
├── paper_c_stats_package_v0_6_6_4_2_0.py              # Main MPI-capable DataProcessor & CLI driver
├── paper_c_stats_package_v0_6_6_4.py                  # Prior v0.6.6.4 snapshot
├── paper_c_stats_package_v0_6_6_3_all_angles_N10000.py# Full-360° N=10k reference sweep script
│
├── # --- Production Sweeps & Estimator Benchmarks ---
├── run_production_exposure_suite_v3_0.py              # Capacity (N_usable) & Exposure (n_total) sweeps
├── run_gabe_vs_parabolic_full_suite_v1_0.py           # 3-point parabolic FND vs. global |sin| CFND
├── run_emd_sparsity_sweep_v1_0.py                     # Fine-mesh sparsity stress test (W1 vs. FND)
├── run_smearing_sensitivity_suite_v1_0.py             # Position resolution smearing sensitivity
├── run_dense_kink_sweep_v1_0.py                       # High-resolution angular kink audit
├── run_pitch_vs_N_matrix_sweep_v1_0.py                # 2D parameter sweep over pitch dx and count N
├── run_unclipped_universal_scaling_v4_0.py            # Unclipped dimensionless scaling sweep
│
├── # --- Analytical Audits & Augmentation ---
├── compute_crlb_and_parabolic_v1_0.py                 # Analytical CRLB & 3-point parabolic verification
├── append_emd_to_pitch_sweep.py                       # Appends 2D spatial W1 (mm) to pitch sweep tables
├── compute_and_plot_zev_benchmark_v3.py               # Authentic unbinned L2 vs. FND vs. W1 benchmark
│
├── # --- Publication Figure Generators ---
├── plot_final_paper_figures_v3_0.py                   # Core Paper C summary figures
├── plot_unclipped_universal_scaling_v6_0.py           # Universal scaling collapse plots
├── plot_l2_vs_fnd_vs_emd_v3.py                        # Loss landscape comparison (L2, FND, 2D W1)
├── plot_full360_publication_v5.py                     # Full [-180°, 180°] angular response
├── plot_3d_phase_space_perfected.py                   # 3D (dx, N, sigma_theta) performance surface
├── plot_gabe_benchmark_suite_v1_0.py                  # Parabolic vs. |sin| CFND comparison plots
├── plot_smearing_sensitivity_v1_0.py                  # Smearing degradation curves
├── plot_detector_cross_sections_v1_0.py               # Lattice segmentation geometry schematics
│
├── audit/                                             # Verified numerical audit logs
│   ├── audit_kinematics_output.txt                    # Uncut/post-cut CRLB & cell quantization tables
│   └── w1_audit_output.txt                            # Codebase scan & 2D W1 formulation verification
├── data/                                              # Baseline reference tables
├── logs/                                              # Execution logs from mtc-b production runs
└── review_deliverables/                               # Generated .txt data tables & .png/.pdf figures
```

> **Note on `audit/` Directory:** On the `mtc-b` cluster, `.py` entries inside `audit/` were symlinks pointing to the parent directory. All executable Python scripts reside directly in the root `mtc-b_scripts/` directory.

---

## 3. Data Prerequisites & Setup

### 3.1. Pre-Processed Binary Event Arrays

To avoid re-parsing multi-gigabyte RATPAC2 ASCII tracking outputs (`neutrons.txt` at 38 GB and `truth.txt` at 821 MB), all Python scripts ingest two pre-filtered NumPy binary arrays generated from the 10M-event reference simulation (`res_10M_001wt_ref`):

* **`fid_vertices.npy`** (~206 MB): $N \times 3$ array of prompt positron annihilation vertices $(x, y, z)$ in mm.
* **`fid_captures.npy`** (~229 MB): $N \times 3$ array of delayed neutron capture coordinates $(x, y, z)$ in mm.

Unpack `fid_arrays.tar.xz` directly into the root `mtc-b_scripts/` working directory so that `fid_vertices.npy` and `fid_captures.npy` resolve via relative paths:

```bash
tar -xJvf fid_arrays.tar.xz
```

### 3.2. Event Index Synchronization & Fiducial Acceptance Cut

Inside `DataProcessor.readData()` (`paper_c_stats_package_v0_6_6_4_2_0.py`), two critical preprocessing steps are applied automatically:

1. **1:1 Event Alignment (`+3` Capture Offset):** Delayed capture entries are offset by 3 indices relative to prompt vertices, yielding **`8,999,997`** synced displacement vectors $\vec{r}_k = \vec{r}_{\text{capture},k} - \vec{r}_{\text{vertex},k}$:

```python
n = min(len(self.vertices), len(self.captures) - 3)
self.captures = self.captures[3:3 + n]
self.vertices = self.vertices[:n]
self.coords = self.captures - self.vertices
```

2. **Segmented Acceptance Filter (Prompt-Cell Veto):** To exclude events where both prompt and delayed signals occur inside the central interaction cell, events must exceed the half-diagonal radius $R = \Delta x / \sqrt{2}$:

$$\Delta x_k^2 + \Delta y_k^2 > \left(\frac{\Delta x}{\sqrt{2}}\right)^2$$

### 3.3. Python Dependencies

Requires **Python 3.8+** with standard scientific packages:

```bash
pip install numpy pandas scipy matplotlib astropy tqdm
```

Optional packages (gracefully bypassed via defensive imports if absent): `mpi4py` (multi-core MPI execution) and `pycircstat2` (Watson $U^2$ and $V$-test circular hypothesis tests).

---

## 4. Quick Start & Execution Commands

### 4.1. Core Statistical Package (`paper_c_stats_package_v0_6_6_4_2_0.py`)

Supports both single-process execution and `mpiexec` parallelization across candidate rotation angles:

* **Full $360^\circ$ Angular Sweep ($N = 10{,}000$, $K = 5$ disjoint trials):**
```bash
python3 paper_c_stats_package_v0_6_6_4_2_0.py \
  --mode angle-sweep \
  --n-events 10000 \
  --theta-min -180 --theta-max 179 --theta-step 1.0 \
  --geoms 150,50,5 \
  --trials 5 \
  --tag full360
```

* **Event-Count Scaling Sweep ($\theta_{\text{true}} = 10^\circ$):**
```bash
python3 paper_c_stats_package_v0_6_6_4_2_0.py \
  --mode count-scaling \
  --n-events 100,500,1000,5000,10000,50000,100000 \
  --meas-angle 10.0 \
  --geoms 150,50,5 \
  --trials 10 \
  --tag scaling
```

* **MPI Multi-Core Execution (e.g., 16 ranks):**
```bash
mpiexec -n 16 python3 paper_c_stats_package_v0_6_6_4_2_0.py --mode angle-sweep --n-events 10000 --trials 10 --tag mpi_run
```

### 4.2. Production Sweeps & Audits

Run any standalone benchmark or audit script directly from `mtc-b_scripts/`:

```bash
# 1. Verify analytical CRLB constants & 3-point parabolic FND refinement
python3 compute_crlb_and_parabolic_v1_0.py

# 2. Run main journal exposure & capacity scaling suite (K = 25 trials)
python3 run_production_exposure_suite_v3_0.py

# 3. Benchmark 3-point parabolic FND vs. global |sin| CFND fit
python3 run_gabe_vs_parabolic_full_suite_v1_0.py

# 4. Generate Zev loss-landscape benchmark (Image L2 vs. FND vs. 2D Spatial W1)
python3 compute_and_plot_zev_benchmark_v3.py
```

---

## 5. Mathematical & Algorithmic Notes (Code-to-Paper Mapping)

### 5.1. 2D Spatial $W_1$ (EMD) vs. Legacy 1D Flattened EMD

* **Why `paper_c_stats_package_v0_6_6_4_2_0.py` retains `wasserstein_distance(flat_k, flat_ref)`:** Inside the base `DataProcessor` class, the `EMD` column evaluates `scipy.stats.wasserstein_distance` on flattened 1D bin-count arrays (`flat_k`, `flat_ref`). As analyzed in Appendix C of Paper C, unrolling a 2D square matrix into a 1D vector breaks spatial topology and induces periodic $C_{4v}$ sawtooth spikes every $45^\circ$ and $90^\circ$.
* **Authentic 2D Spatial $W_1$ Implementation:** All 2D optimal transport results in the paper and figures are computed in physical millimeter space ($\mathbb{R}^2$) without flattening pixel intensities:
  1. **Exact 1D Marginal Coordinate Transport (`append_emd_to_pitch_sweep.py`, `plot_l2_vs_fnd_vs_emd_v3.py`):** Rotates measured vectors by $\mathbf{R}(\phi)$ and evaluates 1D Wasserstein distances along orthogonal $x$ and $y$ coordinate projections (`w1_x`, `w1_y` in mm) via sorted order statistics.
  2. **Pointwise Paired Transport Proxy (`compute_and_plot_zev_benchmark_v3.py`, `run_production_exposure_suite_v3_0.py`):** Evaluates mean Euclidean transport displacement $\frac{1}{N}\sum_{k=1}^N \|\mathbf{R}(\phi)\vec{r}_k - \vec{r}_{0,k}\|_2$ in mm for rapid multi-trial optimization.

### 5.2. Sub-Degree Parabolic FND Refinement

To eliminate $1^\circ$ angular grid quantization when locating the minimum of the discrete Frobenius norm loss profile $D(\phi_i)$, `compute_crlb_and_parabolic_v1_0.py`, `run_production_exposure_suite_v3_0.py`, and `run_gabe_vs_parabolic_full_suite_v1_0.py` apply a 3-point parabolic vertex interpolation around $i_{\min} = \arg\min_i D(\phi_i)$:

$$\Delta\phi = -\frac{1}{2}\frac{D(\phi_{i_{\min}+1}) - D(\phi_{i_{\min}-1})}{D(\phi_{i_{\min}+1}) - 2D(\phi_{i_{\min}}) + D(\phi_{i_{\min}-1})}, \qquad \hat{\theta}_{\text{FND}} = -(\phi_{i_{\min}} + \Delta\phi)$$

---

## 6. Audited Reference Benchmarks (`audit/`)

Values below are verified directly against `fid_vertices.npy` and `fid_captures.npy` in `audit/audit_kinematics_output.txt`:

* **Uncut Pool ($N = 8{,}999{,}997$ synced events):**
  * Mean forward displacement: $\bar{r}_{\text{fwd}} = 16.8249\,\text{mm}$
  * Transverse dispersion: $\sigma_\perp = 43.9523\,\text{mm}$
  * Uncut asymptotic CRLB constant: $\Theta_{0,\text{uncut}} = \sigma_\perp / \bar{r}_{\text{fwd}} = 149.68^\circ\sqrt{N}$

| Pitch $\Delta x$ | Cut $R = \Delta x/\sqrt{2}$ | Surviving Events ($N_{\text{use}}$) | Efficiency $\epsilon$ | Post-Cut $\bar{r}$ | Post-Cut $\sigma_\perp$ | Cut-Sample $\Theta_{0,\text{cut}}$ | Quantization Penalty ($N=10\text{k}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$5.0\,\text{mm}$** | $3.54\,\text{mm}$ | $8{,}963{,}306$ | $99.59\%$ | $16.89\,\text{mm}$ | $44.04\,\text{mm}$ | $149.38^\circ\sqrt{N}$ | $+0.0008^\circ$ |
| **$50.0\,\text{mm}$** | $35.36\,\text{mm}$ | $6{,}296{,}259$ | $69.96\%$ | $22.50\,\text{mm}$ | $51.38\,\text{mm}$ | $130.86^\circ\sqrt{N}$ | $+0.0492^\circ$ |
| **$150.0\,\text{mm}$** | $106.07\,\text{mm}$ | $691{,}040$ | $7.68\%$ | $52.49\,\text{mm}$ | $87.66\,\text{mm}$ | $95.69^\circ\sqrt{N}$ | $+0.1186^\circ$ |

---

## 7. Authorship & Acknowledgments

* **Lead Developer & Maintainer:** Max A. A. Dornfest (`dornfest@hawaii.edu`), University of Hawaiʻi at Mānoa.
* **Contributors:** Jeffrey G. Yepez (MPI parallel architecture) and Gabriel ("Gabe", Processor v2.0/v2.1 CFND formulations).
* **Advisors & Collaborators:** Prof. John Learned (UH Mānoa) and Dr. Viacheslav Li (Lawrence Livermore National Laboratory).