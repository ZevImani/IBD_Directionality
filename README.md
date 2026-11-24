# emd_paper: IBD directional reconstruction (Paper C)

This repo measures how well different estimators recover the neutrino direction from
prompt→capture displacements in a segmented detector. It compares estimators as a function
of segment pitch Δx, event count N, exposure, smearing and so on. The physics and code originate with
Max Dornfest (FROST / Paper C). The 2026-11-16 refactor collapsed his ~25 scripts into one pipeline.

```
dataset/*.npy ──► data + augment ──► reco methods ──► trials.csv ──► summary.csv ──► figures
                 dirreco/data.py     dirreco/methods/   dirreco/engine.py  dirreco/results.py  dirreco/plots.py
                 dirreco/augment.py
                            driven by   configs/<study>.yaml   via   run.py  /  plot.py
```

## Setup

**1. Environment.** Everything (pipeline, CNN, LDM) runs in one conda env, Python 3.8:

```bash
cd ldm && conda env create -f environment.yaml && cd ..   # once; creates the env "ldm"
```

**2. Data.** The MC event pool is 436 MB and is not in git:

| File | Shape | Size |
|---|---|---|
| `fid_vertices.npy` | (8,999,997 × 3) | 216 MB |
| `fid_captures.npy` | (9,999,999 × 3) | 240 MB |

On FASRC (iaifi_lab members), link the shared copy into the repo:

```bash
ln -s /n/holylabs/LABS/iaifi_lab/Lab/emd_paper_dataset dataset
```

Off the cluster, ask the maintainer for `fid_arrays.tar.xz` and unpack it into `dataset/`.

**3. What's in git.** Code, configs, Slurm scripts, docs, Max's original tables (`results/legacy_tables/`), and, for each study, its figures plus `summary.csv`, `config.yaml` and `run.log`.

Not in git, so regenerate it if you need it:
- the dataset
- model weights (`cnn/checkpoints/`)
- LDM training runs
- per-trial `trials.csv` files (rerun the study)
- the `legacy_*` re-plots (`python plot.py legacy:<study>`)

## Quick start

```bash
conda activate ldm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib       # required: system libstdc++ is too old for matplotlib/PIL

python run.py configs/smoke.yaml --jobs 4      # ~30 s, every method, sanity check
python run.py configs/scaling.yaml --jobs 16   # a full study -> results/studies/scaling/
python plot.py scaling                         # re-plot the latest scaling run
python plot.py legacy:capacity_exposure        # plot Max's original tables with the same code
```

`run.py` overrides: `--K`, `--pitch 50,5`, `--N 1000,5000`, `--n-total ...`, `--methods a,b`,
`--seed`, `--jobs`, `--tag`, and `--data DIR`, which runs the study on `DIR/vertices.npy` + `DIR/captures.npy` (paired 1:1, e.g. generated events from `ldm/sample.py`). Results don't depend on `--jobs`, because each grid point has its own seed.
On the cluster:

- `sbatch slurm/reproduce_cpu.sh` runs every CPU study and makes its figures (sapphire, 16 cores, 19 GB, 19 min).
- `sbatch slurm/reproduce_gpu.sh` runs `landscape` and `emd_compare` (iaifi_gpu, 1 A100, 4 cores, 5 GB, 2 h 20 min).

Each request is 1.2× the modelled time and memory. The GPU estimate assumes, without having measured it, that a full A100 is 2.5× faster than the MIG slice the timings came from.

Both take `STUDIES="a b"` to run a subset. The requests come from measured per-method costs; see [slurm/README.md](slurm/README.md).
`run.py` pins BLAS/OpenMP to 1 thread per worker. Multithreaded BLAS inside `--jobs` workers was
measured 200× slower. To give a single process more threads, export `OMP_NUM_THREADS` yourself.

## Layout

| Path | What |
|---|---|
| `run.py`, `plot.py` | The only two entry points |
| `configs/` | One YAML per study (sweep, methods, augmentations, figures) |
| `dirreco/data.py` | Loads the paired vertex/capture pool; `Geometry` (cells, Δx/√2 veto, binning) |
| `dirreco/augment.py` | Input augmentations (currently `smear`) |
| `dirreco/methods/` | Reconstruction methods, one registry |
| `dirreco/engine.py` | Sweep runner: split-half sampling, θ injection, per-trial output |
| `dirreco/results.py` | Summary statistics; readers for runs and legacy tables |
| `dirreco/plots.py` | Every figure, each a function of a run |
| `dataset/` | `fid_vertices.npy`, `fid_captures.npy` (MC truth, mm; `captures[3:]` ↔ `vertices[:]`). Not in git; see [Setup](#setup) |
| `results/studies/<name>/` | One folder per study, replaced on rerun: `plots/*.png`, and `logs/` with `config.yaml` (resolved), `trials.csv` (every estimate; local only, not in git), `summary.csv` (σ, bias, Θ, η per grid point) and `run.log` (all output, with start/end time, host and Slurm job id). Runs with CLI overrides go to `<name>_custom` (ignored by git) or `<name>_<tag>`. |
| `results/slurm_logs/` | Slurm job output (`<job name>-<job id>.out`, not in git). Not read by any code; useful when a job fails before Python starts |
| `results/legacy_tables/` | Max's original `.txt` tables (read-only inputs for `plot.py legacy:*`) |
| `results/plots/` | Data-only figures (`plot.py data --figure hits`), and `legacy_*/` figures from the legacy tables (not in git) |
| `archive/` | Pre-refactor scripts, the old README, Max's docs and audits (in git); run logs and tarballs (local only) |
| `cnn/` | ResNet direction CNN: training (`python -m cnn.train`), shared image builder. See `cnn/README.md` |
| `ldm/` | Diffusion generator for IBD events (DDPM on the displacement, no autoencoder), its quality tests (`evaluate.py`) and the closure test (`closure.py`). See `ldm/README.md` |
| `dirreco/fisher.py` | Data-driven Cramér–Rao bound on θ per pitch, continuous and cell-hop readout (`python -m dirreco.fisher` → `results/crlb/`) |

## How a run works

For each grid point (augmentation × pitch × N or n_total × θ_true), the engine:

1. **Augments** the pool. `apply_to: both` augments measurement and reference alike; `apply_to: meas` leaves the template clean.
2. **Cuts** usable events (|r| > Δx/√2) and records `eff`. It also records `theta0` = σ⊥/r̄ of that sample, the centroid's asymptotic Θ.
3. **Runs K trials.** Each trial draws 2N disjoint usable events: N form the reference at θ = 0, and N form the measurement, rotated CCW by θ_true.
4. **Estimates.** Every method returns θ̂, and the engine writes one row per trial and method to `trials.csv`.

`summary.csv` holds, per grid point and method: `K mean bias sigma sigma_err sigma_circ rho Theta Theta_err Theta_exp eta`.
σ is the ddof=1 std of θ̂ − θ_true about its circular mean, and `sigma_err` = σ/√(2(K−1)).
Θ = σ√N, Θ_exp = σ√n_total and η = (theta0/Θ)².
Because the per-trial estimates are stored, new figures of merit don't require a rerun. This only holds where `trials.csv` exists: it isn't in git, so in a fresh clone you have to rerun the study. The figures that need it are `methods_compare` and `pair`.

## Reconstruction methods

| `type` | Options | Notes |
|---|---|---|
| `centroid` | `discrete` | atan2 of the mean displacement. `discrete: true` snaps vertex and capture to cell centres |
| `fnd` | `scan`, `refine` | Frobenius norm of 2D count-histogram differences (Max's "continuous FND") |
| `emd_paired` | `scan`, `refine` | Max's "W1 proxy" mean‖R(φ)mᵢ − rᵢ‖ with **random index pairing**. An upper bound, not W1 |
| `emd_exact` | `scan`, `refine`, `unbalanced` (null \| `partial` \| `kr`), `mass` (0.9), `lam` (mm), `max_N` (2000) | **New.** Exact 2D W1 (POT network simplex). `kr` = exact unbalanced W1 (Kantorovich–Rubinstein: create/destroy mass at `lam` per unit). `partial` = exact partial OT moving only `mass` |
| `emd_sinkhorn` | `scan`, `refine`, `p` (1\|2), `blur` (mm), `scaling` (0.9), `debias`, `reach` (mm), `device`, `backend` | **New.** Entropic W_p / Sinkhorn divergence (geomloss, GPU). `reach: <mm>` = unbalanced (KL marginals) |
| `emd_sliced` | `scan`, `refine`, `n_proj`, `reduce` (mean\|max) | **New.** Sliced W1 on unbinned points; `reduce: max` = max-sliced W1 |
| `emd_sliced_kr` | `scan`, `refine`, `n_proj`, `lam` (mm), `bin_width` (mm), `device` | **New.** Sliced unbalanced W1 (`suw1`): exact 1D KR per direction via the max-plus DP (torch, GPU) |
| `mmd` | `scan`, `refine`, `kernel` (energy\|gaussian\|laplacian), `blur` (mm) | **New.** Kernel-distance baselines (geomloss); not optimal transport |
| `image_l2` | `scan`, `refine` | L2 between histograms, with the measurement rotated as an image |
| `cnn` | `checkpoint` (path or list, one per pitch), `device` | **New.** ResNet trained by `cnn/train.py`; no reference sample. Needs `data: {max_events: 2000000}` (train/eval split) |

The scan options work like this:

- `scan: [lo, hi, step]` is inclusive. `[-180, 179, 1]` is the full circle; `[-15, 15, 1]` is Max's clipped window.
- `refine` is one of:
  - `argmin`: the best grid point;
  - `parabolic`: a 3-point vertex fit around it;
  - `sine`: Gabe's global fit a|sin((φ−φ₀)/2)|+c.

EMD notes (code: [dirreco/methods/emd.py](dirreco/methods/emd.py)):

- **Origin.** The family is ported from the proton/electron inference project (`inference_loop/protoninfer/losses.py`, `.../unbalanced_sliced/unbalanced_study.py`):

  | Proton project | Here |
  |---|---|
  | emd/w1, uemd, w2 | `emd_sinkhorn` (`reach`, `p: 2`) |
  | sw | `emd_sliced` |
  | suw1 | `emd_sliced_kr` |
  | exact_w1, exact_kr | `emd_exact` (`kr`) |
  | energy/gmmd/lmmd | `mmd` |

  Max-sliced (`reduce: max`) and Sinkhorn without debiasing (`debias: false`) were on that project's "not yet tried" list.
- **Not ported:**
  - `sw_mass`: here measurement and reference both have N events, so it equals `emd_sliced` exactly.
  - POT's `sinkhorn_unbalanced2`: the same objective as `emd_sinkhorn` + `reach`, but not log-domain, so it underflows at mm-scale costs.
  - POT's non-entropic KL solver `mm_unbalanced2`: 75 s per estimate at N=500, and it returned −11° for a 10° injection.
- **Mass and λ.** Every event carries mass 1/N_ref, as the proton losses divide by the target mass. With equal N the totals match, but unbalanced OT still differs from balanced: moving mass farther than about 2λ (or the reach) costs more than destroying and re-creating it, so the far tail stops pulling on the fit.
  - `lam` and `reach` are lengths in mm. Usable displacements have median |r| of 50–120 mm and a maximum of about 370 mm, so `emd_compare` uses 30 mm (aggressive) and 100 mm (far tail only).
- **Validation (2026-11-16, N=300 real displacements):**
  - The 1D KR dynamic programme matches the exact LP to 0.2% (0.25 mm bins).
  - Exact KR with λ→∞ equals exact W1 exactly.
  - Sliced KR with λ→∞ matches sliced W1 to 0.1%.
  - Sinkhorn with huge `reach` matches balanced Sinkhorn to 0.05%.
  - Every variant recovers a 10° injection within noise.
  - Sinkhorn's distance to exact W1 is set by `scaling`, not by blur or float precision: the loss-curve shape error over φ is 0.43 mm at 0.9 and 0.21 mm at 0.95, on a 6 mm landscape.
- **Speed.** `emd_sliced_kr` costs grow with λ/`bin_width`, the number of DP levels: 0.03 s per estimate at λ=30 mm, but 88 s at λ=10⁴ on CPU.
- **Continuous clouds only.** All EMD methods act on the continuous displacement clouds. Cell-centre points don't work with a rotation scan: they sit on the lattice, so the loss has a spurious minimum at φ = 0. In a test, θ̂ = 0.04° for a 10° injection.
- Timing on an A100 MIG 3g.20gb slice, per scan angle (Sinkhorn at `scaling` 0.7; 0.9 needs about 3.4× more iterations):

  | Method | N | Time per angle |
  |---|---|---|
  | `emd_sinkhorn` | 1,000 | 6 ms |
  | `emd_sinkhorn` | 5,000 | 0.15 s |
  | `emd_sinkhorn` | 10,000 | 0.6 s |
  | `emd_exact` (CPU) | 500 | about 0.05 s |

  With `reach`, Sinkhorn costs the same. Exact OT grows roughly as N³.

Methods with an identical loss share one loss evaluation per trial, so `fnd_parab` + `fnd_sine` cost one FND.
Any method also accepts `max_N`, `min_pitch` and `max_grid`; where these exclude it, its rows are skipped.

**Adding a method** takes one file in `dirreco/methods/` plus an import line in `methods/__init__.py` (`dirreco/methods/cnn.py` is a complete example):

```python
from . import Method, register

@register("cnn")
class CNN(Method):
    needs_ref = False
    def __init__(self, checkpoint, **common):
        super().__init__(**common)
        self.model = load_model(checkpoint)           # torch is in the ldm env
    def estimate(self, meas, ref, geom):              # meas/ref: Events (.vtx .cap .disp), geom: Geometry
        img = geom.hist(meas.disp[:, 0], meas.disp[:, 1])   # or any representation you like
        return float(self.model(img))                 # degrees, CCW from +x
```

Then reference it from a config, e.g. `cnn_v1: {type: cnn, checkpoint: path/to.pt}`.
A template-matching method should instead subclass `template.ScanMethod` and implement only `loss(meas, ref, geom)`; it then gets scanning, refinement and landscape output for free.
**Adding an augmentation** works the same way: decorate a function with `@register` in `augment.py` (see the docstring there).
To add a figure, decorate a function with `@figure` in `plots.py`.

## Studies

| Config | Replaces (now in `archive/code/`) | Figures |
|---|---|---|
| `capacity_exposure` | `run_production_exposure_suite_v3_0` + `plot_final_paper_figures_v3_0` | capacity, exposure (Figs 3–4) |
| `scaling` | `run_unclipped_universal_scaling_v4_0`, `run_emd_sparsity_sweep_v1_0` + `plot_unclipped_universal_scaling_v6_0` | scaling (Fig 5) |
| `fnd_refine` | `run_gabe_vs_parabolic_full_suite_v1_0` + `plot_gabe_benchmark_suite_v1_0` | capacity, exposure, pair |
| `smearing` | `run_smearing_sensitivity_suite_v1_0` + `plot_smearing_sensitivity_v1_0` | smearing |
| `phase_space` | `run_pitch_vs_N_matrix_sweep_v1_0`, `run_dense_kink_sweep_v1_0` (see config comment) + `plot_3d_phase_space_perfected` | phase3d |
| `angle_sweep` | `paper_c_stats_package_v0_6_6_4 --mode angle-sweep` + `plot_full360_publication_v5` | full360 |
| `landscape` | `compute_and_plot_zev_benchmark_v3`, `plot_l2_vs_fnd_vs_emd_v3` | landscape |
| `emd_compare` | — (new: 13 EMD/OT/kernel variants (Sinkhorn: balanced and reach100 only), balanced and unbalanced, vs FND/centroid; GPU; results in issue 1) | capacity_balanced, capacity_unbalanced, capacity_variants, scaling |
| `cnn_capacity` | — (new; train first, see `cnn/README.md`) | capacity |
| `ldm_closure` | — (new; run on real, then with `--data ldm/samples/<name> --tag gen`, then `ldm/closure.py`) | capacity, closure |
| — | `plot_detector_cross_sections_v1_0` | `plot.py data --figure hits` |
| — | `compute_crlb_and_parabolic_v1_0` | now the `theta0` column of every run |

Not ported:

- `append_emd_to_pitch_sweep`, whose marginal-W1 number is superseded by `emd_sliced`/`emd_exact`.
- The 12-test statistics suite (KS, Kuiper, KL, …) of the old `DataProcessor`.
- Gabe's toy loss-basin illustration, which used synthetic numbers.

## Validation of the refactor (2026-11-16)

- **Same estimator code.** On identical samples, `centroid` (both variants), `fnd` (argmin, and parabolic on both the clipped and full-circle scans) and `emd_paired` reproduce Max's inline code exactly; only `emd_paired` shows 1e-14 rounding differences.
- **Same statistics.** A rerun of `capacity_exposure` (3 pitches × 3 n_total, K=25) reproduces Max's N_usable exactly. Its σ values agree with his table within the K=25 errors (all pulls < 2.5σ).
- **Sign convention.** Injecting θ_true = 30° is recovered with the right sign by every method (checked with `configs/smoke.yaml`).

## Deliberate differences from Max's code

- **One rotation convention (CCW) everywhere.** This removes the centroid sign flip seen in his 360° sweep (old issue 3).
- **Sine-fit sign.** The ported `refine: sine` returns −φ₀. Max's version returned +φ₀, so in `stats_gabe_vs_parab_residuals_*` `th_sine ≈ −th_parab`. The "Yepez residual" figure therefore shows ≈2|θ̂|, not a parab-vs-sine difference. His σ columns are unaffected, since a sign flip doesn't change a spread.
- **σ uses ddof=1** (his tables mostly used ddof=0, which is 2% lower at K=25), and each figure draws a **per-pitch Θ0 line** instead of the constant 129.58. Legacy plots fall back to 129.58 and say so in the legend.
- **Smearing.** N counts events *after* the cut, and the discrete centroid hops the smeared vertex and capture cells. Max instead drew N before the cut and quantised the displacement.
- **Angle sweep uses one dataset for all angles.** `sweep.same_samples: true` in `angle_sweep.yaml` makes any angle from the same data, as Max did, by rotating the detector bins relative to the point cloud. A side effect of the 1° full-circle FND scan: θ_true and the scan angle combine, so each sample's loss curve only shifts. FND's σ therefore depends only on θ_true mod 1°. Max's table has exactly two FND σ values per pitch (150 mm: 0.748 at integer θ, 0.800 at half-integer θ). The flat FND σ in `full360` reflects the scan grid, not measured lattice isotropy. The discrete centroid has no scan, so its curve does probe the lattice orientation.

## Open physics issues

These are carried over from the old README; the details are in `archive/README_2026-11-16.md`.

1. **`emd_paired` is not W1.** The real OT estimators, balanced and unbalanced, now exist and are validated against their exact references (see EMD notes).
   **`emd_compare` results (2026-11-17, Slurm job 61323105, K=25):** σ_θ [deg] at N=10k:

   | Method | 150 mm | 50 mm | 5 mm | 1 mm |
   |---|---|---|---|---|
   | `centroid_disc` | 1.05 | 1.54 | 1.93 | 1.04 |
   | `emd_sliced` / `emd_sliced_kr100` | 1.49 | 1.68 | 2.35 | 1.92 |
   | `mmd_energy` | 1.47 | 1.71 | 2.37 | 1.96 |
   | `emd_sinkhorn` | 1.49 | 1.84 | 2.61 | 2.01 |
   | `emd_sinkhorn_reach100` | 1.73 | 2.24 | 3.05 | 2.23 |
   | `fnd_parab` | 1.64 | 2.52 | 8.89 | 7.83 |
   | `emd_paired` | 2.46 | 4.67 | 7.19 | 5.86 |

   - **Real OT works and paired EMD does not.** The real OT estimators beat `emd_paired` by 1.7–3.6×. Sliced W1 matches Sinkhorn, and at N ≤ 1k it also matches exact W1 within errors. It costs about 0.1 s per estimate.
   - **Unbalanced variants don't help.** KR with λ = 100 mm is identical to balanced. KR30, reach100 and especially partial-90% are equal or worse.
   - **Nothing beats the centroid.** The OT methods trail the discrete centroid by 1.2–1.9×.
   - **Loss shapes.** `landscape` shows that Sinkhorn, energy MMD and `emd_paired` have a wide, quadratic minimum, while FND and sliced W1 rise roughly linearly. `image_l2` has a false minimum at 5 mm.
   - With K=25, σ is only known to ±14%, so treat differences under about 30% as noise.
2. **K = 25 is too few.** Use K ≥ 200 for paper numbers; `sigma_err` is now reported.
3. **`theta0` is not a true bound.** It is the centroid's asymptotic error, not a CRLB; the centroid reaches η ≥ 1 in places.
   **Now computed (2026-11-16):** `python -m dirreco.fisher` (or `sbatch slurm/crlb.sh`) gives the Cramér–Rao bound Θ_CRLB [deg·√N_usable] from the full pool, in `results/crlb/crlb.csv`.
   - *Continuous readout* (exact r_xy): 146.6 (no cut), 128.8 (50 mm), 108.1 (100 mm), 95.1 (150 mm). That is 1–2% below `theta0`, so the continuous centroid is 96–99% efficient (η vs the CRLB).
   - *Cell-hop readout* (what `centroid_disc` and the detector see), resolved by the 9M pool only at ≥ 50 mm: 138.5 (50 mm), 126.0 (100 mm), 116.8 (150 mm).
   - The existing centroid numbers agree with these bounds within their K = 20–25 errors. FND reaches η ≈ 0.3–0.65 and `emd_paired` 0.1–0.3 of the cell-hop bound.
   - Validation: on Gaussian displacements the continuous bound equals `theta0` to 0.2% at every pitch, as it must.
4. **FND has a 1–2° bias** in Max's 360° sweep. Rerun `angle_sweep` to check it with the new convention.
5. **The audit used unpaired data.** `archive/audit/w1_audit_output.txt` Parts 2–4 loaded `fid_captures.npy` alone, so their numbers are invalid.
6. **FND fails at fine pitch (new, `emd_compare`).**
   - At Δx = 5 and 1 mm, `fnd_parab` gives σ ≈ 8° at both N = 1k and N = 10k, with a bias of about −1°. Every other method improves roughly as 1/√N.
   - At 150 and 50 mm FND behaves normally.
   - Suspects: the ±15° 1° scan grid and parabolic refinement when the histograms are very sparse (ρ ≪ 1). Check this before using FND numbers at fine pitch.
