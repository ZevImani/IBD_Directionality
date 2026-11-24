# ldm: diffusion generator for IBD events

A denoising diffusion model (DDPM) that generates IBD events, i.e. the displacement
r = capture − vertex, for the direction-reconstruction studies in `emd_paper/`. It is a fork of
CompVis latent-diffusion, but **there is no autoencoder**. One event is 2–3 numbers, so the
diffusion runs directly in data space with a small residual MLP as the denoiser.

```
dataset/*.npy ─► ldm/data/ibd.py ─► main.py (train EventDDPM) ─► sample.py ─► samples/<name>/{vertices,captures}.npy
                                                                                │
                      evaluate.py (quality tests vs held-out real) ◄────────────┤
                      run.py --data ldm/samples/<name>  +  closure.py  ◄────────┘  (closure test)
```

## Quick start (from `ldm/`)

```bash
conda activate ldm; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

sbatch submit_train.sh                                              # train (GPU)
sbatch submit_sample.sh logs/<run>/checkpoints/last.ckpt v1 2000000 # generate 2M events + evaluate.py
cd .. && python run.py configs/ldm_closure.yaml --jobs 16           # closure test, real events
python run.py configs/ldm_closure.yaml --data ldm/samples/v1 --tag gen --jobs 16
python ldm/closure.py ldm_closure ldm_closure_gen                   # sigma_gen / sigma_real per method
```

## What is modelled, and why

Facts about the MC pool that the design rests on (all checked on the full 9M events):

- **The vertex is uniform in a 40 m cube and independent of r.** Correlations are below 0.003. Mean r_x within 400 mm of a wall differs from the bulk by 0.16 mm, and 0.4% of captures are clipped to the wall.
  - So only r is learned. `sample.py` draws vertices uniformly in the same cube, which also makes the cell phase uniform, as in the real pool.
  - Generated events reproduce the real *marginal* p(r), wall-clipped events included. They do not reproduce the tiny vertex–r correlation at the walls, which no method uses.
- **The signal is small.** Mean r = (16.8, 0.0, 0.9) mm on a width of 44 mm. A bias δ in mean r_y rotates the direction by δ/16.8 rad, and a bias in the width scales θ₀ = std(r_y)/mean(r_x) and with it every method's σ_θ. The generator must get the first two moments right to ~0.1%.
- **y → −y is an exact symmetry** (beam along +x): KS p = 0.43, mean r_y = −0.015 ± 0.015 mm.
  - Training flips r_y at random (`flip_y`).
  - `sample.py` randomises the sign of r_y, so the generated mean r_y is exactly 0 in expectation.
  - **The y–z (axial) symmetry does not hold:** mean r_z = +0.90 mm (61σ), and the r_y/r_z KS p = 4e-39. So z is never symmetrised.
- **z is unused downstream.** `dims: 3` (default) models (r_x, r_y, r_z), so files are complete events. `dims: 2` models only (r_x, r_y), which is all `dirreco` and `cnn` use. That is easier to learn, and r_z is then written as 0.
- **Normalisation.** The network sees x = r / 50 mm: one isotropic scale and no centring, so rotations commute with it.

## Model and sampler choices (and the tests behind them)

| Choice | Value | Why |
|---|---|---|
| Denoiser | `EventMLP`: 6 residual blocks, width 512, FiLM time conditioning, ~6M params | Events are vectors; a UNet does not apply |
| Noise schedule | cosine, T = 1000 | With an **exact** Gaussian denoiser, the LDM "linear" schedule (ᾱ_T = 7e-4) biases mean r_x by −0.5 mm at any step count; cosine (ᾱ_T = 2e-9) does not |
| Sampler for paper data | DDIM, 1000 steps, η = 0 (`sample.py` default) or ancestral DDPM (`--ddim-steps 0`) | Exact-denoiser test: DDIM-100 gives std −1.1%, mean r_x −0.26 mm; DDIM-250 gives −0.4% / −0.09 mm; DDIM-1000 is unbiased within 0.1% |
| Monitoring during training | DDIM-250 on 100k events every 2 epochs (`val/gen_*`) | Cheap; read its ~0.4% width deficit as sampler error, not model error |
| Precision | fp32 throughout | Moments are needed to 0.1% |

These sampler-bias numbers are floors, measured with the analytically exact noise predictor. A trained model can only add to them.

The training split matches `cnn/`, so the generator never sees events used by `run.py` studies:

| Events | Use |
|---|---|
| `[0, 2M)` | `eval`: reserved for `run.py` studies (`data: {max_events: 2000000}`) and for `evaluate.py` |
| `[2M, 2.5M)` | validation (loss, checkpoint choice, `val/gen_*`) |
| `[2.5M, end)` | training (~6.5M events) |

## Testing the generated data

Quality is judged at three levels, from cheap to the one that matters for the paper.

**1. During training** (W&B / `logs/<run>/testtube/.../metrics.csv`):

- `val/gen_dmean_{x,y,z}` and `val/gen_dstd_*`: moment offsets in mm.
- `val/gen_w1_*`: 1D Wasserstein distance.
- `val/gen_theta0_50_ratio` and `val/gen_eff50_ratio`.
- `val/gen_angle_deg`: centroid angle of the generated sample.
- `logs/<run>/images/train/marginals_*.png`: marginals with ratio panels.

Pick checkpoints on these, not only on the loss.

**2. `evaluate.py samples/<name>`:**

- **Method.** Two-sample tests against real events the model never saw. Every number is shown next to the same test between two disjoint *real* halves (B vs A), which is what a perfect generator would score at that sample size.
- **Tests:**
  - moments with pulls;
  - **per-pitch usable fraction, θ₀ and centroid angle**;
  - KS/W1 of 6 projections, and tail fractions;
  - χ²/ndf of the 2D (r_x, r_y) histogram and of the detector's cell-hop histogram per pitch;
  - classifier two-sample test (C2ST, gradient-boosted trees, AUC);
  - memorization: nearest-neighbour distance to the 6.5M training events vs unseen real events.
- **Validated on known inputs (1M events):**
  - *Real events posing as generated:* every test matches the baseline, AUC = 0.5001, and the memorization test correctly flags the half of that sample taken from the training split.
  - *r_y stretched by 1%:* caught by the std_y pull (7.8σ), the θ₀ ratio (1.007 vs 1.000), KS on |r| (p = 3e-7) and the cell-hop χ².
  - The C2ST moves only 1.6σ for this distortion, so for small width errors the physics tests are the more sensitive ones.

**3. Closure test** ([configs/ldm_closure.yaml](../configs/ldm_closure.yaml) + [closure.py](closure.py)):

- Run the same study (centroid ×2, FND, sliced EMD; pitch 150/50/5 mm; N = 100–10⁴; K = 200) on real `[0, 2M)` and on 2M generated events, and compare σ_θ per method and grid point (ratio and pull) and the biases.
- This tests exactly what the paper uses. Generate the same pool size as the real run, so finite-pool trial overlap is identical.

## What the generator is (and isn't) good for in this paper

- **Not a replacement for MC at the sizes used so far.** The pool has 9M events. At N ≤ 10⁴ and K ≤ 200, real MC is free and exact.
- **Independent trials at large N.** Every trial draws its N measurement events from one pool of M usable events. Two trials then share ~N²/M events, so for a method without a reference sample (centroid, CNN) the trial-to-trial correlation is ρ ≈ N/M. The sample std of K such trials is low by a factor √(1−ρ) ≈ 1 − N/2M.
  - For example, about 5% at n_total = 1M (`capacity_exposure`), and more for `phase_space`'s N = 500k on the smaller usable pools at large pitch.
  - For template methods, the meas–meas and ref–ref overlaps (positive) and the meas–ref cross-overlaps (negative) cancel at linear order.
  - A generator gives fresh events per trial, valid up to the largest N where the closure test passed.
- **A noiseless template.** FND/EMD draw a reference of N events, so template noise is part of their σ. A reference of 10⁷ generated events removes it, which separates estimator noise from template noise.
- **The learned score.** The model learns ∇ log p of the (noised) displacement distribution, which is exactly what the Fisher information for the direction needs: I = E[(s·Jr)²], with J the 90° rotation. This gives a CRLB cross-check that is independent of [dirreco/fisher.py](../dirreco/fisher.py), which computes the bound directly from the real pool. Taken one step further, a likelihood-ratio / maximum-likelihood direction estimator would be the optimal reference method.
- **Conditional generation**, if MC with other settings exists (neutron energy, detector material, smearing). Conditioning on the setting would let one interpolate between them. The conditioning machinery from the original code (cross-attention, `conditioning_key`) is still in `ddpm.py`.

## Files

| File | What |
|---|---|
| `configs/ibd-ddpm.yaml` | The model/data/trainer config |
| `ldm/data/ibd.py` | Loads the paired pool and splits it; fast batched `IterableDataset`; `IBDDataModule` |
| `ldm/modules/diffusionmodules/mlp.py` | `EventMLP` denoiser |
| `ldm/models/diffusion/event_ddpm.py` | `EventDDPM(DDPM)`: vector inputs, `sample_events()`, per-epoch generation metrics and plots |
| `ldm/ibd_metrics.py` | θ₀, usable cut, centroid angle, marginal comparison figure |
| `main.py` | Training entry point (Lightning); `--resume <run dir or ckpt>` |
| `sample.py`, `evaluate.py`, `closure.py` | Generate; test vs held-out real; compare a study run on real vs generated |
| `submit_train.sh`, `submit_sample.sh` | Slurm wrappers (iaifi_gpu). Their headers give the costs measured on an A100 MIG slice: training 63 s/epoch + 62 s per validation (5.2 h for 200 epochs); sampling 2.2 ms/event at 1000 DDIM steps |

Changes to the shared library code from upstream and the old proton version:

- DDIM/PLMS samplers work for any data shape and on CPU.
- DDIM classifier-free guidance is restored.
- An off-by-one at S = T DDIM steps is fixed.
- The DDPM loss reduction is shape-agnostic.
- The missing `ldm.emd_loss` import and the image-EMD loss types are removed.

The proton/electron/LArTPC configs, loaders, scripts, momentum embedder and renoising tests were removed. They are still in `~/latent-diffusion`.

`main.py` flags: `--base` config(s), `--logdir` (also the W&B project), `--log_wandb`, `--train`, `--gpus 0,`, `--resume`, `--scale_lr False` (the config's LR is absolute), `--seed`. Any `key.sub=value` argument overrides the config.

## Acknowledgements

Fork of [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) ([paper](https://arxiv.org/abs/2112.10752)), which builds on [OpenAI's ADM](https://github.com/openai/guided-diffusion) and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
