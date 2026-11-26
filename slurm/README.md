# Cluster runs: sizing notes

The resource requests in the two submit scripts come from measured timings and memory (2026-11-16):

- **Timings:** one estimate of every method at the expensive grid points of each config, on job 61267943 (`gpu_test`: an A100 MIG 3g.20gb slice and 1 core).
- **Totals:** those timings were summed over every config's real task grid, using the engine's own task expansion, and counting shared loss curves once.
- **Model check:** a reduced `emd_compare` run took 160 s against 180 s predicted.
- **Wall time:** the per-task costs were scheduled the way the engine runs them (largest N first, `imap_unordered` over the workers). Workers running GPU code at the same time share the GPU. Each study adds about 45 s for imports, event loading and plots, and each GPU worker adds a 60 s CUDA warm-up.
- **Requests:** time and memory are 1.2× the expected values.

| Script | Studies | Compute | Expected | Request |
|---|---|---|---|---|
| `reproduce_cpu.sh` | capacity_exposure, scaling, fnd_refine, smearing, phase_space, angle_sweep | 2.6 CPU-h (2.3 of it angle_sweep) | 15.7 min, 15.4 GB | sapphire, 16 cores, 19 GB, 19 min |
| `reproduce_gpu.sh` | landscape, emd_compare | 4.5 GPU-h on a MIG slice + 0.6 CPU-h | 1.94 h, 4.1 GB | iaifi_gpu, 1 A100, 4 cores, 5 GB, 2 h 20 min |
| `reproduce_gpu.sh`, 4-GPU variant | same | same | 43 min | `--gres=gpu:4 -c 5 --mem=7G -t 00:52:00 --export=ALL,JOBS=4` |
| `emd_crlb.sh` | emd_crlb (timed 2026-11-25 on job 63595732: a K = 2 run of the full grid took 400 s on one core; summed PSS with 4 workers 4.0 GB) | 11.1 CPU-h (exact W1 at N = 1k ~4, large-template sliced W1 on CPU torch ~5) | 33 min on 24 workers, 20.4 GB | sapphire, 24 cores, 25 GB, 45 min |
| `crlb.sh` | — (`python -m dirreco.fisher`: Gaussian self-test + real pool → `results/crlb/`) | ~0.25 CPU-h | ~4 min, 1.8 GB (measured) | sapphire, 4 cores, 6 GB, 20 min |

**Measured on the first full runs (2026-11-16/17, `seff`):**

| Job | Expected | Wall time | Peak memory | CPU efficiency |
|---|---|---|---|---|
| `reproduce_cpu.sh` (61323086) | 15.7 min, 15.4 GB | 11.4 min | 7.05 GB | 69% |
| `reproduce_gpu.sh` (61323105, 1 A100-80GB) | 1.94 h, 4.1 GB | 1.99 h (landscape 15 min, emd_compare 1 h 44 min) | 3.98 GB | 71% |
| `crlb.sh` (61359345) | ~4 min, 1.8 GB | 4.8 min | 1.26 GB | — |

The GPU job came in close to the model: the assumed 2.5× full-A100 / MIG speed-up is about right, perhaps slightly lower. The 2 h 20 min request leaves only about 15% headroom, so raise `-t` before increasing K. The CPU job used less than half its memory request, so `--mem` could drop to about 9 GB.

The model assumed a full A100 is 2.5× faster than the 3g.20gb MIG slice the timings came from. That is the ratio of SMs (108 vs 42) and of memory bandwidth. With 20% headroom, the job only fits if the speed-up is at least about 2.1×:

| Speed-up | 1-GPU job | 4-GPU job |
|---|---|---|
| 2.5× (assumed) | 1.94 h | 43 min |
| 2.0× | 2.36 h (times out) | 50 min |
| 1.0× (no gain) | 4.6 h | — |

The first run's log shows the per-task seconds. If the speed-up comes in low, raise `-t` before rerunning.

The CPU timings come from one core of a gpu_test node. They assume sapphire cores are no slower when all 16 workers are busy.

Where the GPU time goes in `emd_compare` (MIG-slice GPU-h, plus 0.6 CPU-h in every case):

| Sinkhorn variants | GPU-h |
|---|---|
| **Balanced and reach100, both to N=10k (current config)** | **4.0** |
| Balanced and reach100, capped at N=5k | 1.3 |
| No Sinkhorn | 0.55 |

The two N=10k Sinkhorn points take 2.7 of the 4.0 GPU-h. `landscape` is 0.5 GPU-h; its two Sinkhorn methods take almost all of that.

## Cost per estimate (MIG slice / one CPU core)

Scan methods are priced per scan angle and scale linearly with the number of angles. These figures are for `emd_compare`'s ±15° scan, which has 31 angles.

| Method | N=1k | N=5k | N=10k | Notes |
|---|---|---|---|---|
| `emd_sinkhorn` (p=1, debias, scaling 0.9) | ~0.5 s (first CUDA call: 60 s warm-up) | 12.4 s | 48.5 s | ∝ N² (tensorized); GPU 1.7 / 2.3 GB at 5k / 10k |
| `emd_sinkhorn` + `reach` | ~0.5 s | 12.2 s | 49 s | same as balanced |
| `emd_sinkhorn` debias: false | — | 6.0 s | — | about half (no self terms) |
| `emd_sinkhorn` p=2 | 0.45 s | 12.6 s | — | |
| `mmd` (energy / gaussian / laplacian) | <0.1 s | 0.2–1.1 s | 0.8 s | single pass |
| `emd_sliced_kr` (λ = 30 or 100 mm) | 0.14 s | 0.14 s | 0.15 s | launch-bound, flat in N |
| `emd_exact` (CPU) | 2.2 s | — | — | 0.5 s at N=500; ∝ N^2.1 here |
| `emd_exact` kr (CPU) | 2.5–3.0 s | — | — | |
| `emd_exact` partial (CPU) | 7.4 s | — | — | slowest exact variant |
| `fnd` (CPU), 360 angles | 0.3 s | — | 0.38 s | 3.6 s at 100k (Δx=5); 5.0 s at 100k (Δx=1, G=450) |
| `emd_paired`, `emd_sliced`, `centroid` (CPU) | ≤0.1 s | | | 0.06 s for `emd_paired` at 100k |

Fixed costs:

- **Event pool:** loading takes 13 s and 0.5 GB. Forked workers share it, so it is counted once.
- **GPU workers:** the CUDA context adds about 1 GB of host RSS per worker.
- **`smearing` workers:** each smeared copy of the pool adds 0.43 GB per worker.

Measured peak memory (PSS of the whole process tree):

| Run | Peak PSS |
|---|---|
| 1 GPU worker | 1.66 GB |
| `smearing`, 3 workers | 4.33 GB (about 0.75 GB main + 1.2 GB per worker) |
| `capacity_exposure` at N≈1M, 2 workers | 1.46 GB |

## Re-measuring

After a run, `seff <jobid>` gives the real CPU efficiency and peak memory. To size a new method, run it on a reduced grid under `slurm/memwatch.sh python run.py ... --K 2`. The script prints the peak PSS summed over the process tree.

The engine starts the largest-N grid points first when `--jobs > 1`, so the slowest tasks don't form a tail. Seeds are per task, so the results don't depend on task order.
