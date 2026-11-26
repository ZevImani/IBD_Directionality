# emd_paper: instructions for Claude

IBD directional reconstruction (Paper C): how well estimators (centroid, FND, EMD/OT, CNN) recover the neutrino direction from prompt→capture displacements, versus pitch Δx, N, angle and readout. There's also a diffusion generator (`ldm/`) for IBD events. Read `README.md` for the pipeline and `ROADMAP.md` for goals and status.

Cluster, Python and Slurm rules are in `~/.claude/CLAUDE.md` and apply here: the login node only probes, ask for an interactive job for anything heavy, and never `sbatch` or `scancel`.

## Run
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ldm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python run.py configs/<study>.yaml --jobs N      # -> results/studies/<study>/
python plot.py <study>                           # re-plot from logs/ (needs trials.csv for some figures)
```

## How the code is organised (keep it this way)
- **Two entry points:** `run.py` and `plot.py`. Don't add top-level scripts.
- **A new study is a config** in `configs/<study>.yaml`, plus a row in the README Studies table. It is not a new script.
- **A new estimator is a registered method** in `dirreco/methods/`, documented in the README methods table.
- **A new figure is an `@figure` function** in `dirreco/plots.py`.
- `cnn/` and `ldm/` are sub-projects with their own READMEs and `submit_*.sh` scripts. Pipeline-wide Slurm scripts live in `slurm/`, with sizing notes in `slurm/README.md`.
- `archive/` is frozen pre-refactor code; don't edit it. `ldm/ldm/` is vendored upstream LDM code, so keep changes there minimal and documented in `ldm/README.md`.

## Checks (all need a compute node)
| Check | Command | Covers |
|---|---|---|
| smoke | `python run.py configs/smoke.yaml --jobs 4` (~30 s) | every method runs; sign convention |
| EMD | `python -m dirreco.selftest_emd` → save the output to `results/selftest_emd.log` | sliced/large-template/discrete EMD equivalences, injection recovery, timings |
| CRLB | `python -m dirreco.fisher` (Gaussian self-test included) | bounds in `results/crlb/` |
| LDM sampler | `cd ldm && python test_sampler.py` (~10 s, CPU) | DDIM step ratios, exact-denoiser bias |

A change that alters numbers needs an equivalence check or a rerun of every study it affects. State which in the commit message.

## Results and git
- What's tracked: code, configs, Slurm scripts, docs, and per study `plots/*.png` plus `logs/{config.yaml,summary.csv,run.log}`. Not tracked: `trials.csv`, the dataset, weights, samples, Slurm `.out` files, and `overleaf/`.
- Don't delete or move `results/` folders without asking. Superseded runs, e.g. `ldm_closure_gen_e137` from before the sampler fix, are kept on purpose as records.
- Paper numbers need **K ≥ 200 and full-circle `zoom` scans**. Older studies at K = 20–25 or with ±15° windows are exploratory.
- One commit per logical change. Push only when the user says so.

## Docs style
- READMEs hold the findings, with dates and job IDs (e.g. "2026-11-26, job 63795499"). When code changes a documented number or default, update that README in the same commit.
- PNG names: no `plot_` prefix in this repo.
