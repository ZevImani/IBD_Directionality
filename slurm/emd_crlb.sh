#!/bin/bash
# emd_crlb: EMD vs centroid vs CRLB, continuous and cell-hop readout (configs/emd_crlb.yaml).
# CPU only (torch runs the large-template sliced W1 on the CPU workers). Submit from the repo root:
#   sbatch slurm/emd_crlb.sh
#   python -m dirreco.selftest_emd     # optional first, on an interactive node: checks + timings
#
# Requests = ~1.25 x the modelled expectation (timings: gpu_test node, one core, 2026-11-25):
#   Compute: ~11 CPU-h for 50 tasks (5 pitches x 5 N x 2 angles) at K = 200. Per-estimate cost
#     at N = 10k: emd_sliced_t20 6.3 s (torch CPU), emd_sliced 1.7 s, *_disc_t20 0.7 s each;
#     at N = 1k: emd_exact 6.6 s (continuous POT, max_N 1000) -- ~4 CPU-h of the total.
#     Check: a K = 2 run of the full grid took 400 s on one core -> 11.1 CPU-h at K = 200.
#   Wall: 33 min expected on 24 workers (engine task order simulated from the K = 2 run; the
#     longest single task, N = 10k at 50 mm, is ~32 min, so more cores don't help) + ~2 min
#     load/plots -> 35 min -> 45 min on sapphire; 1:00 requested so the job also fits on
#     shared's older Cascade Lake cores (-p sapphire,shared: whichever starts first; sapphire
#     queued too long on 2026-11-25). 16 workers would take ~52 min.
#   These timings predate the shared-sort emd_sliced (2026-11-26): emd_sliced is now ~3.5x faster
#     at n_proj 32 and emd_sliced_t20 ~1.3x (see README EMD notes), so this request is generous.
#     The K = 200 run on 2026-11-26 (old code, 24 workers) took 40.0 min, N = 10k tasks 33-38 min.
#   Memory: summed PSS measured with 4 workers on the heaviest tasks (N = 10k, ref 200k):
#     4.0 GB = 0.75 GB main + 0.82 GB per worker -> 0.75 + 24 x 0.82 = 20.4 GB -> 25 GB.
#SBATCH -J emd_crlb
#SBATCH -p sapphire,shared
#SBATCH -c 24
#SBATCH --mem=25G
#SBATCH -t 01:00:00
#SBATCH -o results/slurm_logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Mail goes to the address on your cluster account; add --mail-user=<address> on the sbatch line to override.

set -eo pipefail          # no -u yet: ~/.bashrc and conda's activate scripts use unset variables
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
set -u
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDA_VISIBLE_DEVICES=""
JOBS=${JOBS:-$SLURM_CPUS_PER_TASK}
echo "=== emd_crlb  $(date)"
python run.py configs/emd_crlb.yaml --jobs "$JOBS"
echo "=== done $(date)   (check usage afterwards with: seff $SLURM_JOB_ID)"
