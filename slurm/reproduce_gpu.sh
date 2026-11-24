#!/bin/bash
# GPU studies: landscape + emd_compare (geomloss / torch on the GPU, exact OT on CPU workers).
# Submit from the repo root:
#   sbatch slurm/reproduce_gpu.sh                                          # 1 GPU, expect ~1.95 h
#   sbatch --gres=gpu:4 -c 5 --mem=7G -t 00:52:00 --export=ALL,JOBS=4 slurm/reproduce_gpu.sh
#                              # 4 GPUs, one worker per GPU (engine pins worker i -> GPU i), expect ~43 min
#   STUDIES=smoke sbatch -p gpu_test -t 00:30:00 slurm/reproduce_gpu.sh    # quick check
#
# Requests = 1.2 x the modelled expectation (see slurm/README.md for the model):
#   Time: 1.94 h expected = landscape 14 min + emd_compare 102 min, simulated with the engine's task
#     order and 3 workers sharing the GPU. GPU work: emd_compare 4.0 GPU-h + landscape 0.5 GPU-h on an
#     A100 MIG 3g.20gb slice (measured), ~90% of it Sinkhorn (balanced + reach100, run to N=10k), taken
#     as 2.5x faster on a full A100 (NOT measured). At 2.0x the job needs ~2.36 h and would time out.
#   Host memory: 0.75 GB main (0.5 GB shared event pool) + 3 x ~1.1 GB per GPU worker (CUDA context)
#     = 4.1 GB expected -> 5 GB.  GPU memory: <= 2.3 GB allocated / 4.1 GB reserved per worker.
#   CPUs: 3 workers + the (mostly idle) main process.
#SBATCH -J emd_gpu
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=5G
#SBATCH -t 02:20:00
#SBATCH -o results/slurm_logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Mail goes to the address on your cluster account; add --mail-user=<address> on the sbatch line to override.

set -eo pipefail          # no -u yet: ~/.bashrc and conda's activate scripts use unset variables
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
set -u
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
nvidia-smi --query-gpu=name,memory.total --format=csv
JOBS=${JOBS:-3}
STUDIES=${STUDIES:-"landscape emd_compare"}
for s in $STUDIES; do
    echo "=== $s  $(date)"
    python run.py configs/$s.yaml --jobs "$JOBS"
done
echo "=== done $(date)   (check usage afterwards with: seff $SLURM_JOB_ID)"
