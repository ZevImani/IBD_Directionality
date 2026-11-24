#!/bin/bash
# Re-run every CPU study and make its figures.   Submit from the repo root:
#   sbatch slurm/reproduce_cpu.sh                                  # all CPU studies, expect ~16 min
#   STUDIES="scaling smearing" sbatch slurm/reproduce_cpu.sh
#
# Requests = 1.2 x the modelled expectation (see slurm/README.md for the model):
#   Time: 15.7 min expected (angle_sweep 9.5, fnd_refine 2.3, the other four ~1 each, incl. ~45 s
#     per study for imports, event loading and plots), simulated with the engine's task order on
#     16 workers. Total compute ~2.6 CPU-h, timed on one core of a gpu_test node; sapphire cores
#     are assumed no slower.
#   Memory: peak is smearing, which smears a private 0.43 GB copy of the pool per worker:
#     0.75 GB main (0.5 GB shared pool) + 12 tasks x ~1.2 GB = 15.4 GB expected -> 19 GB.
#     Every other study stays under ~0.75 + 16 x 0.4 = 7 GB.
#SBATCH -J emd_cpu
#SBATCH -p sapphire
#SBATCH -c 16
#SBATCH --mem=19G
#SBATCH -t 00:19:00
#SBATCH -o results/slurm_logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
# Mail goes to the address on your cluster account; add --mail-user=<address> on the sbatch line to override.

set -eo pipefail          # no -u yet: ~/.bashrc and conda's activate scripts use unset variables
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
set -u
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
JOBS=${JOBS:-$SLURM_CPUS_PER_TASK}
STUDIES=${STUDIES:-"capacity_exposure scaling fnd_refine smearing phase_space angle_sweep"}
for s in $STUDIES; do
    echo "=== $s  $(date)"
    python run.py configs/$s.yaml --jobs "$JOBS"
done
echo "=== done $(date)   (check usage afterwards with: seff $SLURM_JOB_ID)"
