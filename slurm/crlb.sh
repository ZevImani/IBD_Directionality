#!/bin/bash
# Data-driven Cramer-Rao bound (dirreco/fisher.py): Gaussian self-test, then the real pool.
#   sbatch slurm/crlb.sh          (from the repo root; ~4 min, 1.8 GB measured)
#SBATCH -J crlb
#SBATCH -p sapphire
#SBATCH -c 4
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH -o results/slurm_logs/%x-%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python -u -m dirreco.fisher --selftest
python -u -m dirreco.fisher
