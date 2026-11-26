#!/bin/bash
# Generate events with a trained model, then run the quality tests on them. Submit from ldm/:
#   sbatch submit_sample.sh logs/<run>/checkpoints/last.ckpt <name> [n_events] [extra sample.py args]
#   sbatch submit_sample.sh logs/<run>/checkpoints/last.ckpt closure_v1 2000000
#   sbatch submit_sample.sh logs/<run>/checkpoints/last.ckpt full_v1 9000000 --ddim-steps 0   # ancestral DDPM
# Writes samples/<name>/{vertices,captures}.npy + meta.json, and samples/<name>/eval/ (evaluate.py).
#
# Requests = ~1.2x the cost MEASURED on 2026-11-16 on an A100 MIG 3g.20gb slice:
#   Sampling is compute-bound (5.6 TFLOP/s fp32) and linear in steps and events:
#     DDIM-250: 560 us/event (262k events in 147 s), GPU peak 4.1 GB at the default batch of 262,144.
#     DDIM-1000 (the default): ~2.2 ms/event -> 2M events ~75 min, 9M events ~5.6 h.
#   evaluate.py: ~1 min, 1.6 GB (measured on CPU).
#   Default request fits 2M events at 1000 steps: 1.2 x 77 min -> -t 0-01:35. For 9M events pass
#   -t 0-06:45 on the sbatch line. Host RAM for sampling not measured (estimate ~3.5 GB for 9M) -> 5G.
#   sample.py writes its measured time to samples/<name>/meta.json ("seconds").
#SBATCH -J ldm_sample
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=5G
#SBATCH -t 0-01:35
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
# Mail goes to the address on your cluster account; add --mail-user=<address> on the sbatch line to override.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

CKPT=${1:?checkpoint}; NAME=${2:?name}; N=${3:-2000000}
shift $(( $# < 3 ? $# : 3 ))
python -u sample.py -r "$CKPT" -n "$N" -o "samples/$NAME" "$@"
python -u evaluate.py "samples/$NAME"
