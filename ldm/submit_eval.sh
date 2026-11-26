#!/bin/bash
# Full evaluation of a trained generator: sample -> evaluate.py -> closure test (real vs generated).
# Run with bash (NOT sbatch) from anywhere; it submits two chained jobs:
#   bash ldm/submit_eval.sh                                                  # defaults below
#   bash ldm/submit_eval.sh logs/<run>/checkpoints/epoch=000113.ckpt e113    # another checkpoint
#   N=2000000 SAMPLE_ARGS="--ddim-steps 0" bash ldm/submit_eval.sh <ckpt> e137_ddpm   # ancestral sampler
#
#   job 1  ldm_eval_gen      iaifi_gpu  sample.py (N events, DDIM-1000) + evaluate.py
#                                       -> ldm/samples/<name>/{vertices,captures}.npy, eval/report.txt
#   job 2  ldm_eval_closure  sapphire   (afterok job 1) run.py configs/ldm_closure.yaml on real [0, 2M)
#                                       and on ldm/samples/<name>, then ldm/closure.py
#                                       -> results/studies/ldm_closure{,_gen_<name>}/, closure.png
#
# Requests = ~1.25x the cost MEASURED on the first run (2026-11-24, jobs 63312878 / 63312881, 2M events):
#   Job 1: 32.4 min, MaxRSS 0.8 GB on an A100-SXM4-80GB (iaifi_gpu): sampling 1756 s = 0.88 ms/event at
#     DDIM-1000 (2.5x the MIG-slice rate), evaluate.py ~2 min -> 1 GPU, 2 cores, 2G, 0:42.
#     For N=9M pass GPU_TIME=02:50:00 (9M x 0.88 ms = 132 min + evaluate, x1.25; memory not measured at 9M).
#   Job 2: 25.0 min, MaxRSS 2.5 GB. Each run is bounded by its slowest N=10k grid point (K=1000 trials,
#     1417-1456 s on sapphire); real and generated runs go side by side with 9 workers each (18 cores)
#     -> 0:32, 3G.
#   The real run is deterministic (seed 11) and is skipped if results/studies/ldm_closure already has
#   a summary; set REDO_REAL=1 to rerun it.
#
# Read the results: ldm/samples/<name>/eval/report.txt (gen column vs the B real-vs-real baseline;
# also the extreme-outlier line) and the closure.py printout at the end of the job 2 log (sigma
# ratios consistent with 1, |pull| < 2-3, biases consistent with 0).

LDM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$LDM_DIR")

# ------------------------------------------------------------------ driver (login node)
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    set -eo pipefail
    # Default: epoch 137, the best val/loss_simple_ema of run 2026-11-17 with clean val/gen_* metrics.
    # last.ckpt (epoch 149) had a theta0 ratio of 1.109 and dstd_y = +3.5 mm at its last validation.
    CKPT=${1:-logs/2026-11-17T00-25-22_ibd-ddpm/checkpoints/epoch=000137.ckpt}
    NAME=${2:-e137}
    N=${N:-2000000}
    [[ "$CKPT" = /* ]] || CKPT="$LDM_DIR/$CKPT"
    [[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
    mkdir -p "$LDM_DIR/logs" "$ROOT/results/slurm_logs"
    export STAGE_CKPT="$CKPT" STAGE_NAME="$NAME" STAGE_N="$N" SAMPLE_ARGS="${SAMPLE_ARGS:-}" REDO_REAL="${REDO_REAL:-0}"

    # Email: job 1 only on failure (job 2 is then cancelled), job 2 on end or failure -> one mail per outcome.
    # Mail goes to the address on your cluster account; MAIL_USER=<address> overrides it.
    MAIL=(${MAIL_USER:+--mail-user=$MAIL_USER})
    j1=$(sbatch --parsable "${MAIL[@]}" --mail-type=FAIL -J ldm_eval_gen -p iaifi_gpu --gres=gpu:1 -c 2 --mem=2G -t "${GPU_TIME:-00:42:00}" \
        --chdir="$LDM_DIR" -o "$LDM_DIR/logs/slurm-%x-%j.out" --export=ALL,STAGE=gen "$LDM_DIR/submit_eval.sh")
    j2=$(sbatch --parsable "${MAIL[@]}" --mail-type=END,FAIL -J ldm_eval_closure -p sapphire -c 18 --mem=3G -t 00:32:00 \
        --dependency=afterok:"$j1" --kill-on-invalid-dep=yes \
        --chdir="$ROOT" -o "$ROOT/results/slurm_logs/%x-%j.out" --export=ALL,STAGE=closure "$LDM_DIR/submit_eval.sh")
    echo "checkpoint: $CKPT"
    echo "samples:    $LDM_DIR/samples/$NAME  ($N events)"
    echo "job $j1  sample + evaluate.py  -> $LDM_DIR/logs/slurm-ldm_eval_gen-$j1.out"
    echo "job $j2  closure (after $j1)   -> $ROOT/results/slurm_logs/ldm_eval_closure-$j2.out"
    exit 0
fi

# ------------------------------------------------------------------ job body
set -eo pipefail          # no -u yet: ~/.bashrc and conda's activate scripts use unset variables
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
set -u
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
echo "=== $STAGE  $(date)  on $(hostname)  ckpt=$STAGE_CKPT  name=$STAGE_NAME"

case "$STAGE" in
gen)
    nvidia-smi --query-gpu=name,memory.total --format=csv
    python -u test_sampler.py          # ~10 s, CPU; stops the chain if the sampler regressed
    # shellcheck disable=SC2086  # SAMPLE_ARGS is a list of extra flags
    python -u sample.py -r "$STAGE_CKPT" -n "$STAGE_N" -o "samples/$STAGE_NAME" ${SAMPLE_ARGS}
    python -u evaluate.py "samples/$STAGE_NAME"
    ;;
closure)
    export OMP_NUM_THREADS=1
    if [[ "$REDO_REAL" == 1 || ! -f results/studies/ldm_closure/logs/summary.csv ]]; then
        python -u run.py configs/ldm_closure.yaml --jobs 9 > results/slurm_logs/ldm_closure_real-$SLURM_JOB_ID.log 2>&1 &
        real_pid=$!
    else
        echo "real closure run exists (results/studies/ldm_closure); skipping. REDO_REAL=1 to rerun."
        real_pid=
    fi
    python -u run.py configs/ldm_closure.yaml --data "ldm/samples/$STAGE_NAME" --tag "gen_$STAGE_NAME" --jobs 9
    if [[ -n "$real_pid" ]]; then
        wait "$real_pid" || { echo "real closure run failed:"; tail -30 results/slurm_logs/ldm_closure_real-$SLURM_JOB_ID.log; exit 1; }
        tail -3 results/slurm_logs/ldm_closure_real-$SLURM_JOB_ID.log
    fi
    echo
    python -u ldm/closure.py ldm_closure "ldm_closure_gen_$STAGE_NAME"
    echo
    echo "evaluate.py report: ldm/samples/$STAGE_NAME/eval/report.txt"
    ;;
*)
    echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
echo "=== done $(date)   (check usage afterwards with: seff $SLURM_JOB_ID)"
