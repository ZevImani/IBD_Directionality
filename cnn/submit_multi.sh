#!/bin/bash
# Train several CNN setups at once on ONE GPU, to use a full A100 for small-image runs
# (pitch >= 25 mm: ~5 GB GPU each and GPU idle much of the time, see submit_train.sh).
#
# Run it directly from a login shell (NOT with sbatch). Each quoted string is one cnn/train.py
# argument set; the script sizes the request to the number of runs and submits itself:
#
#   cnn/submit_multi.sh "--pitch 50" "--pitch 150" "--pitch 50 --representation discrete"
#   cnn/submit_multi.sh -t 0-03:00 "--pitch 50 --lr 3e-4 --name p50_lr3e-4" "--pitch 50 --lr 3e-3 --name p50_lr3e-3"
#   cnn/submit_multi.sh --dry-run "--pitch 50" "--pitch 25"    # check + print the sbatch command only
#   cnn/submit_multi.sh --here "--pitch 50" "--pitch 150"      # run inside the current allocation
#                                                              # (e.g. in `srun --jobid=<id> --overlap --pty bash`)
# Request for n runs: 1 GPU (iaifi_gpu), n+1 cores (each run keeps one core busy), 3n+3 GB RAM
# (2.4 GB peak per run), -t 0-02:00 unless given (packed runs slow each other down; unmeasured).
#
# Outputs (cnn/logs/):
#   cnn_multi_<job>.out             this script: what was launched, final exit code + time per run
#   multi_<job>_<i>_<name>.out      each run's own stdout/stderr
#   multi_<job>_gpu.csv             nvidia-smi every 30 s (utilisation, memory): shows whether packing paid off
# Checkpoints go to cnn/checkpoints/<name>/ as usual. Runs that would write to the same <name>
# are refused: give them distinct --name. SIGTERM (time limit) is forwarded to every run, which
# finishes its epoch and saves last.pt; resume each with "--resume cnn/checkpoints/<name>/last.pt".
#
# Email when the job ends (END,FAIL), to the address on your cluster account, or to
# MAIL_USER=<address> cnn/submit_multi.sh ...   The job is FAILED if any run exited non-zero
# (75 = stopped at the time limit after saving last.pt; the end of this script's log lists each run).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO/cnn/submit_multi.sh"
CMD="${CNN_TRAIN_CMD:-python -u -m cnn.train}"   # override only for testing

mode=submit; time_limit=0-02:00
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t) time_limit="$2"; shift 2 ;;
        --dry-run) mode=dry; shift ;;
        --here) mode=here; shift ;;
        --inner) mode=inner; shift ;;
        -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        -*) [[ "$1" == *" "* || "$1" == --pitch* || "$1" == --resume* ]] && break
            echo "unknown option $1 (argument sets must be quoted strings)" >&2; exit 2 ;;
        *) break ;;
    esac
done
SETS=("$@")
n=${#SETS[@]}
(( n >= 1 )) || { echo "usage: cnn/submit_multi.sh [-t D-HH:MM] [--dry-run|--here] \"<train args>\" ..." >&2; exit 2; }
if [[ $mode == submit && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Inside SLURM job $SLURM_JOB_ID: use --here to run in this allocation, or run from a login shell to submit." >&2
    exit 2
fi

# value of option $2 in argument string $1 (or default $3)
opt() {
    local -a a; read -ra a <<< "$1"; local i
    for ((i = 0; i < ${#a[@]}; i++)); do
        case "${a[i]}" in "$2") echo "${a[i+1]:-}"; return ;; "$2"=*) echo "${a[i]#*=}"; return ;; esac
    done
    echo "$3"
}

# --- check the argument sets and derive each run's checkpoint name (as cnn/train.py does)
NAMES=(); err=0
for i in "${!SETS[@]}"; do
    s="${SETS[i]}"
    resume=$(opt "$s" --resume ""); pitch=$(opt "$s" --pitch "")
    if [[ -n "$resume" ]]; then
        name=$(basename "$(dirname "$resume")")
    elif [[ -n "$pitch" ]]; then
        arch=$(opt "$s" --arch resnet50); stem=$(opt "$s" --stem small)
        name=$(opt "$s" --name "${arch}_p$(printf '%g' "$pitch")_$(opt "$s" --representation continuous)")
        if awk "BEGIN{exit !($pitch < 25)}"; then
            if [[ $arch == resnet50 && $stem == small ]]; then
                echo "WARNING run $i ($name): pitch $pitch with resnet50/small is ~60 GB GPU at 5 mm and" \
                     "compute-bound; it will likely run out of memory packed with others. Submit it alone." >&2
            else
                echo "note: run $i ($name): pitch $pitch makes large images; these keep the GPU busy," \
                     "so packing gains little." >&2
            fi
        fi
    else
        echo "ERROR run $i: '$s' has neither --pitch nor --resume" >&2; err=1; continue
    fi
    for j in "${!NAMES[@]}"; do
        [[ "${NAMES[j]}" == "$name" ]] && { echo "ERROR runs $j and $i both write cnn/checkpoints/$name: add distinct --name" >&2; err=1; }
    done
    NAMES+=("$name")
done
(( err == 0 )) || exit 2
(( n > 6 )) && echo "note: $n runs share one GPU; above ~4 the per-run slowdown is unmeasured." >&2

# --- submit: size the request and re-run this script inside the job
if [[ $mode == submit || $mode == dry ]]; then
    sb=(sbatch -J cnn_multi -p iaifi_gpu --gres=gpu:1 -c $((n + 1)) --mem=$((3 * n + 3))G -t "$time_limit"
        -o "cnn/logs/%x_%j.out" -e "cnn/logs/%x_%j.out" --signal=B:SIGTERM@900
        --mail-type=END,FAIL ${MAIL_USER:+--mail-user="$MAIL_USER"} "$SELF" --inner "${SETS[@]}")
    for i in "${!SETS[@]}"; do echo "run $i: ${NAMES[i]}  <-  ${SETS[i]}"; done
    printf '%q ' "${sb[@]}"; echo
    [[ $mode == dry ]] && exit 0
    cd "$REPO" && mkdir -p cnn/logs && exec "${sb[@]}"
fi

# --- inner / here: launch every run in the background on the same GPU, then wait for all
cd "$REPO" && mkdir -p cnn/logs
tag="${SLURM_JOB_ID:-local$$}"
if [[ $mode == here && -n "${SLURM_CPUS_ON_NODE:-}" ]] && (( SLURM_CPUS_ON_NODE < n )); then
    echo "WARNING: $n runs on $SLURM_CPUS_ON_NODE cores; each run needs one, so all will be slowed." >&2
fi
if [[ "$CMD" == python* ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ldm
fi
echo "Start: $(date)  host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)  runs: $n"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
    > "cnn/logs/multi_${tag}_gpu.csv" 2>/dev/null &
smi=$!

start=$(date +%s); pids=(); logs=()
set -f   # no globbing when word-splitting the argument strings
for i in "${!SETS[@]}"; do
    logs[i]="cnn/logs/multi_${tag}_${i}_${NAMES[i]}.out"
    $CMD ${SETS[i]} > "${logs[i]}" 2>&1 &
    pids[i]=$!
    echo "run $i: pid ${pids[i]}  ${NAMES[i]}  <-  ${SETS[i]}  -> ${logs[i]}"
done
set +f
trap 'echo "[$(date +%T)] SIGTERM: forwarding to all runs (each finishes its epoch and saves last.pt)"; kill -TERM "${pids[@]}" 2>/dev/null' TERM
trap 'kill -INT "${pids[@]}" 2>/dev/null' INT

rcs=()
for i in "${!pids[@]}"; do
    while :; do
        wait "${pids[i]}"; rc=$?
        (( rc > 128 )) && kill -0 "${pids[i]}" 2>/dev/null && continue   # wait interrupted by a signal
        break
    done
    rcs[i]=$rc
done
kill "$smi" 2>/dev/null

echo "End: $(date)  total: $((($(date +%s) - start) / 60)) min"
fail=0
for i in "${!pids[@]}"; do
    mins=$((($(stat -c %Y "${logs[i]}") - start) / 60))
    echo "run $i: exit ${rcs[i]}  ~${mins} min  ${NAMES[i]}  (${logs[i]})"
    (( rcs[i] != 0 )) && fail=1
done
exit $fail
