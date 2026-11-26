#!/bin/bash
# Outlier effect on the closure test: cut generated events beyond the real pool's max |r|, rerun
# evaluate.py + the closure on the cut sample, and compare with the uncut closure.
# Run with bash (NOT sbatch) from anywhere; it submits one CPU job:
#   bash ldm/submit_cut_closure.sh                 # e137_fix (default)
#   bash ldm/submit_cut_closure.sh e137_ddpm       # any ldm/samples/<name> whose closure already ran
#   CUT_MM=450 bash ldm/submit_cut_closure.sh      # explicit threshold; output name <name>_cut450
#
# Needs: results/studies/ldm_closure (real run) and results/studies/ldm_closure_gen_<name> (uncut
# closure, from submit_eval.sh). The default cut comes from ldm/samples/<name>/eval/metrics.json,
# or cut_outliers.py recomputes it from the eval split if that file lacks the "outliers" block.
# Don't edit this file while a copy runs via bash on an interactive node: bash reads it lazily.
#
# Outputs (both uncut and cut are kept):
#   ldm/samples/<name>_cut/{vertices,captures,meta.json}   cut sample; meta.json "cut" lists what was removed
#   ldm/samples/<name>_cut/eval/report.txt                 evaluate.py on the cut sample
#   results/studies/ldm_closure_gen_<name>/logs/closure.txt       closure.py printout, uncut
#   results/studies/ldm_closure_gen_<name>_cut/logs/closure.txt   closure.py printout, cut
#   results/studies/ldm_closure_gen_<name>_cut/logs/closure_compare.txt   side by side
#   results/studies/ldm_closure_gen_<name>_cut/plots/{closure,closure_compare}.png
#
# Requests = ~1.25x measured (2026-11-25, interactive job 63599773, e137_fix closure with 4 workers
# on sapphire-class cores: 24.7 min, MaxRSS 0.77 GB; evaluate.py ~2 min, 0.8 GB) -> 4 cores, 2G, 0:35
# on sapphire; 0:50 requested so it also fits on shared's older Cascade Lake cores
# (-p sapphire,shared: whichever starts first; sapphire queued too long on 2026-11-25).
# 4 workers suffice: the three N=10k grid points (~24 min each) run in parallel and bound the time.

LDM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$LDM_DIR")

# ------------------------------------------------------------------ driver (login node)
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    set -eo pipefail
    NAME=${1:-e137_fix}
    CUT_MM=${CUT_MM:-}
    OUT=${NAME}_cut${CUT_MM}
    [[ -f "$LDM_DIR/samples/$NAME/vertices.npy" ]] || { echo "no samples: $LDM_DIR/samples/$NAME" >&2; exit 1; }
    [[ -f "$ROOT/results/studies/ldm_closure/logs/summary.csv" ]] || { echo "real closure run missing" >&2; exit 1; }
    [[ -f "$ROOT/results/studies/ldm_closure_gen_$NAME/logs/summary.csv" ]] || {
        echo "uncut closure run missing: results/studies/ldm_closure_gen_$NAME" >&2; exit 1; }
    mkdir -p "$ROOT/results/slurm_logs"
    export STAGE_NAME="$NAME" STAGE_OUT="$OUT" CUT_MM
    MAIL=(${MAIL_USER:+--mail-user=$MAIL_USER})
    j=$(sbatch --parsable "${MAIL[@]}" --mail-type=END,FAIL -J ldm_cut_closure -p sapphire,shared -c 4 --mem=2G -t 00:50:00 \
        --chdir="$ROOT" -o "$ROOT/results/slurm_logs/%x-%j.out" --export=ALL "$LDM_DIR/submit_cut_closure.sh")
    echo "samples: ldm/samples/$NAME -> ldm/samples/$OUT  (cut ${CUT_MM:-= real max from eval/metrics.json})"
    echo "job $j -> $ROOT/results/slurm_logs/ldm_cut_closure-$j.out"
    exit 0
fi

# ------------------------------------------------------------------ job body
set -eo pipefail          # no -u yet: ~/.bashrc and conda's activate scripts use unset variables
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
set -u
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib OMP_NUM_THREADS=1
echo "=== cut closure  $(date)  on $(hostname)  name=$STAGE_NAME  out=$STAGE_OUT  cut=${CUT_MM:-real max}"

UNCUT=results/studies/ldm_closure_gen_$STAGE_NAME
CUT=results/studies/ldm_closure_gen_$STAGE_OUT

python -u ldm/cut_outliers.py "ldm/samples/$STAGE_NAME" -o "ldm/samples/$STAGE_OUT" ${CUT_MM:+--cut "$CUT_MM"}
echo
(cd ldm && python -u evaluate.py "samples/$STAGE_OUT")
echo
python -u run.py configs/ldm_closure.yaml --data "ldm/samples/$STAGE_OUT" --tag "gen_$STAGE_OUT" --jobs "$SLURM_CPUS_PER_TASK"
echo
python -u ldm/closure.py ldm_closure "$UNCUT" | tee "$UNCUT/logs/closure.txt"
echo
python -u ldm/closure.py ldm_closure "$CUT" | tee "$CUT/logs/closure.txt"
echo
python -u ldm/closure_compare.py ldm_closure "$UNCUT" "$CUT" -o "$CUT/plots/closure_compare.png" \
    | tee "$CUT/logs/closure_compare.txt"
echo
echo "evaluate.py reports: ldm/samples/$STAGE_NAME/eval/report.txt  vs  ldm/samples/$STAGE_OUT/eval/report.txt"
echo "=== done $(date)   (check usage afterwards with: seff $SLURM_JOB_ID)"
