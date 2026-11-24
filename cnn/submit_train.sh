#!/bin/bash
# Train the direction CNN on a GPU node. Submit from the repo root; arguments go to cnn/train.py.
# The #SBATCH defaults below fit pitch >= 25 mm with the default ResNet50 (small stem, batch 256,
# 60 x 500 steps). Larger images need more time: override on the sbatch line.
#
#   sbatch cnn/submit_train.sh --pitch 50                         # ~30 min
#   sbatch cnn/submit_train.sh --pitch 150                        # ~15 min
#   sbatch -t 0-08:00 cnn/submit_train.sh --pitch 5               # ResNet50 81x81: ~5 h, ~60 GB GPU (est.)
#   sbatch -t 0-02:00 cnn/submit_train.sh --pitch 5 --arch resnet18   # ~1-3 h, 11 GB GPU
#   sbatch cnn/submit_train.sh --resume cnn/checkpoints/resnet50_p50_continuous/last.pt
#
# Measured with cnn/benchmark.py on an A100 MIG 3g.20gb slice (42 of 108 SMs) on 2026-11-16,
# log in cnn/logs/benchmark_mig3g.log. Training step = data generation (on GPU, ~2 ms) + model:
#   pitch  image  arch/stem/batch          s/step  GPU peak   per-epoch validation
#   150    5x5    resnet50/small/256       0.024    4.0 GB    1.5 s
#   50     9x9    resnet50/small/256       0.039    4.7 GB    1.1 s   (full epoch incl. saves: 27 s)
#   5      81x81  resnet50/small/256        OOM    > 20 GB    (64: 0.35 s, 15.8 GB -> ~60 GB at 256)
#   5      81x81  resnet18/small/256       0.33    10.5 GB    5.2 s
#   5      81x81  resnet50/imagenet/256    0.12     7.3 GB    2.4 s
# Times on a full A100 (iaifi_gpu) should be <= these; up to ~2.5x faster for the 81x81 images.
#   Host RAM: 2.4 GB peak (9M-event pool 0.8 GB + per-pitch split + CUDA context) -> --mem=6G.
#   CPU: one core 100% busy (the Python thread launching kernels; data are made on the GPU),
#        so more cores do not help; -c 2 leaves one for I/O and driver threads.
#   Disk: best.pt + last.pt, ~300 MB each (weights + AdamW state).
# After a job, `seff <jobid>` shows what it actually used.
#SBATCH -J cnn_train
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 0-01:00
#SBATCH -o cnn/logs/%x_%j.out
#SBATCH -e cnn/logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@900  # 15 min before the limit: finish the epoch (<= ~12 min at 5 mm), save last.pt
#SBATCH --mail-type=END,FAIL    # email when the job ends; to the address on your cluster account
                                #   unless you add  --mail-user=<address>  (here or on the sbatch line)
# Exit code (shown in the email and in sacct): 0 = training finished, 75 = stopped at the time
# limit after saving last.pt (resume it), anything else = crashed (see cnn/logs/<job>.err).

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Start: $(date)  host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)  args: $*"
start=$(date +%s)

# run python in the background and forward SIGTERM to it (train.py traps it)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ldm
python -u -m cnn.train "$@" &
pid=$!
trap 'kill -TERM $pid' TERM
while :; do
    wait $pid; rc=$?
    (( rc > 128 )) && kill -0 $pid 2>/dev/null && continue   # wait interrupted by SIGTERM; keep waiting
    break
done

echo "End: $(date)  elapsed: $((($(date +%s) - start) / 60)) min  exit code: $rc"
exit $rc   # so SLURM (and the email) report FAILED when training did not finish
