#!/bin/bash
# Train the IBD event DDPM (configs/ibd-ddpm.yaml) on one GPU. Submit from ldm/:
#   sbatch submit_train.sh
#   sbatch submit_train.sh model.params.dims=2 data.params.batch_size=8192   # config overrides (dotlist)
#   sbatch submit_train.sh --resume logs/<run>                               # continue a run
# Extra arguments go to main.py. Runs land in logs/<date>_ibd-ddpm/ (checkpoints/, configs/,
# images/ with the real-vs-generated marginals, testtube/ metrics.csv) and in W&B project "logs".
#
# Requests = ~1.2x the cost MEASURED on 2026-11-16 on an A100 MIG 3g.20gb slice (42 of 108 SMs; the
# kind of GPU an iaifi_gpu interactive job got), default config, 2-epoch run:
#   Training epoch: 63 s (1587 steps, 38 ms/step; partly compute-bound at fp32 on the slice).
#   Validation epoch: +62 s (val loss + generating 100k events with DDIM-250). With
#     check_val_every_n_epoch: 2, 150 epochs = 150 x 63 s + 75 x 62 s = 3.9 h -> -t 0-04:45.
#     A full A100 should be faster (not measured).
#   Memory: GPU peak 1.75 GB; host RSS peak 1.71 GB -> --mem=3G.
#   CPU: one Python thread drives the GPU; data are batched in-process (num_workers=0) -> -c 2.
# Cheaper variants (pass on the command line; the time scales accordingly):
#   lightning.trainer.check_val_every_n_epoch=5   -> 150 x 63 + 30 x 62 s = 3.1 h  (sbatch -t 0-03:45)
#   Changing max_epochs: also set model.params.scheduler_config.params.max_decay_steps=<epochs x 1587>.
# Lightning saves last.ckpt at every validation, so a run cut off by the time limit resumes with
#   sbatch submit_train.sh --resume logs/<run>
#SBATCH -J ldm_train
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=3G
#SBATCH -t 0-04:45
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
# Mail goes to the address on your cluster account; add --mail-user=<address> on the sbatch line to override.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source ~/.bashrc >/dev/null 2>&1 || true
conda activate ldm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

if [[ " $* " == *" --resume "* || " $* " == *" -r "* ]]; then BASE=(); else BASE=(--base configs/ibd-ddpm.yaml); fi
python -u main.py "${BASE[@]}" \
	--logdir logs \
	--scale_lr False \
	--log_wandb "${WANDB:-True}" \
	--train \
	--gpus 0, \
	"$@"
