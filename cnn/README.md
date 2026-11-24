# cnn: direction reconstruction with a ResNet

This folder trains a CNN to estimate the neutrino direction θ from a bulk sample of N IBD events.
The goal is to reach a given angular resolution σ_θ with as few events as possible, which means
pushing Θ = σ_θ√N below the centroid's. The model is Zev's modified ResNet50, repurposed from
`Proton64_Reco_Model`; the original proton code is still in `~/inference_loop/Proton64_Reco_Model`.

## What the network sees and predicts

- **Input.** One image is the 2D histogram of the displacements r = capture − vertex of the N events
  (cells of one pitch, centred on r = 0, ±200 mm; 9×9 at 50 mm, 81×81 at 5 mm). Pixels hold
  counts·S²/N. `--representation discrete` histograms the cell-to-cell displacement instead, which is
  what the detector reads out. See `images.py`.
- **Output.** A 2-vector u. The estimate is θ̂ = atan2(u_y, u_x), so there is no wrap-around problem.
- **Loss.** N·(1 − cos(θ̂ − θ)) ≈ N·err²/2, the mean Θ² over the batch, so every N counts equally.
  `--weighting none` drops the N factor, which makes the smallest N dominate.
- **Training data.** Samples are generated on the GPU on the fly. Each image draws N log-uniform in
  [30, 10000] usable events and rotates them by θ ~ U(−180°, 180°). Each epoch is new data.
- **No reference sample.** Like the centroid, the CNN needs none. FND/EMD build their θ = 0
  template from N reference events; the CNN learns it from the training pool. Keep this in mind
  when comparing it to the template methods.

## Data split

| Event indices | Use |
|---|---|
| `[0, 2,000,000)` | Never trained on. Reserved for `run.py` studies: set `data: {max_events: 2000000}` |
| `[2,000,000, 2,500,000)` | Validation (fixed set, same seed every epoch) |
| `[2,500,000, end)` | Training |

## Usage (from the repo root)

```bash
sbatch cnn/submit_train.sh --pitch 50                           # one model per pitch (~30 min)
sbatch -t 0-08:00 cnn/submit_train.sh --pitch 5                 # 81x81 images: ~5 h, ~60 GB GPU
sbatch cnn/submit_train.sh --resume cnn/checkpoints/resnet50_p50_continuous/last.pt

python cnn/plot_loss.py cnn/checkpoints/resnet50_p50_continuous # -> training.png
python run.py configs/cnn_capacity.yaml --jobs 1                # CNN vs centroid / FND
```

Run `python -m cnn.train -h` for all options:

- `--representation`;
- `--arch` (resnet18/34/50/101/152);
- `--stem` (`small` for small images, `imagenet` for the original 7×7/2 + max-pool);
- `--n-min/--n-max`;
- `--theta-range`;
- `--epochs/--steps-per-epoch/--batch-size/--lr`.

Each checkpoint stores its args and image spec, so the `cnn` method in `dirreco/methods/cnn.py` can
rebuild the identical input. Outputs go to `cnn/checkpoints/<arch>_p<pitch>_<representation>/`:
`best.pt` (lowest validation loss), `last.pt`, `log.csv` and `args.json`.

`log.csv` and the per-epoch printout give, for each validation N, the CNN rms error and its ratio to
the centroid computed on the same samples. A ratio < 1 means the CNN beats the centroid.

## Files

| File | What |
|---|---|
| `images.py` | Event sample → image. Shared by training and evaluation |
| `resnet.py` | ResNet18–152; 1 input channel, 2 outputs, `stem` option |
| `train.py` | Training loop, validation, checkpoints |
| `submit_train.sh` | SLURM wrapper (iaifi_gpu; 1 h, 2 cores, 6 GB by default). Its header has the measured costs per pitch. Forwards SIGTERM so the last epoch is saved |
| `submit_multi.sh` | Packs several small-pitch runs onto one GPU. Run it directly, not with sbatch: `cnn/submit_multi.sh "--pitch 50" "--pitch 150"`. It sizes the request and submits itself |
| `benchmark.py` | Measures s/step, GPU and host memory, and CPU use per config, for sizing jobs (`python -m cnn.benchmark` on a GPU node) |
| `plot_loss.py` | Loss and rms/centroid ratio vs epoch |
