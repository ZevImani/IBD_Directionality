# Paper C roadmap

Living goal tracker, maintained by `/manage` (`.claude/skills/manage/SKILL.md`). Findings live in the READMEs; this file tracks what's done, what's next, and what's blocked.

**Last /manage pass:** none yet (set up 2026-11-26)

## Pillars
1. **Estimators, continuous vs cell readout**: centroid, FND, EMD/OT and CNN compared against the CRLB.
2. **Pitch dependence**: everything versus Δx, with per-pitch bounds.
3. **All angles**: performance is flat in θ_true and in lattice orientation.
4. **LDM generator**: generated IBD events keep the properties the analysis relies on (closure).

## Running / waiting on
- Nothing running as of 2026-11-26 01:00.

## Next up
### Pillar 1
- [ ] **Review `emd_crlb`** (ran 2026-11-26, K = 200, interactive job 63795499). Does EMD with a 20N template (`*_t20`) reach the CRLB, or stay near the √2 two-sample bound? This decides the draft's "W1 saturates the CRLB" claim, which current data (Θ ≈ 170–240 vs CRLB 129–146) contradicts.
- [ ] **FND at fine pitch**: σ floors at about 8° at 5 and 1 mm, and Θ grows with N (README open issue 6). Find the cause (the ±15° 1° grid? parabolic refinement on sparse histograms?) before the ρ-collapse story.
- [ ] **Paper-quality reruns**: `capacity_exposure`, `scaling`, `fnd_refine`, `smearing` and `phase_space` are at K = 5–25, some with ±15° windows. Rerun them at K ≥ 200 with full-circle `zoom` (as `configs/emd_crlb.yaml` does). This needs Slurm sizing.

### Pillar 2
- [ ] Every figure uses per-pitch bounds: continuous 146.6 (≤5 mm) / 128.8 (50) / 95.1 (150); cell-hop 138.5 / 126.0 / 116.8 (50 / 100 / 150). The draft still uses a single Θ0 = 129.58.
- [ ] `phase_space` was run at K = 5; rerun it.

### Pillar 3
- [ ] Rerun `angle_sweep` with the new CCW convention to check FND's 1–2° bias (README open issue 4).
- [x] `cnn_angle` at 50 mm, 0–45° (2026-11-25): the CNN's Θ varies ±3% irregularly, with a bias of up to +0.15° that doesn't shrink with N. The cell-hop CRLB is flat with angle.

### Pillar 4
- [x] DDIM t=999 outlier fix plus `test_sampler.py` (commit `2c1e5fc`).
- [ ] **Review closures** `ldm_closure_gen_e137_fix` and `_fix_cut` against `ldm_closure`. Does σ_gen/σ_real pass at the N the paper will use?
- [ ] Rewrite the draft's LDM section. It describes θ/pitch conditioning, but the model is an unconditional EventMLP on r.
- [ ] Pick which generator applications go in the paper (memory note `ldm-application-ideas`; also `ldm/README.md`).

### Paper draft sync
- [ ] Regenerate the draft's figures from `plot.py`. The current ones are Oct-2026 PDFs from the pre-refactor pipeline.

## Decisions needed (user)
- **CNN in the paper:** the draft has a 62k-parameter CoordConv MTCNN at 3 pitches; the repo has a ResNet50 at 50 mm only. Train more pitches (`cnn_capacity` needs a 5 mm model), port the CoordConv, or drop the CNN to a short section?
- **Which EMD variant(s)** headline the paper once `emd_crlb` is reviewed?

## Verification owed
- [ ] `python -m dirreco.selftest_emd | tee results/selftest_emd.log` on a compute node. The committed log (2026-11-25) predates check 0, which validates the `emd_sliced` sort-sharing speedup (commit `32ad77f`).

## Housekeeping
- [ ] First `/manage audit` pass over `dirreco/`, `cnn/`, `ldm/` and the pipeline files.
- [ ] `slurm/emd_crlb.sh` is oversized now that `emd_sliced` is faster: the run took 40 min against a 1:00 request. Resize it from `sacct` before the next run.
