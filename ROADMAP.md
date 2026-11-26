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
- [ ] **Headline study `configs/paper_main.yaml`**: one run of the final method shortlist on the same samples, so every comparison is paired. Methods: centroid (cont/disc), `fnd_parab`, `emd_sliced` (cont/disc) plus `emd_exact` at N ≤ 1000, and the CNN wherever a checkpoint exists. K ≥ 200, full-circle `zoom`, θ_true ∈ {0, 30}, same grid as the pitch figure below. Where it can, this replaces the separate reruns above. In `emd_compare` (K = 25), sliced ≈ exact ≈ Sinkhorn ≈ energy MMD, the unbalanced variants gain nothing, and `emd_paired` is 2–4× worse. So the rest of the EMD family goes into one appendix table.

### Pillar 2
- [ ] Every figure uses per-pitch bounds: continuous 146.6 (≤5 mm) / 128.8 (50) / 95.1 (150); cell-hop 138.5 / 126.0 / 116.8 (50 / 100 / 150). The draft still uses a single Θ0 = 129.58.
- [ ] `phase_space` was run at K = 5; rerun it.
- [ ] **Headline pitch figure** (from `paper_main`): efficiency η = (CRLB/Θ)² vs Δx at fixed N (e.g. 10³ and 10⁴) for every shortlisted method, in both readouts. Pitch grid 1, 5, 10, 25, 50, 100, 150 mm. Add a companion plot against exposure n_total (Θ_exp), since the containment cut is what makes 50 mm FND's optimum.

### Pillar 3
- [ ] Rerun `angle_sweep` with the new CCW convention to check FND's 1–2° bias (README open issue 4).
- [ ] **Headline angle study `configs/paper_angle.yaml`**: θ_true from 0 to 45° in 5° steps (the lattice symmetry covers the other angles), `same_samples: true`, pitches 150/50/5 mm, every shortlisted method, K ≥ 500. Report σ(θ) and bias(θ) relative to `centroid_cont`, as `cnn_angle` does. Regenerate the 360° linearity figure with the final methods; it doesn't need K ≥ 200.
- [ ] CRLB vs angle at 100 and 150 mm (`python -m dirreco.fisher --pitches 100,150 --theta 0,5,...,45`). Only 50 mm is done (`results/crlb/crlb_angle.csv`).
- [x] `cnn_angle` at 50 mm, 0–45° (2026-11-25): the CNN's Θ varies ±3% irregularly, with a bias of up to +0.15° that doesn't shrink with N. The cell-hop CRLB is flat with angle.

### Pillar 4
- [x] DDIM t=999 outlier fix plus `test_sampler.py` (commit `2c1e5fc`).
- [ ] **Review closures** `ldm_closure_gen_e137_fix` and `_fix_cut` against `ldm_closure`. Does σ_gen/σ_real pass at the N the paper will use?
- [ ] Rewrite the draft's LDM section. It describes θ/pitch conditioning, but the model is an unconditional EventMLP on r. The new text should cover:
  - the architecture: 6 residual blocks of width 512, FiLM time conditioning, about 6M parameters;
  - the noise schedule (cosine, T = 1000) and the scaling x = r/50 mm;
  - the y-flip symmetry, and the vertex being uniform and independent of r;
  - the t = 999 sampler finding;
  - a table of `evaluate.py` metrics next to the real-vs-real baseline;
  - the closure figure.
- [ ] Pick which generator applications go in the paper (memory note `ldm-application-ideas`; also `ldm/README.md`).
- [ ] **Remaining generated-vs-real mismatches** (`ldm/samples/e137_fix_cut/eval/report.txt`):
  - cell-hop χ²/ndf is 1.51 at 150 mm, against 0.80 between the two real halves;
  - it is 1.27 at 5 mm, against 0.90;
  - φ_xy KS gives p = 0.028.

  Explain or fix these before the paper claims the generator reproduces the cell readout.
- [ ] **CRLB computed from the generated pool vs the real pool**, per pitch, continuous and cell-hop. This makes a clean "keeps the expected properties" figure. It needs a `--data DIR` option in `dirreco/fisher.py`.
- [ ] **Extend the closure** (`configs/ldm_closure.yaml`):
  - add `emd_sliced` with `discrete: true`;
  - add the CNN, trained on real events and evaluated on generated ones;
  - go up to the largest N that any chosen application uses (closure currently stops at N = 10⁴).

### Paper draft sync
- [ ] Regenerate the draft's figures from `plot.py`. The current ones are Oct-2026 PDFs from the pre-refactor pipeline.
- [ ] Abstract: it claims Δx ∈ [1 µm, 200 mm], but the studies cover 0.5–150 mm. It also opens with two leftover questions ("Can directionality improve machine vision reliably? …").
- [ ] The "Algorithmic Capacity vs. Operational Exposure" section has the same paragraph twice (draft lines 521 and 524).
- [ ] The Conclusions repeat the "W1 saturates the CRLB" claim. Update them together with the abstract once `emd_crlb` has been reviewed.

## Decisions needed (user)
- **CNN in the paper:** the draft has a 62k-parameter CoordConv MTCNN at 3 pitches; the repo has a ResNet50 at 50 mm only. Train more pitches (`cnn_capacity` needs a 5 mm model), port the CoordConv, or drop the CNN to a short section?
- **Which EMD variant(s)** headline the paper once `emd_crlb` is reviewed? Proposed shortlist (2026-11-26): `emd_sliced` in both readouts, with `emd_exact` as a small-N check. This feeds `paper_main` and `paper_angle`.
- **LDM sampler for the paper samples:**
  - Option A: regenerate 2M events with the ancestral DDPM sampler. It gave 0 outliers in the 150k-event A/B test and costs about 30 GPU-minutes.
  - Option B: keep DDIM-1000 and cut events beyond |r| = 412 mm (13 of 2M removed).

  Proposed: A, so the paper has no after-the-fact cut to justify.

## Verification owed
- [ ] `python -m dirreco.selftest_emd | tee results/selftest_emd.log` on a compute node. The committed log (2026-11-25) predates check 0, which validates the `emd_sliced` sort-sharing speedup (commit `32ad77f`).

## Housekeeping
- [ ] First `/manage audit` pass over `dirreco/`, `cnn/`, `ldm/` and the pipeline files.
- [ ] `slurm/emd_crlb.sh` is oversized now that `emd_sliced` is faster: the run took 40 min against a 1:00 request. Resize it from `sacct` before the next run.
