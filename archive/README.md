# archive/: pre-refactor state (frozen 2026-11-16)

Nothing here is used by the new pipeline (`run.py` / `plot.py` / `dirreco/`). It is kept for provenance.

| Path | What | In git |
|---|---|---|
| `README_2026-11-16.md` | The repo README before the refactor: full FoM definitions, value tables, and the issues list | yes |
| `code/` | All of Max's Python scripts as they were, including the 5,400-line `paper_c_stats_package_v0_6_6_3` monolith | yes |
| `docs/` | Max's own READMEs (`.md` current, `.txt` Colab-era) | yes |
| `audit/` | `audit_kinematics_output.txt` (CRLB / cut audit), `w1_audit_output.txt` (EMD audit; Parts 2–4 invalid, unpaired data) | yes |
| `logs/` | stdout of Max's mtc-b runs that produced `results/legacy_tables/` | no (maintainer's copy only) |
| `tarballs/` | As received from mtc-b: `paperC_zev_claude_pkg.tar.gz` (Max's scripts), `fid_arrays.tar.xz` (= `dataset/`) | no (maintainer's copy only) |

The old scripts still run from `archive/code/`. They read their inputs from the current directory, which needs three symlinks. Git doesn't track these symlinks, so a fresh clone has to create them first:

```bash
cd archive/code
ln -s ../../dataset/fid_vertices.npy ../../dataset/fid_captures.npy .
ln -s ../../results/legacy_tables review_deliverables
LD_LIBRARY_PATH=$CONDA_PREFIX/lib MPLBACKEND=Agg python plot_final_paper_figures_v3_0.py
```

Old run scripts write new tables into `results/legacy_tables/`. Old plot scripts write to
`archive/results/plots/`.
