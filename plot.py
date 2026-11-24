#!/usr/bin/env python
"""Make figures from a run's output (or from Max's legacy tables).

    python plot.py scaling                         # results/studies/scaling; all figures in its config
    python plot.py results/studies/scaling --figure scaling
    python plot.py legacy:capacity_exposure        # Max's tables -> results/plots/legacy_capacity_exposure/
    python plot.py capacity_exposure --figure capacity --methods fnd_parab,emd_paired --ref centroid_disc
    python plot.py data --figure hits              # figures that only need the dataset

Legacy keys: capacity_exposure scaling emd_sparsity phase_space dense_kink smearing fnd_refine angle_sweep
Figures: capacity exposure methods scaling smearing phase3d full360 pair landscape hits
"""
import argparse
from pathlib import Path

from dirreco import plots, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="run dir | run name | legacy:<key> | data")
    ap.add_argument("--figure", help="only this figure (default: all in the config's plots: list)")
    ap.add_argument("--methods", help="comma list (capacity/exposure/scaling/smearing/landscape)")
    ap.add_argument("--method", help="single method (phase3d/full360)")
    ap.add_argument("--ref", help="reference method drawn dotted")
    ap.add_argument("--out", help="output directory (default: <run>/plots)")
    a = ap.parse_args()

    if a.source == "data":
        run = results.Run("data", plot_dir=a.out or results.PLOTS_DIR)
    else:
        run = results.load_run(a.source)
        if a.out:
            run.plot_dir = Path(a.out)
    plots.make_plots(run, only=a.figure,
                     methods=a.methods.split(",") if a.methods else None,
                     method=a.method, ref=a.ref)


if __name__ == "__main__":
    main()
