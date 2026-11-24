#!/usr/bin/env python
"""Run a study config: data -> reco methods -> results/studies/<name>[_<tag>]/.

    python run.py configs/scaling.yaml
    python run.py configs/scaling.yaml --K 5 --pitch 50,5 --N 1000,10000 --jobs 8 --tag test
    python run.py configs/smoke.yaml --methods centroid_disc,emd_sliced
    python run.py configs/ldm_closure.yaml --data ldm/samples/<name> --tag gen   (generated events)

Writes plots/*.png (the config's `plots:` list) and logs/: config.yaml (resolved),
trials.csv (or landscape.csv), summary.csv and run.log (everything printed, including
start and end times, host and Slurm job id). Slurm output goes to results/slurm_logs/.

Rerunning a study replaces its folder. Runs that override the grid, methods or seed are
saved as <name>_custom unless --tag is given, so a quick test never replaces a full run.
"""
import argparse
import os
import shutil
import sys

# Parallelism comes from --jobs; multithreaded BLAS inside each worker oversubscribes the
# node and was measured 200x slower (POT distance matrices). Set before numpy is imported;
# export e.g. OMP_NUM_THREADS=8 yourself to override (e.g. for a GPU/torch method with --jobs 1).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import yaml  # noqa: E402

from dirreco import engine, plots, results  # noqa: E402


class _Tee:
    """Copy everything written to stdout/stderr into the study's logs/run.log."""
    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _list(s, typ=float):
    return [typ(v) for v in s.split(",")] if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--K", type=int, help="trials per grid point")
    ap.add_argument("--pitch", help="comma list, overrides sweep.pitch_mm")
    ap.add_argument("--N", help="comma list, overrides sweep.N (and drops n_total)")
    ap.add_argument("--n-total", help="comma list, overrides sweep.n_total (and drops N)")
    ap.add_argument("--methods", help="comma list of method labels to keep")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (fork)")
    ap.add_argument("--data", help="directory with vertices.npy + captures.npy, paired 1:1 (e.g. from "
                    "ldm/sample.py); replaces the config's data block")
    ap.add_argument("--tag", help="appended to the run directory name")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    sw = cfg["sweep"]
    if a.K:
        sw["K"] = a.K
    if a.pitch:
        sw["pitch_mm"] = _list(a.pitch)
    if a.N:
        sw["N"] = _list(a.N, int)
        sw.pop("n_total", None)
    if a.n_total:
        sw["n_total"] = _list(a.n_total, int)
        sw.pop("N", None)
    if a.methods:
        keep = a.methods.split(",")
        missing = set(keep) - set(cfg["methods"])
        if missing:
            ap.error(f"methods not in config: {sorted(missing)}")
        cfg["methods"] = {k: v for k, v in cfg["methods"].items() if k in keep}
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.data:
        d = os.path.abspath(a.data)
        cfg["data"] = {"vertices": os.path.join(d, "vertices.npy"),
                       "captures": os.path.join(d, "captures.npy"), "capture_offset": 0}

    overridden = any(v is not None for v in (a.K, a.pitch, a.N, a.n_total, a.methods, a.seed, a.data))
    tag = a.tag or ("custom" if overridden else None)
    name = cfg["name"] + (f"_{tag}" if tag else "")
    out = results.STUDIES_DIR / name
    if out.exists():
        shutil.rmtree(out)
    (out / "logs").mkdir(parents=True)
    logf = open(out / "logs" / "run.log", "w")
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, logf), _Tee(sys.__stderr__, logf)
    engine.run(cfg, out, jobs=a.jobs)

    run = results.load_run(out)
    if run.summary is not None:
        run.summary.to_csv(out / "logs" / "summary.csv", index=False)
    if not a.no_plot:
        plots.make_plots(run)
    print(f"Run directory: {out}")


if __name__ == "__main__":
    main()
