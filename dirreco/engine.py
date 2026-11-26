"""Sweep engine: data -> (augment, cut, split-half sample) -> methods -> trials.csv.

For every grid point (augmentation x pitch x N-or-n_total x theta_true) it runs K trials.
Each trial draws 2N disjoint usable events: N form the reference (theta = 0) and N form
the measurement, which is rotated CCW by theta_true (equivalently: the detector bins are
rotated by -theta_true before binning). With sweep.same_samples: true the same K draws are
reused for every theta_true, so any angle is obtained from the same data. Every method then returns an
estimate. One CSV row is written per (grid point, trial, method). Nothing is
summarised here; see results.summarize().
"""
import itertools
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import augment as aug
from .data import Geometry, load_events, theta0
from .methods import build
from .methods.template import ScanMethod

# Worker globals (inherited through fork, so the 450 MB pool is shared, not copied).
_POOL = None
_CFG = None
_METHODS = None
_AUGS = None
_CACHE = {}


# ----------------------------------------------------------------------------- config
def expand_theta(spec):
    if spec is None:
        return [0.0]
    if isinstance(spec, dict):
        return list(np.arange(spec["start"], spec["stop"] + spec["step"] / 2.0, spec["step"]))
    return [float(t) for t in spec]


def expand_augments(spec):
    """-> list of (columns dict, resolved step list); list-valued params are swept."""
    steps = (spec or {}).get("steps", [])
    if not steps:
        return [({}, [])]
    per_step = []
    for st in steps:
        keys = [k for k in st if k != "type"]
        vals = [st[k] if isinstance(st[k], list) else [st[k]] for k in keys]
        per_step.append([dict(type=st["type"], **dict(zip(keys, combo)))
                         for combo in itertools.product(*vals)])
    out = []
    for combo in itertools.product(*per_step):
        cols = {f"aug_{s['type']}_{k}": v for s in combo for k, v in s.items() if k != "type"}
        out.append((cols, list(combo)))
    return out


def make_tasks(cfg, n_augs):
    sw = cfg["sweep"]
    specs = [("N", int(v)) for v in sw.get("N", [])] + \
            [("n_total", int(v)) for v in sw.get("n_total", [])]
    if not specs:
        raise ValueError("sweep needs an `N` and/or `n_total` list")
    tasks = []
    sid = 0  # sample-stream id: one per (augmentation, pitch, N spec), shared across theta_true
    for ai in range(n_augs):
        for pitch in sw["pitch_mm"]:
            for var, val in specs:
                for th in expand_theta(sw.get("theta_true")):
                    tasks.append((len(tasks), sid, ai, float(pitch), var, val, float(th)))
                sid += 1
    return tasks


# ----------------------------------------------------------------------------- worker
def _pools(ai):
    key = ("aug", ai)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[0] == "aug"]:
            del _CACHE[k]
        steps = _AUGS[ai][1]
        if steps:
            rng = np.random.default_rng([_CFG["seed"], 0, ai])
            meas = aug.apply(_POOL, steps, rng)
            ref = meas if _CFG.get("augment", {}).get("apply_to", "both") == "both" else _POOL
        else:
            meas = ref = _POOL
        _CACHE[key] = (meas, ref)
    return _CACHE[key]


def _usable(ai, geom):
    key = ("cut", ai, geom)
    if key not in _CACHE:
        cut_keys = [k for k in _CACHE if k[0] == "cut"]
        for k in cut_keys[:max(0, len(cut_keys) - 3)]:
            del _CACHE[k]
        meas, ref = _pools(ai)
        m_ok = np.flatnonzero(geom.usable_mask(meas))
        r_ok = m_ok if ref is meas else np.flatnonzero(geom.usable_mask(ref))
        _CACHE[key] = (m_ok, r_ok, len(m_ok) / len(meas), theta0(meas.take(m_ok)))
    return _CACHE[key]


def _draw(rng, m_ok, r_ok, N, R):
    """N measurement + R reference events, disjoint. R = N reproduces the split-half draw
    (same random stream as before ref_mult existed)."""
    if m_ok is r_ok:
        idx = rng.choice(m_ok, N + R, replace=False)
        return idx[R:], idx[:R]
    mi = rng.choice(m_ok, N, replace=False)
    cand = rng.choice(r_ok, N + R, replace=False)
    return mi, cand[~np.isin(cand, mi)][:R]


def _run_task(task):
    tid, sid, ai, pitch, var, val, th = task
    t0 = time.time()
    cfg = _CFG
    geom = Geometry(pitch, cfg.get("detector", {}).get("width_mm", 450.0))
    meas_pool, ref_pool = _pools(ai)
    m_ok, r_ok, eff, th0 = _usable(ai, geom)
    N = val if var == "N" else int(round(val * eff))
    tag = f"dx={pitch:g} {var}={val} th={th:g} {_AUGS[ai][0] or ''}".strip()
    if N < cfg["sweep"].get("min_N", 5) or min(len(m_ok), len(r_ok)) < 2 * N:
        return tid, [], f"[skip] {tag}: N={N}, usable={len(m_ok)}"
    base = dict(pitch_mm=pitch, grid_size=geom.grid_size, sweep_var=var, N=N,
                n_total=(val if var == "n_total" else N / eff), theta_true=th,
                eff=eff, theta0=th0, **_AUGS[ai][0])
    # Reference size: the largest ref_mult among methods that run here and fit in the pool.
    # Methods with a smaller ref_mult use the first N * ref_mult reference events.
    avail = min(len(m_ok), len(r_ok))
    active, too_big = {}, []
    for label, m in _METHODS.items():
        if not m.applicable(N, geom):
            continue
        if m.needs_ref and N * (1 + m.ref_mult) > avail:
            too_big.append(label)
            continue
        active[label] = m
    R = N * max([m.ref_mult for m in active.values() if m.needs_ref] or [1])
    landscape = cfg.get("kind", "trials") == "landscape"
    # same_samples: true -> every theta_true sees the identical K samples (the "rotate the
    # bins, keep the data" construction); false -> fresh samples per theta_true.
    same = cfg["sweep"].get("same_samples", False)
    rng = np.random.default_rng([cfg["seed"], 1, sid] if same else [cfg["seed"], 2, tid])
    rows = []
    for k in range(cfg["sweep"]["K"]):
        mi, ri = _draw(rng, m_ok, r_ok, N, R)
        meas = meas_pool.take(mi).rotated(th)
        ref_all = ref_pool.take(ri)
        refs = {}
        losses = {}
        for label, m in active.items():
            k_ref = m.ref_mult * N
            if k_ref not in refs:
                refs[k_ref] = ref_all if k_ref == R else ref_all.take(slice(0, k_ref))
            ref = refs[k_ref]
            if isinstance(m, ScanMethod):
                key = m.loss_key(geom)
                if key not in losses:
                    losses[key] = m.loss(meas, ref, geom)
                if landscape:
                    rows += [dict(base, trial=k, method=label, phi=p, loss=l)
                             for p, l in zip(m.phis, losses[key])]
                    continue
                est = m.estimate(meas, ref, geom, losses=losses[key])
            elif landscape:
                continue
            else:
                est = m.estimate(meas, ref, geom)
            rows.append(dict(base, trial=k, method=label, theta_hat=est))
    note = f" | skipped (pool too small for ref_mult): {', '.join(too_big)}" if too_big else ""
    return tid, rows, f"{tag}: N={N} R={R} K={cfg['sweep']['K']} ({time.time() - t0:.1f}s){note}"


def _init_worker():
    """Per-worker setup. With several GPUs visible (e.g. --gres=gpu:4 and --jobs 4), pin
    worker i to GPU i mod n_gpus before anything initialises CUDA in this process."""
    _CACHE.clear()
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    ident = mp.current_process()._identity
    if len(gpus) > 1 and ident:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[(ident[0] - 1) % len(gpus)]


# ----------------------------------------------------------------------------- driver
def run(cfg, out_dir, jobs=1):
    """Run a resolved config: writes config.yaml and trials.csv (or landscape.csv)
    into out_dir/logs/ and prints timestamped progress, including when and where it ran
    (run.py copies everything printed into logs/run.log). Returns out_dir."""
    global _POOL, _CFG, _METHODS, _AUGS
    out_dir = Path(out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    cfg.setdefault("seed", int(time.time()) % 2**31)

    def log(msg):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    with open(out_dir / "logs" / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log(f"Started {datetime.now().isoformat(timespec='seconds')} on {platform.node()}"
        f" | slurm job {os.environ.get('SLURM_JOB_ID', '-')} | python {platform.python_version()},"
        f" numpy {np.__version__} | {' '.join(sys.argv)}")
    log(f"Loading events ({cfg.get('data') or 'default dataset'}) ...")
    _POOL = load_events(**(cfg.get("data") or {}))
    _CFG = cfg
    _METHODS = {label: build(label, spec) for label, spec in cfg["methods"].items()}
    _AUGS = expand_augments(cfg.get("augment"))
    tasks = make_tasks(cfg, len(_AUGS))
    log(f"{len(_POOL):,} events | {len(tasks)} grid points x K={cfg['sweep']['K']} | "
        f"methods: {', '.join(_METHODS)} | jobs={jobs}")

    out_csv = out_dir / "logs" / ("landscape.csv" if cfg.get("kind") == "landscape" else "trials.csv")
    if out_csv.exists():
        out_csv.unlink()
    first = True
    t_start = time.time()

    def handle(result, i):
        nonlocal first
        tid, rows, msg = result
        log(f"({i}/{len(tasks)}) {msg}")
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=first, index=False)
            first = False

    if jobs > 1:
        # Largest N first, so the slowest grid points don't form a tail with idle workers.
        # (Seeds are fixed per task/sample id, so the order does not change the results.)
        ordered = sorted(tasks, key=lambda t: -t[5])
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker) as p:
            for i, res in enumerate(p.imap_unordered(_run_task, ordered, chunksize=1), 1):
                handle(res, i)
    else:
        _init_worker()
        for i, t in enumerate(tasks, 1):
            handle(_run_task(t), i)

    log(f"Finished {datetime.now().isoformat(timespec='seconds')}, "
        f"{time.time() - t_start:.1f}s -> {out_csv}")
    return out_dir
