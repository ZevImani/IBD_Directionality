#!/usr/bin/env python
"""Copy a generated sample directory, dropping events whose |r| = |capture - vertex| is beyond the
largest |r| in the real pool.

    python ldm/cut_outliers.py ldm/samples/e137_fix                     # -> ldm/samples/e137_fix_cut
    python ldm/cut_outliers.py ldm/samples/e137_fix --cut 450 -o ldm/samples/e137_fix_cut450

Default cut: max(max_A, max_B) from <src>/eval/metrics.json, i.e. the largest |r| in the real eval
split [0, 2M). That split is also the real closure pool (configs/ldm_closure.yaml), so the cut
removes nothing a real trial could contain. Kept events stay in their original order. meta.json
copies the source's and adds a "cut" block (threshold, counts, the removed |r| values).

Fallback: if metrics.json is missing or predates the "outliers" block (evaluate.py before
2026-11-25), the same max is computed here from the eval split, with evaluate.py's sample size
(metrics.json "n", else min(1M, len(src))). That loads 2M real events: ~2 min, ~1 GB.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def real_max_r(n):
    """max(max_A, max_B) as evaluate.py defines it: |capture - vertex| over eval events [0, 2n)."""
    from ldm.data.ibd import load_split
    v, c = load_split("eval", max_events=2 * n)
    r = np.linalg.norm(c - v, axis=1)
    return float(r[:n].max()), float(r[n:2 * n].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="sample directory with vertices.npy + captures.npy (from sample.py)")
    ap.add_argument("--cut", type=float, help="keep |r| <= CUT mm (default: real max from <src>/eval/metrics.json)")
    ap.add_argument("-o", "--out", help="output directory (default: <src>_cut)")
    a = ap.parse_args()

    src = os.path.abspath(a.src.rstrip("/"))
    out = os.path.abspath(a.out) if a.out else src + "_cut"
    if out == src:
        raise SystemExit("output directory must differ from the source")

    v = np.load(os.path.join(src, "vertices.npy"))
    c = np.load(os.path.join(src, "captures.npy"))
    if v.shape != c.shape:
        raise SystemExit(f"vertices {v.shape} and captures {c.shape} differ")

    if a.cut is not None:
        cut, cut_source = a.cut, "--cut"
    else:
        mfile = os.path.join(src, "eval", "metrics.json")
        m = json.load(open(mfile)) if os.path.exists(mfile) else {}
        if "outliers" in m:
            o = m["outliers"]
            cut, cut_source = max(o["max_A"], o["max_B"]), f"max(max_A, max_B) in {mfile}"
        else:
            n = int(m.get("n", min(1_000_000, len(v))))
            print(f"no outliers block in {mfile}; computing the real max |r| from eval events [0, {2 * n:,})")
            ma, mb = real_max_r(n)
            cut = max(ma, mb)
            cut_source = f"max(max_A, max_B) computed from the eval split, n={n} (max_A {ma:.2f}, max_B {mb:.2f})"
    rabs = np.linalg.norm(c.astype(np.float64) - v.astype(np.float64), axis=1)
    keep = rabs <= cut
    removed = np.sort(rabs[~keep])[::-1]

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "vertices.npy"), v[keep])
    np.save(os.path.join(out, "captures.npy"), c[keep])
    mfile = os.path.join(src, "meta.json")
    meta = json.load(open(mfile)) if os.path.exists(mfile) else {}
    meta["cut"] = dict(source=src, cut_mm=float(cut), cut_source=cut_source, n_in=int(len(v)),
                       n_kept=int(keep.sum()), n_removed=int((~keep).sum()),
                       removed_r_mm=[round(float(x), 1) for x in removed])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    print(f"{src} -> {out}")
    print(f"cut |r| <= {cut:.2f} mm ({cut_source})")
    print(f"kept {keep.sum():,} of {len(v):,}; removed {(~keep).sum()} "
          f"(|r| mm: {np.round(removed).astype(int).tolist()})")


if __name__ == "__main__":
    main()
