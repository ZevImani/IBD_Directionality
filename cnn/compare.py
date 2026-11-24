#!/usr/bin/env python
"""Print Theta = sigma*sqrt(N) per method and its ratio to a reference method, from a run.py study.

    python cnn/compare.py results/studies/cnn_p50_check
    python cnn/compare.py results/studies/cnn_capacity --ref centroid_cont --pitch 5
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", help="results/studies/<name> directory")
    ap.add_argument("--ref", default="centroid_disc", help="method the ratios are taken against")
    ap.add_argument("--pitch", type=float, help="only this pitch [mm]")
    a = ap.parse_args()

    s = pd.read_csv(Path(a.study) / "logs" / "summary.csv")
    if a.pitch is not None:
        s = s[s.pitch_mm == a.pitch]
    pd.set_option("display.width", 200)
    for (pitch, th), g in s.groupby(["pitch_mm", "theta_true"]):
        t = g.pivot_table(index="N", columns="method", values="Theta")
        err = g.pivot_table(index="N", columns="method", values="Theta_err")
        K = int(g["K"].min())
        print(f"=== pitch {pitch:g} mm, theta_true = {th:g} deg, K = {K}")
        print("Theta = sigma*sqrt(N) [deg]   (lower is better; +- Theta_err in brackets)")
        print(t.round(1).astype(str).add(" (" + err.round(1).astype(str) + ")").to_string())
        if a.ref in t:
            print(f"ratio to {a.ref}   (< 1 beats it; K={K} gives roughly +-{100 / (2 * (K - 1)) ** 0.5:.0f}% per entry)")
            print(t.div(t[a.ref], axis=0).round(3).to_string())
        print()


if __name__ == "__main__":
    main()
