"""Output layer: locate runs, summarise per-trial estimates, and read Max's legacy tables
into the same summary schema so one set of plotting code serves both.

Summary schema (one row per grid point x method):
  pitch_mm grid_size sweep_var N n_total theta_true eff theta0 [aug_*] method
  K mean bias sigma sigma_err sigma_circ rho Theta Theta_err Theta_exp eta
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import REPO, wrap

STUDIES_DIR = REPO / "results" / "studies"
LEGACY_DIR = REPO / "results" / "legacy_tables"
CONFIG_DIR = REPO / "configs"
PLOTS_DIR = REPO / "results" / "plots"


# ----------------------------------------------------------------------------- summary
def _stats(res):
    res = np.asarray(res, dtype=float)
    res = res[np.isfinite(res)]
    K = len(res)
    if K == 0:
        return pd.Series(dict(K=0, bias=np.nan, sigma=np.nan, sigma_circ=np.nan))
    r = np.radians(res)
    C, S = np.cos(r).mean(), np.sin(r).mean()
    bias = float(np.degrees(np.arctan2(S, C)))
    R = np.hypot(C, S)
    sigma = float(np.std(wrap(res - bias), ddof=1)) if K > 1 else np.nan
    sigma_circ = float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))))
    return pd.Series(dict(K=K, bias=bias, sigma=sigma, sigma_circ=sigma_circ))


def add_derived(s):
    """Add rho, Theta, Theta_exp, eta, error columns where inputs exist."""
    s = s.copy()
    if "grid_size" not in s:
        s["grid_size"] = np.round(450.0 / s["pitch_mm"]).astype(int)
    if "n_total" not in s and "eff" in s:
        s["n_total"] = s["N"] / s["eff"]
    if "sigma_err" not in s:
        s["sigma_err"] = s["sigma"] / np.sqrt(2.0 * (s["K"] - 1).clip(lower=1))
    s["rho"] = s["N"] / s["grid_size"] ** 2
    s["Theta"] = s["sigma"] * np.sqrt(s["N"])
    s["Theta_err"] = s["sigma_err"] * np.sqrt(s["N"])
    if "n_total" in s:
        s["Theta_exp"] = s["sigma"] * np.sqrt(s["n_total"])
    if "theta0" in s:
        s["eta"] = (s["theta0"] / s["Theta"]) ** 2
    return s


def summarize(trials):
    """Per-trial estimates -> one row per (grid point, method).

    sigma is the sample std (ddof=1) of the residual theta_hat - theta_true about its
    circular mean; bias is that circular mean. (Max's tables used np.std, ddof=0.)
    """
    t = trials.copy()
    t["res"] = wrap(t["theta_hat"] - t["theta_true"])
    keys = [c for c in t.columns if c not in ("trial", "theta_hat", "res")]
    s = t.groupby(keys, dropna=False, sort=True)["res"].apply(_stats).unstack().reset_index()
    s["K"] = s["K"].astype(int)
    s["mean"] = wrap(s["theta_true"] + s["bias"])
    return add_derived(s)


# ----------------------------------------------------------------------------- runs
class Run:
    def __init__(self, name, config=None, trials=None, summary=None, landscape=None,
                 directory=None, plot_dir=None):
        self.name = name
        self.config = config or {}
        self.trials = trials
        self.summary = summary
        self.landscape = landscape
        self.dir = directory
        self.plot_dir = Path(plot_dir or (directory / "plots" if directory else PLOTS_DIR / name))


def find_run(name):
    """results/studies/<name> (the folder name, e.g. scaling or scaling_test)."""
    d = STUDIES_DIR / name
    if not d.is_dir():
        have = sorted(p.name for p in STUDIES_DIR.iterdir() if (p / "logs" / "config.yaml").exists())
        raise FileNotFoundError(f"No study named '{name}' in {STUDIES_DIR}; have: {', '.join(have)}")
    return d


def load_run(source):
    """source: run directory | run name (latest) | 'legacy:<key>'."""
    if str(source).startswith("legacy:"):
        return load_legacy(str(source).split(":", 1)[1])
    d = Path(source)
    if not d.is_dir():
        d = find_run(str(source))
    cfg_file = d / "logs" / "config.yaml"
    cfg = yaml.safe_load(open(cfg_file)) if cfg_file.exists() else {}
    L = d / "logs"
    trials = pd.read_csv(L / "trials.csv") if (L / "trials.csv").exists() else None
    land = pd.read_csv(L / "landscape.csv") if (L / "landscape.csv").exists() else None
    if (L / "summary.csv").exists():
        summary = pd.read_csv(L / "summary.csv")
    else:
        summary = summarize(trials) if trials is not None else None
    return Run(d.name, cfg, trials, summary, land, directory=d)


# ----------------------------------------------------------------------------- legacy
# Max's tables (results/legacy_tables). `methods` maps his column -> our method label;
# `config` names the shipped config whose `plots:` list is used by default.
LEGACY = {
    "capacity_exposure": dict(
        files={"n_total": "stats_production_exposure_v3_*"},
        cols="dx_mm n_total N_usable usable_eff trials Cent_cont_std Cent_disc_std FND_para_std EMD_std",
        methods={"Cent_cont_std": "centroid_cont", "Cent_disc_std": "centroid_disc",
                 "FND_para_std": "fnd_parab", "EMD_std": "emd_paired"}),
    "scaling": dict(
        files={"N": "stats_unclipped_universal_scaling_*"},
        cols="dx_mm N rho trials Cent_disc_std FND_para_std EMD_para_std usable_eff",
        methods={"Cent_disc_std": "centroid_disc", "FND_para_std": "fnd_parab",
                 "EMD_para_std": "emd_paired"}),
    "emd_sparsity": dict(
        config="scaling", files={"N": "stats_emd_sparsity_sweep_*"},
        cols="dx_mm N rho trials Cent_disc_std FND_para_std EMD_std usable_eff",
        methods={"Cent_disc_std": "centroid_disc", "FND_para_std": "fnd_parab",
                 "EMD_std": "emd_paired"}),
    "phase_space": dict(
        files={"N": "stats_dx_vs_N_matrix_*"},
        cols="dx_mm N trials usable_eff grid_size Cent_cont_std Cent_disc_std FND_std",
        methods={"Cent_cont_std": "centroid_cont", "Cent_disc_std": "centroid_disc",
                 "FND_std": "fnd_argmin"}),
    "dense_kink": dict(
        config="phase_space", files={"N": "stats_dense_kink_audit_*"},
        cols="dx_mm N trials Cent_disc_std FND_std usable_eff",
        methods={"Cent_disc_std": "centroid_disc", "FND_std": "fnd_argmin"}),
    "smearing": dict(
        files={"N": "stats_smearing_sensitivity_*"},
        cols="dx_mm aug_smear_sigma_mm N trials Cent_std FND_std EMD_std",
        methods={"Cent_std": "centroid_disc", "FND_std": "fnd_parab", "EMD_std": "emd_paired"}),
    "fnd_refine": dict(
        files={"N": "stats_gabe_vs_parab_capacity_*", "n_total": "stats_gabe_vs_parab_exposure_*"},
        cols={"N": "dx_mm N_usable trials Cent_std FND_parab_std FND_sine_std eff",
              "n_total": "dx_mm n_total N_usable trials Cent_std FND_parab_std FND_sine_std eff"},
        methods={"Cent_std": "centroid_disc", "FND_parab_std": "fnd_parab",
                 "FND_sine_std": "fnd_sine"}),
    "angle_sweep": dict(
        files={"N": "stats_angle_sweep_full360_0p5deg_*"},
        cols="dx_mm theta_true N FND_mean FND_std Def1_pct Cent_mean Cent_std "
             "EMD_mean EMD_std KL_mean KL_std Watson_mean Watson_std",
        methods={"FND_std": "fnd_argmin", "Cent_std": "centroid_grid"},
        means={"FND_std": "FND_mean", "Cent_std": "Cent_mean"}),
}

_RENAME = {"dx_mm": "pitch_mm", "N_usable": "N", "usable_eff": "eff", "trials": "K"}


def _latest(pattern):
    hits = sorted(LEGACY_DIR.glob(pattern + ".txt"))
    if not hits:
        raise FileNotFoundError(f"No legacy table {LEGACY_DIR}/{pattern}.txt")
    return hits[-1]


def load_legacy(key):
    spec = LEGACY[key]
    parts = []
    for var, pattern in spec["files"].items():
        cols = spec["cols"][var] if isinstance(spec["cols"], dict) else spec["cols"]
        df = pd.read_csv(_latest(pattern), sep=r"\s+", comment="#", names=cols.split())
        df = df.rename(columns=_RENAME)
        df["sweep_var"] = var
        if "theta_true" not in df:
            df["theta_true"] = 0.0
        if "K" not in df:
            df["K"] = 10  # angle sweep: Max's K_disjoint default
        keep = [c for c in df.columns if c not in spec["methods"] and not c.endswith(("_mean", "_std", "_pct"))
                and c != "rho"]
        for col, label in spec["methods"].items():
            p = df[keep].copy()
            p["method"] = label
            p["sigma"] = df[col].astype(float)
            mcol = spec.get("means", {}).get(col)
            p["mean"] = df[mcol] if mcol else np.nan
            p["bias"] = wrap(p["mean"] - p["theta_true"]) if mcol else np.nan
            parts.append(p)
    s = add_derived(pd.concat(parts, ignore_index=True).dropna(subset=["sigma"]))

    trials = None
    if key == "fnd_refine":
        r = pd.read_csv(_latest("stats_gabe_vs_parab_residuals_*"), sep=r"\s+", comment="#",
                        names="pitch_mm trial theta_true fnd_parab fnd_sine diff_deg".split())
        r["N"] = 10000
        trials = r.melt(id_vars=["pitch_mm", "trial", "theta_true", "N"],
                        value_vars=["fnd_parab", "fnd_sine"], var_name="method", value_name="theta_hat")

    cfg_path = CONFIG_DIR / f"{spec.get('config', key)}.yaml"
    cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
    return Run(f"legacy_{key}", cfg, trials, s, plot_dir=PLOTS_DIR / f"legacy_{key}")
