# NOTE: v0-6-6-4 2026-10-06 'Unified flexible execution engine, clean pandas logging, guarded MPI/circ'
# Author: Max A. A. Dornfest (University of Hawai'i at Manoa)
# Collaboration: FROST / Paper C
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson, vonmises, ks_2samp, chisquare, anderson, permutation_test, wasserstein_distance
from scipy.special import gamma, i0, i1, i0e, i1e, rel_entr
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon, mahalanobis
from astropy.stats import kuiper_two

# ------------------------------------------------------------------------------
# Defensive Imports for MPI, PyCircStat2, and TQDM (Node-agnostic compatibility)
# ------------------------------------------------------------------------------
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = type("DummyMPI", (), {"COMM_WORLD": None})()

try:
    from pycircstat2 import Circular
    from pycircstat2.hypothesis import V_test, watson_u2_test
    HAS_CIRC = True
except ImportError:
    HAS_CIRC = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


class DataProcessor:
    def __init__(self, n=10000, dx=150, gs=3):
        if HAS_MPI and MPI.COMM_WORLD is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        self.debug_L1 = False
        self.debug_L2 = False
        self.latex = False
        self.flow = False

        self.positron_file = "/data3/ratpac2/paper_B/money_plot/res_10M_001wt_ref/truth.txt"
        self.neutron_file = "/data3/ratpac2/paper_B/money_plot/res_10M_001wt_ref/neutrons.txt"
        self.vertices_file = "fid_vertices.npy"
        self.captures_file = "fid_captures.npy"

        self.grid_size = gs
        self.seg_size = dx
        self.true_angle = 0.0
        self.n = n

        self.l = (self.seg_size * self.grid_size) / 2.0
        self.x_range = (-self.l, self.l)
        self.y_range = (-self.l, self.l)

        if self.rank == 0:
            self.readData()

        self.all_angles = np.arange(-180, 180, 1.0)
        if self.size > 1 and self.comm is not None:
            self.angles = np.array_split(self.all_angles, self.size)[self.rank].tolist()
        else:
            self.angles = self.all_angles.tolist()

    def readData(self):
        t_start = time.time()
        if not os.path.exists(self.vertices_file) or not os.path.exists(self.captures_file):
            raise FileNotFoundError(f"Missing {self.vertices_file} or {self.captures_file} on host.")

        self.vertices = np.load(self.vertices_file)
        self.captures = np.load(self.captures_file)

        # Event index sync: offset captures by 3 to match vertices 1:1
        n = min(len(self.vertices), len(self.captures) - 3)
        self.captures = self.captures[3:3 + n]
        self.vertices = self.vertices[:n]

        self.coords = self.captures - self.vertices
        self.coords_buf = np.empty((0, 3), dtype=self.coords.dtype)

        # Segmented acceptance filter (usable events excluding prompt segment)
        R = self.seg_size / np.sqrt(2.0)
        r_sq = self.coords[:, 0]**2 + self.coords[:, 1]**2
        mask = r_sq > (R**2)
        self.usable_coords = self.coords[mask]
        self.usable_coords_buf = np.empty((0, 3), dtype=self.usable_coords.dtype)

        self.data = {
            "detected": {"main": self.coords, "buffer": self.coords_buf},
            "usable":   {"main": self.usable_coords, "buffer": self.usable_coords_buf}
        }
        print(f"[Rank 0] Ingested {len(self.coords)} events. Usable fraction for dx={self.seg_size}mm: "
              f"{(len(self.usable_coords)/len(self.coords))*100:.2f}% (Time: {time.time()-t_start:.2f}s)")

    def set_geometry(self, dx, gs):
        self.seg_size = dx
        self.grid_size = gs
        self.l = (self.seg_size * self.grid_size) / 2.0
        self.x_range = (-self.l, self.l)
        self.y_range = (-self.l, self.l)

        if self.rank == 0:
            R = self.seg_size / np.sqrt(2.0)
            r_sq = self.coords[:, 0]**2 + self.coords[:, 1]**2
            mask = r_sq > (R**2)
            self.usable_coords = self.coords[mask]
            self.usable_coords_buf = np.empty((0, 3), dtype=self.usable_coords.dtype)
            self.flow = False
            self.data["usable"] = {
                "main": self.usable_coords,
                "buffer": self.usable_coords_buf
            }

    def sampleData(self, n_samples, center=True):
        k = "detected" if center else "usable"
        if len(self.data[k]["main"]) < n_samples and not self.flow:
            self.flow = True
        if len(self.data[k]["buffer"]) < n_samples and self.flow:
            self.flow = False

        if not self.flow:
            idx = np.random.choice(self.data[k]["main"].shape[0], size=n_samples, replace=False)
            samples = self.data[k]["main"][idx]
            self.data[k]["main"] = np.delete(self.data[k]["main"], idx, axis=0)
            self.data[k]["buffer"] = np.vstack([self.data[k]["buffer"], samples])
        else:
            idx = np.random.choice(self.data[k]["buffer"].shape[0], size=n_samples, replace=False)
            samples = self.data[k]["buffer"][idx]
            self.data[k]["buffer"] = np.delete(self.data[k]["buffer"], idx, axis=0)
            self.data[k]["main"] = np.vstack([self.data[k]["main"], samples])
        return samples

    def get_disjoint_sample(self, n_sample, batch_idx=0):
        pool = self.usable_coords
        total_available = len(pool)
        k_max = total_available // n_sample
        if batch_idx >= k_max:
            start_idx = (batch_idx * n_sample) % max(1, total_available - n_sample)
        else:
            start_idx = batch_idx * n_sample
        return pool[start_idx : start_idx + n_sample]

    def rotateCoords(self, x_coords, y_coords, theta):
        phi = -theta * np.pi / 180.0
        c, s = np.cos(phi), np.sin(phi)
        x_rot = c * x_coords - s * y_coords
        y_rot = s * x_coords + c * y_coords
        return x_rot, y_rot

    def wrap_angle(self, angle):
        return ((angle + 180.0) % 360.0) - 180.0

    def binEvents(self, x, y):
        arr, _, _ = np.histogram2d(y, x, bins=self.grid_size, range=[self.x_range, self.y_range])
        s = np.sum(arr)
        return arr / s if s > 0 else np.zeros_like(arr)

    def evaluate_12_tests(self, flat_M, flat_A, flat_B, grid_M):
        sum_M = np.sum(flat_M)
        sum_A = np.sum(flat_A)
        sum_B = np.sum(flat_B)

        norm_M = (flat_M / (sum_M if sum_M > 0 else 1.0)) + 1e-6
        norm_A = (flat_A / (sum_A if sum_A > 0 else 1.0)) + 1e-6
        norm_B = (flat_B / (sum_B if sum_B > 0 else 1.0)) + 1e-6

        def bhatta(p, q):
            val = np.sum(np.sqrt(p * q))
            return -np.log(val) if val > 0 else 999.0

        def stat_diff(d1, d2, axis=0):
            return np.mean(d1, axis=axis) - np.mean(d2, axis=axis)

        res = {}
        try:
            k_s_A, k_p_A = kuiper_two(flat_M, flat_A)
            k_s_B, k_p_B = kuiper_two(flat_M, flat_B)
            res['kuiper'] = (k_s_A, k_p_A, k_s_B, k_p_B)
        except Exception:
            res['kuiper'] = (np.nan, np.nan, np.nan, np.nan)

        try:
            res['ks'] = (ks_2samp(flat_M, flat_A).statistic, ks_2samp(flat_M, flat_A).pvalue,
                         ks_2samp(flat_M, flat_B).statistic, ks_2samp(flat_M, flat_B).pvalue)
        except Exception:
            res['ks'] = (np.nan, np.nan, np.nan, np.nan)

        try:
            res['chi2'] = (chisquare(f_obs=norm_M, f_exp=norm_A).statistic, chisquare(f_obs=norm_M, f_exp=norm_A).pvalue,
                           chisquare(f_obs=norm_M, f_exp=norm_B).statistic, chisquare(f_obs=norm_M, f_exp=norm_B).pvalue)
        except Exception:
            res['chi2'] = (np.nan, np.nan, np.nan, np.nan)

        try:
            res['emd'] = (wasserstein_distance(flat_M, flat_A), wasserstein_distance(flat_M, flat_B))
        except Exception:
            res['emd'] = (np.nan, np.nan)

        try:
            res['kl'] = (np.sum(rel_entr(norm_M, norm_A)), np.sum(rel_entr(norm_M, norm_B)))
        except Exception:
            res['kl'] = (np.nan, np.nan)

        try:
            res['js'] = (jensenshannon(norm_M, norm_A), jensenshannon(norm_M, norm_B))
        except Exception:
            res['js'] = (np.nan, np.nan)

        res['bhatta'] = (bhatta(norm_M, norm_A), bhatta(norm_M, norm_B))

        try:
            if sum_M > 1:
                p_A = permutation_test((flat_M, flat_A), stat_diff, n_resamples=500)
                p_B = permutation_test((flat_M, flat_B), stat_diff, n_resamples=500)
                res['perm'] = (p_A.statistic, p_A.pvalue, p_B.statistic, p_B.pvalue)
            else:
                res['perm'] = (0.0, 1.0, 0.0, 1.0)
        except Exception:
            res['perm'] = (np.nan, np.nan, np.nan, np.nan)

        try:
            if self.grid_size <= 9 and sum_M > 2:
                cov_A = np.cov(grid_M, rowvar=False) + np.eye(self.grid_size) * 1e-3
                cov_inv_A = np.linalg.pinv(cov_A)
                m_len = min(self.grid_size, len(flat_M))
                res['maha'] = (mahalanobis(flat_M[:m_len], flat_A[:m_len], cov_inv_A[:m_len, :m_len]),
                               mahalanobis(flat_M[:m_len], flat_B[:m_len], cov_inv_A[:m_len, :m_len]))
            else:
                res['maha'] = (0.0, 0.0)
        except Exception:
            res['maha'] = (np.nan, np.nan)

        try:
            res['ad'] = (anderson(flat_M).statistic, anderson(flat_A).statistic, anderson(flat_B).statistic)
        except Exception:
            res['ad'] = (np.nan, np.nan, np.nan)

        if HAS_CIRC and sum_M > 0:
            try:
                def get_stat(r, default_val=np.nan):
                    s = getattr(r, 'statistic', getattr(r, 'V', getattr(r, 'u2', getattr(r, 'U2', None))))
                    if s is None and isinstance(r, (tuple, list)):
                        s = r[0]
                    return float(s) if s is not None else default_val

                c_M = Circular(data=np.deg2rad(flat_M), kwargs_median={'method': None})
                res['vtest'] = (get_stat(V_test(angle=0.0, alpha=c_M.alpha, verbose=False)), 0.0)

                angles_c = np.linspace(0, 2 * np.pi, len(flat_M), endpoint=False)
                w_M = np.maximum(0, flat_M.astype(int))
                w_A = np.maximum(0, flat_A.astype(int))
                w_B = np.maximum(0, flat_B.astype(int))

                cM = Circular(data=np.repeat(angles_c, w_M), kwargs_median={'method': None})
                cA = Circular(data=np.repeat(angles_c, w_A), kwargs_median={'method': None})
                cB = Circular(data=np.repeat(angles_c, w_B), kwargs_median={'method': None})

                res['watson'] = (get_stat(watson_u2_test([cM, cA])), 0.0,
                                 get_stat(watson_u2_test([cM, cB])), 0.0)
            except Exception:
                res['vtest'] = (np.nan, np.nan)
                res['watson'] = (np.nan, np.nan, np.nan, np.nan)
        else:
            res['vtest'] = (np.nan, np.nan)
            res['watson'] = (np.nan, np.nan, np.nan, np.nan)

        return res

    def run_flexible_angle_sweep(self, theta_min=-180.0, theta_max=179.0, theta_step=1.0,
                                 n_sample=10000, geoms=None, k_disjoint=10, out_file=None):
        if geoms is None:
            geoms = [(150, 3), (50, 9), (5, 89)]
        test_angles = np.arange(theta_min, theta_max + 0.5 * theta_step, theta_step)

        if out_file is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = f"review_deliverables/stats_angle_sweep_{date_str}.txt"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if self.rank == 0:
            header = ("# dx_mm\tth_true\tN\tFND_mean\tFND_std\tDef1_pct\tCent_mean\tCent_std\t"
                      "EMD_mean\tEMD_std\tKL_mean\tKL_std\tWatson_mean\tWatson_std")
            with open(out_file, "w") as f:
                f.write(header + "\n")
            print(f"\n[INFO] Launching Flexible Angle Sweep: [{theta_min}°, {theta_max}°], "
                  f"step={theta_step}°, N={n_sample}, K={k_disjoint} -> {out_file}")

        for dx, gs in geoms:
            self.set_geometry(dx, gs)
            usable_tot = len(self.usable_coords) if (self.rank == 0 and hasattr(self, 'usable_coords')) else 0
            if self.comm is not None:
                usable_tot = self.comm.bcast(usable_tot, root=0)

            k_avail = usable_tot // n_sample
            k_eval = min(k_disjoint, max(1, k_avail - 1))

            if self.rank == 0:
                print(f"\n--- GEOMETRY: dx = {dx:3d} mm | gs = {gs:2d}x{gs:2d} | Usable = {usable_tot} | K = {k_eval} ---")
                ref_sample = self.get_disjoint_sample(n_sample, batch_idx=0)
                ref_grid = self.binEvents(ref_sample[:, 0], ref_sample[:, 1])
                flat_ref = np.roll(np.rot90(ref_grid * n_sample).ravel(), -1)
                norm_ref = (flat_ref / np.sum(flat_ref)) + 1e-6
            else:
                ref_grid, flat_ref, norm_ref = None, None, None

            if self.comm is not None:
                ref_grid = self.comm.bcast(ref_grid, root=0)
                flat_ref = self.comm.bcast(flat_ref, root=0)
                norm_ref = self.comm.bcast(norm_ref, root=0)

            half_cells = gs // 2

            for th in test_angles:
                fnd_recons, cent_recons = [], []
                emds, kls, watsons = [], [], []
                quad_success = 0

                for k in range(1, k_eval + 1):
                    sample_k = self.get_disjoint_sample(n_sample, batch_idx=k) if self.rank == 0 else None
                    if self.comm is not None:
                        sample_k = self.comm.bcast(sample_k, root=0)

                    xk, yk = self.rotateCoords(sample_k[:, 0], sample_k[:, 1], th)
                    meas_coords = np.column_stack([xk, yk, sample_k[:, 2]])

                    local_norms = []
                    for trial_th in self.angles:
                        xr, yr = self.rotateCoords(meas_coords[:, 0], meas_coords[:, 1], trial_th)
                        rot_grid = self.binEvents(xr, yr)
                        local_norms.append(np.linalg.norm(ref_grid - rot_grid))

                    if self.comm is not None and self.size > 1:
                        all_norms = self.comm.gather(local_norms, root=0)
                    else:
                        all_norms = [local_norms]

                    if self.rank == 0:
                        stitched = [v for sl in all_norms for v in sl]
                        th_fnd = float(-self.all_angles[np.argmin(stitched)])
                        fnd_recons.append(th_fnd)

                        err_fnd = abs(((th_fnd - th + 180.0) % 360.0) - 180.0)
                        if err_fnd <= 45.0:
                            quad_success += 1

                        grid_k = self.binEvents(xk, yk) * n_sample
                        xc = (np.arange(gs) - half_cells) * dx
                        yc = (np.arange(gs) - half_cells) * dx
                        x_cent = np.sum(np.sum(grid_k, axis=0) * xc)
                        y_cent = np.sum(np.sum(grid_k, axis=1) * yc)
                        cent_recons.append(float(np.rad2deg(np.arctan2(y_cent, x_cent))))

                        flat_k = np.roll(np.rot90(grid_k).ravel(), -1)
                        norm_k = (flat_k / np.sum(flat_k)) + 1e-6
                        emds.append(wasserstein_distance(flat_k, flat_ref))
                        kls.append(np.sum(rel_entr(norm_k, norm_ref)))

                        if HAS_CIRC:
                            try:
                                angles_c = np.linspace(0, 2 * np.pi, len(flat_ref), endpoint=False)
                                w_k = np.maximum(0, flat_k.astype(int))
                                w_r = np.maximum(0, flat_ref.astype(int))
                                c_k = Circular(data=np.repeat(angles_c, w_k), kwargs_median={'method': None})
                                c_r = Circular(data=np.repeat(angles_c, w_r), kwargs_median={'method': None})
                                res_u2 = watson_u2_test([c_k, c_r])
                                u2_val = getattr(res_u2, 'statistic', getattr(res_u2, 'u2', None))
                                watsons.append(float(u2_val) if u2_val is not None else np.nan)
                            except Exception:
                                watsons.append(np.nan)

                if self.rank == 0:
                    rad_fnd = np.deg2rad(fnd_recons)
                    m_fnd = np.arctan2(np.mean(np.sin(rad_fnd)), np.mean(np.cos(rad_fnd)))
                    R_fnd = np.sqrt(np.mean(np.sin(rad_fnd))**2 + np.mean(np.cos(rad_fnd))**2)
                    s_fnd = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(1e-6, R_fnd)))))

                    rad_cent = np.deg2rad(cent_recons)
                    m_cent = np.arctan2(np.mean(np.sin(rad_cent)), np.mean(np.cos(rad_cent)))
                    R_cent = np.sqrt(np.mean(np.sin(rad_cent))**2 + np.mean(np.cos(rad_cent))**2)
                    s_cent = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(1e-6, R_cent)))))

                    quad_pct = (quad_success / k_eval) * 100.0
                    emd_m, emd_s = float(np.mean(emds)), float(np.std(emds))
                    kl_m, kl_s = float(np.mean(kls)), float(np.std(kls))
                    wat_m = float(np.nanmean(watsons)) if len(watsons) > 0 else np.nan
                    wat_s = float(np.nanstd(watsons)) if len(watsons) > 0 else np.nan

                    with open(out_file, "a") as f:
                        f.write(f"{dx}\t{th:.2f}\t{n_sample}\t{np.rad2deg(m_fnd):.4f}\t{s_fnd:.4f}\t"
                                f"{quad_pct:.2f}\t{np.rad2deg(m_cent):.4f}\t{s_cent:.4f}\t"
                                f"{emd_m:.4f}\t{emd_s:.4f}\t{kl_m:.6f}\t{kl_s:.6f}\t"
                                f"{wat_m:.6f}\t{wat_s:.6f}\n")

                    if int(round(th)) % 10 == 0:
                        print(f"[{dx:3d}mm] th={th:6.1f}° | FND={np.rad2deg(m_fnd):6.2f}±{s_fnd:4.2f}° | "
                              f"Cent={np.rad2deg(m_cent):6.2f}±{s_cent:4.2f}° | EMD={emd_m:6.2f} | KL={kl_m:.4f}")

        if self.rank == 0:
            print(f"[SUCCESS] Flexible angle sweep complete: {out_file}\n")
        if self.comm is not None:
            self.comm.Barrier()

    def run_flexible_count_scaling(self, counts=None, meas_angle=10.0, geoms=None, k_trials=10, out_file=None):
        if counts is None:
            counts = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        if geoms is None:
            geoms = [(150, 3), (50, 9), (5, 89)]

        if out_file is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = f"review_deliverables/stats_scaling_sweep_{date_str}.txt"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if self.rank == 0:
            header = ("# dx_mm\tN\tth_true\tFND_mean\tFND_std\tCent_mean\tCent_std\t"
                      "EMD_mean\tEMD_std\tKL_mean\tKL_std\tBhatt_mean\tBhatt_std")
            with open(out_file, "w") as f:
                f.write(header + "\n")
            print(f"\n[INFO] Launching Flexible Count Scaling: N in {counts}, "
                  f"Target={meas_angle}°, K={k_trials} -> {out_file}")

        for dx, gs in geoms:
            self.set_geometry(dx, gs)
            usable_tot = len(self.usable_coords) if (self.rank == 0 and hasattr(self, 'usable_coords')) else 0
            if self.comm is not None:
                usable_tot = self.comm.bcast(usable_tot, root=0)

            if self.rank == 0:
                print(f"\n--- GEOMETRY: dx = {dx:3d} mm | gs = {gs:2d}x{gs:2d} | Usable Pool = {usable_tot} ---")
                n_ref = min(50000, usable_tot)
                ref_sample = self.get_disjoint_sample(n_ref, batch_idx=0)
                xr_ref, yr_ref = self.rotateCoords(ref_sample[:, 0], ref_sample[:, 1], meas_angle)
                ref_grid = self.binEvents(xr_ref, yr_ref)
                flat_ref = np.roll(np.rot90(ref_grid * n_ref).ravel(), -1)
                norm_ref = (flat_ref / np.sum(flat_ref)) + 1e-6
            else:
                ref_grid, flat_ref, norm_ref = None, None, None

            if self.comm is not None:
                ref_grid = self.comm.bcast(ref_grid, root=0)
                flat_ref = self.comm.bcast(flat_ref, root=0)
                norm_ref = self.comm.bcast(norm_ref, root=0)

            half_cells = gs // 2

            for N in counts:
                if dx == 150 and N > 500000:
                    if self.rank == 0:
                        print(f"[SKIP] dx = 150 mm capped at 500k events (7.68% containment limit).")
                    continue
                if N > usable_tot:
                    continue

                actual_trials = min(k_trials, max(2, usable_tot // N))
                fnd_list, cent_list = [], []
                emd_list, kl_list, bhatt_list = [], [], []

                for tr in range(actual_trials):
                    meas_sample = self.get_disjoint_sample(N, batch_idx=tr + 1) if self.rank == 0 else None
                    if self.comm is not None:
                        meas_sample = self.comm.bcast(meas_sample, root=0)

                    xm, ym = self.rotateCoords(meas_sample[:, 0], meas_sample[:, 1], meas_angle)
                    meas_coords = np.column_stack([xm, ym, meas_sample[:, 2]])

                    local_norms = []
                    for trial_th in self.angles:
                        xr, yr = self.rotateCoords(meas_coords[:, 0], meas_coords[:, 1], trial_th)
                        rot_grid = self.binEvents(xr, yr)
                        local_norms.append(np.linalg.norm(ref_grid - rot_grid))

                    if self.comm is not None and self.size > 1:
                        all_norms = self.comm.gather(local_norms, root=0)
                    else:
                        all_norms = [local_norms]

                    if self.rank == 0:
                        stitched = [v for sl in all_norms for v in sl]
                        th_fnd = float(-self.all_angles[np.argmin(stitched)])
                        fnd_list.append(th_fnd)

                        grid_m = self.binEvents(xm, ym) * N
                        xc = (np.arange(gs) - half_cells) * dx
                        yc = (np.arange(gs) - half_cells) * dx
                        x_cent = np.sum(np.sum(grid_m, axis=0) * xc)
                        y_cent = np.sum(np.sum(grid_m, axis=1) * yc)
                        cent_list.append(float(np.rad2deg(np.arctan2(y_cent, x_cent))))

                        flat_m = np.roll(np.rot90(grid_m).ravel(), -1)
                        norm_m = (flat_m / np.sum(flat_m)) + 1e-6
                        emd_list.append(wasserstein_distance(flat_m, flat_ref))
                        kl_list.append(np.sum(rel_entr(norm_m, norm_ref)))

                        val_bhatt = np.sum(np.sqrt(norm_m * norm_ref))
                        bhatt_list.append(-np.log(val_bhatt) if val_bhatt > 1e-12 else 50.0)

                if self.rank == 0:
                    rad_f = np.deg2rad(fnd_list)
                    mf = np.arctan2(np.mean(np.sin(rad_f)), np.mean(np.cos(rad_f)))
                    Rf = np.sqrt(np.mean(np.sin(rad_f))**2 + np.mean(np.cos(rad_f))**2)
                    sf = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(1e-6, Rf)))))

                    rad_c = np.deg2rad(cent_list)
                    mc = np.arctan2(np.mean(np.sin(rad_c)), np.mean(np.cos(rad_c)))
                    Rc = np.sqrt(np.mean(np.sin(rad_c))**2 + np.mean(np.cos(rad_c))**2)
                    sc = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(1e-6, Rc)))))

                    em, es = float(np.mean(emd_list)), float(np.std(emd_list))
                    km, ks = float(np.mean(kl_list)), float(np.std(kl_list))
                    bm, bs = float(np.mean(bhatt_list)), float(np.std(bhatt_list))

                    with open(out_file, "a") as f:
                        f.write(f"{dx}\t{N}\t{meas_angle:.2f}\t{np.rad2deg(mf):.4f}\t{sf:.4f}\t"
                                f"{np.rad2deg(mc):.4f}\t{sc:.4f}\t{em:.4f}\t{es:.4f}\t"
                                f"{km:.6f}\t{ks:.6f}\t{bm:.6f}\t{bs:.6f}\n")

                    print(f"[{dx:3d}mm] N={N:7d} | FND={np.rad2deg(mf):5.2f}±{sf:5.2f}° | "
                          f"Cent={np.rad2deg(mc):5.2f}±{sc:5.2f}° | EMD={em:6.2f} | KL={km:.4f}")

        if self.rank == 0:
            print(f"[SUCCESS] Flexible count scaling complete: {out_file}\n")
        if self.comm is not None:
            self.comm.Barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper C Unified Statistical Package (v0-6-6-4)")
    parser.add_argument("--mode", choices=["angle-sweep", "count-scaling", "paper-b-geoms", "benchmark"],
                        default="angle-sweep", help="Pipeline execution mode")
    parser.add_argument("--n-events", type=str, default="10000",
                        help="Comma-separated N values")
    parser.add_argument("--theta-min", type=float, default=-180.0, help="Minimum angle (deg)")
    parser.add_argument("--theta-max", type=float, default=179.0, help="Maximum angle (deg)")
    parser.add_argument("--theta-step", type=float, default=1.0, help="Angle step resolution (deg)")
    parser.add_argument("--meas-angle", type=float, default=10.0, help="Target injection angle (deg)")
    parser.add_argument("--geoms", type=str, default="150,50,5", help="Pitches (e.g. 150,50,5)")
    parser.add_argument("--trials", type=int, default=5, help="Number of disjoint batches / realizations")
    parser.add_argument("--tag", type=str, default="run", help="File identifier tag")
    args = parser.parse_args()

    proc = DataProcessor()

    dx_vals = [int(x) for x in args.geoms.split(",") if x.strip()]
    geom_tuples = []
    for d in dx_vals:
        gs = 3 if d >= 100 else (9 if d >= 25 else 89)
        geom_tuples.append((d, gs))

    date_tag = datetime.now().strftime("%Y%m%d")

    if args.mode == "angle-sweep":
        n_val = int(args.n_events.split(",")[0])
        out_name = f"review_deliverables/stats_angle_sweep_{args.tag}_{date_tag}.txt"
        proc.run_flexible_angle_sweep(
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            theta_step=args.theta_step,
            n_sample=n_val,
            geoms=geom_tuples,
            k_disjoint=args.trials,
            out_file=out_name
        )

    elif args.mode == "count-scaling":
        n_list = [int(x) for x in args.n_events.split(",") if x.strip()]
        out_name = f"review_deliverables/stats_count_scaling_{args.tag}_{date_tag}.txt"
        proc.run_flexible_count_scaling(
            counts=n_list,
            meas_angle=args.meas_angle,
            geoms=geom_tuples,
            k_trials=args.trials,
            out_file=out_name
        )

    if proc.rank == 0:
        print("=== Execution completed successfully ===")
    sys.exit(0)
