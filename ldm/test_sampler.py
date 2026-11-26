#!/usr/bin/env python
"""Sampler tests that need no trained network (CPU, a few seconds).

    python test_sampler.py                 (from ldm/; exit code 1 on any failure; pytest also works)

The noise schedule is read from configs/ibd-ddpm.yaml, so a schedule change is tested too. The data
are a Gaussian toy with the IBD scale (mean 16.8 mm, std 44 mm per axis, i.e. 0.336 / 0.88 in model
units). For Gaussian data the exact noise predictor is known analytically, and a deterministic DDIM
sampler is then an affine map of its starting noise. Running two starting points (0 and 1) gives the
output mean and std exactly, with no sampling noise.

  test_step_ratios         Every DDIM step the sampler takes has alpha_bar[prev] / alpha_bar[t] <= 10.
                           A step with ratio r multiplies eps-prediction errors by ~sqrt(r). The
                           cosine schedule's clipped last beta gives r = 1000 at t=999 (r = 4000 for
                           DDIM-500), which produced 277 m outliers and a +1 mm mean shift with the
                           trained model (samples/sampler_ab, 2026-11-24).
  test_exact_bias          With the EXACT predictor, the output mean and std match the truth within the
                           discretisation error of each step count (the old "exact-denoiser test").
  test_error_amplification The exact predictor plus a constant error delta: the output moves by
                           delta * D. D must be within 1.5x of the continuous-time value
                           s * asinh(sigma_start / s). Analytic: guarded DDIM gives 1.08-1.19x; the
                           unguarded cosine sampler gives 3.6x (S=1000) and 6.5x (S=500).
                           The exact-bias test alone cannot see this: its predictor has no error.

Each cliff test also runs the sampler with the guard switched off (min_alpha_cumprod=0). If the schedule
has a cliff, that control must fail, which shows the test can detect it.
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldm.models.diffusion.ddim import DDIMSampler  # noqa: E402
from ldm.models.diffusion.plms import PLMSSampler  # noqa: E402
from ldm.modules.diffusionmodules.util import make_beta_schedule, make_ddim_timesteps  # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "ibd-ddpm.yaml")
P = OmegaConf.load(CFG).model.params
T = int(P.get("timesteps", 1000))
SCALE_MM = float(P.get("scale_mm", 50.0))
BETAS = make_beta_schedule(P.get("beta_schedule", "linear"), T, linear_start=P.get("linear_start", 1e-4),
                           linear_end=P.get("linear_end", 2e-2), cosine_s=P.get("cosine_s", 8e-3))
AC = np.cumprod(1 - BETAS)
M, S = 16.8 / SCALE_MM, 44.0 / SCALE_MM          # toy data mean / std per axis, model units
STEPS = (100, 250, 500, 1000)                     # DDIM step counts to test
MAX_RATIO = 10.0
MAX_AMPLIFICATION = 1.5
# (max |mean - truth| mm, max |std/truth - 1|) per step count. Analytic values for the cosine schedule
# (2026-11-24): S=1000 -0.023 mm / -0.12%; 500 -0.046 / -0.25%; 250 -0.046 / -0.49%; 100 -0.18 / -1.2%.
BIAS_TOL = {1000: (0.05, 0.002), 500: (0.07, 0.004), 250: (0.08, 0.007), 100: (0.25, 0.016)}
HAS_CLIFF = (AC[:-1] / AC[1:]).max() > MAX_RATIO  # the raw schedule has a step with ratio > 10


class GaussianToy:
    """Stand-in for EventDDPM: the exact eps-predictor for x0 ~ N(M, S^2) per axis, plus `delta`."""

    def __init__(self, delta=0.0):
        self.num_timesteps = T
        self.betas = torch.tensor(BETAS, dtype=torch.float64)
        self.alphas_cumprod = torch.tensor(AC, dtype=torch.float64)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), self.alphas_cumprod[:-1]])
        self.device = torch.device("cpu")
        self.parameterization = "eps"
        self.delta = delta

    def apply_model(self, x, t, c):
        a = self.alphas_cumprod[t].to(x.dtype)[:, None]
        return (1 - a).sqrt() * (x - a.sqrt() * M) / (a * S ** 2 + 1 - a) + self.delta


def ddim_affine(n_steps, delta=0.0, guard=1e-6):
    """(offset, gain, first_t): the DDIM output is offset + gain * z for starting noise z (per axis)."""
    smp = DDIMSampler(GaussianToy(delta))
    smp.make_schedule(n_steps, ddim_eta=0.0, verbose=False, min_alpha_cumprod=guard)
    z = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.float64)
    with torch.no_grad():
        x, _ = smp.ddim_sampling(None, (2, 3), x_T=z)
    x = x[:, 0].double().numpy()
    return x[0], x[1] - x[0], int(smp.ddim_timesteps[-1])


def step_ratios(n_steps, guard):
    ts = make_ddim_timesteps("uniform", n_steps, T, verbose=False, alphas_cumprod=AC, min_alpha_cumprod=guard)
    prev = np.r_[AC[0], AC[ts[:-1]]]
    return prev / AC[ts], ts


# ------------------------------------------------------------------------------------------- tests
def test_step_ratios():
    for n in STEPS:
        r, ts = step_ratios(n, 1e-6)
        print(f"  DDIM-{n:4d}: first t={ts[-1]}, max step ratio {r.max():.1f} (limit {MAX_RATIO:g})")
        assert r.max() <= MAX_RATIO, f"DDIM-{n}: a step with alpha_bar ratio {r.max():.0f} amplifies errors ~{np.sqrt(r.max()):.0f}x"
    plms = PLMSSampler(GaussianToy())
    plms.make_schedule(1000, verbose=False)
    assert plms.ddim_timesteps[-1] == step_ratios(1000, 1e-6)[1][-1], "PLMS does not use the guarded timesteps"
    if HAS_CLIFF:
        r, ts = step_ratios(1000, 0)
        print(f"  control, guard off: DDIM-1000 first t={ts[-1]}, max step ratio {r.max():.0f} (must exceed the limit)")
        assert r.max() > MAX_RATIO, "control: the test no longer sees the schedule's cliff"


def test_exact_bias():
    for n in STEPS:
        off, gain, t0 = ddim_affine(n)
        dmean, dstd = (off - M) * SCALE_MM, gain / S - 1
        tm, ts = BIAS_TOL[n]
        print(f"  DDIM-{n:4d} (first t={t0}): mean {dmean:+.3f} mm (tol {tm}), std {100 * dstd:+.3f}% (tol {100 * ts:.1f}%)")
        assert abs(dmean) <= tm and abs(dstd) <= ts, f"DDIM-{n}: exact-denoiser bias too large"


def test_error_amplification(delta=1e-3):
    def amp(n, guard):
        off0, _, t0 = ddim_affine(n, 0.0, guard)
        offd, _, _ = ddim_affine(n, delta, guard)
        sigma = np.sqrt((1 - AC[t0]) / AC[t0])
        return ((offd - off0) / delta) / (-S * np.arcsinh(sigma / S)), t0

    for n in STEPS:
        a, t0 = amp(n, 1e-6)
        print(f"  DDIM-{n:4d} (first t={t0}): error response {a:.2f}x continuous time (limit {MAX_AMPLIFICATION})")
        assert 0.8 <= a <= MAX_AMPLIFICATION, f"DDIM-{n}: eps errors amplified {a:.1f}x"
    if HAS_CLIFF:
        for n in (500, 1000):
            a, t0 = amp(n, 0)
            print(f"  control, guard off: DDIM-{n} (first t={t0}): {a:.2f}x (must exceed the limit)")
            assert a > MAX_AMPLIFICATION, "control: the test no longer sees the amplification"


if __name__ == "__main__":
    print(f"schedule {P.get('beta_schedule')} T={T}: alpha_bar_T = {AC[-1]:.2g}, raw schedule has a cliff: {HAS_CLIFF}")
    failed = []
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        print(name)
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(name)
    print("ALL PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
