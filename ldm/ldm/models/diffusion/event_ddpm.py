"""DDPM directly on per-event vectors: no autoencoder (no latent space) and no conditioning.

Each IBD event is only a few numbers (the displacement r, see ldm/data/ibd.py), so the
diffusion runs in data space. The denoiser is the residual MLP in
ldm/modules/diffusionmodules/mlp.py, and the forward/reverse process is the base DDPM class.
Data are r / scale_mm and are not clipped during sampling (clip_denoised=False).

Every validation epoch also generates `eval_events` events with DDIM and logs gen-vs-real
physics metrics as val/gen_* (mean/std offsets and W1 per axis in mm, usable-fraction and
theta0 ratios at 50 mm pitch, centroid angle). The image logger shows the marginals.
"""
from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ldm import ibd_metrics
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import instantiate_from_config


class EventDDPM(DDPM):
    def __init__(self, dims=3, scale_mm=50.0, eval_events=100_000, eval_ddim_steps=250,
                 log_events=100_000, weight_decay=0.0, **kwargs):
        kwargs.setdefault("first_stage_key", "event")
        kwargs.setdefault("clip_denoised", False)
        kwargs["channels"] = dims
        kwargs["image_size"] = 1           # unused: events are vectors
        super().__init__(**kwargs)
        self.dims, self.scale_mm = int(dims), float(scale_mm)
        self.eval_events, self.eval_ddim_steps = int(eval_events), eval_ddim_steps
        self.log_events, self.weight_decay = int(log_events), weight_decay
        self._val_real, self._val_ref = [], None

    # ------------------------------------------------------------------ model plumbing
    def get_input(self, batch, k):
        return batch[k].to(memory_format=torch.contiguous_format).float()

    def apply_model(self, x_noisy, t, cond=None, **kwargs):
        """Interface used by DDIMSampler / PLMSSampler."""
        return self.model(x_noisy, t)

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        return self.p_sample_loop((batch_size, self.dims), return_intermediates=return_intermediates)

    @torch.no_grad()
    def sample_events(self, n, ddim_steps=1000, eta=0.0, batch_size=262_144, use_ema=True, progress=False):
        """n generated displacements, (n, dims) float64 in mm on the CPU.

        ddim_steps: number of DDIM steps; 0/None runs the full ancestral DDPM chain
        (num_timesteps steps). eta=0 is deterministic DDIM, eta=1 is DDPM-like.
        """
        out = []
        with (self.ema_scope() if use_ema else nullcontext()):
            sampler = DDIMSampler(self) if ddim_steps else None
            for i in tqdm(range(0, n, batch_size), desc="sampling", disable=not progress):
                b = min(batch_size, n - i)
                if sampler is not None:
                    x, _ = sampler.sample(S=ddim_steps, batch_size=b, shape=(self.dims,), eta=eta, verbose=False)
                else:
                    x = self.p_sample_loop((b, self.dims))
                out.append(x.double().cpu() * self.scale_mm)
        return torch.cat(out)

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        if sum(len(x) for x in self._val_real) < self.eval_events:
            self._val_real.append(batch[self.first_stage_key].detach().cpu())

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        if not self._val_real or self.eval_events <= 0:
            return
        real = (torch.cat(self._val_real)[:self.eval_events].double() * self.scale_mm).numpy()
        self._val_real = []
        self._val_ref = real
        gen = self.sample_events(len(real), ddim_steps=self.eval_ddim_steps).numpy()
        metrics = ibd_metrics.compare(real, gen)
        self.log_dict({f"val/gen_{k}": float(v) for k, v in metrics.items()},
                      prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        """Marginals of real vs generated displacements, as one image for main.ImageLogger."""
        real = self._val_ref
        if real is None:
            real = (batch[self.first_stage_key].double().cpu() * self.scale_mm).numpy()
        real = real[:self.log_events]
        gen = self.sample_events(len(real), ddim_steps=self.eval_ddim_steps).numpy()
        fig = ibd_metrics.comparison_figure(
            real, gen, labels=("validation", "generated"),
            title=f"step {self.global_step}, {len(real):,} events, DDIM {self.eval_ddim_steps}")
        return {"marginals": ibd_metrics.figure_to_tensor(fig)}

    # ------------------------------------------------------------------ optimisation
    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if not self.use_scheduler:
            return opt
        sched = instantiate_from_config(self.scheduler_config)
        return [opt], [{"scheduler": LambdaLR(opt, lr_lambda=sched.schedule), "interval": "step", "frequency": 1}]


def load_event_ddpm(config, ckpt, device=None):
    """Instantiate EventDDPM from an OmegaConf config and load a Lightning checkpoint (eval mode)."""
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd.get("state_dict", sd), strict=False)
    if missing or unexpected:
        print(f"[load] missing keys: {missing}\n[load] unexpected keys: {unexpected}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()
