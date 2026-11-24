"""Residual-MLP denoiser eps_theta(x_t, t) for low-dimensional event vectors.

Drop-in replacement for the UNet in a DDPM config (`unet_config.target`): same call
signature forward(x, timesteps, context=None, y=None), but x is (B, in_dim) instead of an
image. Each block is pre-norm, and the timestep conditions it through a FiLM scale and
shift. The last layer of every block and the output are zero-initialised, so the
untrained network is the identity map plus a zero prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import timestep_embedding, zero_module


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.film = nn.Linear(dim, 2 * dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = zero_module(nn.Linear(dim, dim))

    def forward(self, h, emb):
        z = self.fc1(F.silu(self.norm1(h)))
        scale, shift = self.film(emb).chunk(2, dim=1)
        z = self.norm2(z) * (1 + scale) + shift
        return h + self.fc2(self.drop(F.silu(z)))


class EventMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, n_blocks=6, time_dim=128, dropout=0.0):
        super().__init__()
        self.in_channels = in_dim          # read by generic sampling code
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), zero_module(nn.Linear(hidden, in_dim)))

    def forward(self, x, timesteps, context=None, y=None, **kwargs):
        emb = self.time_embed(timestep_embedding(timesteps, self.time_dim))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.out(h)
