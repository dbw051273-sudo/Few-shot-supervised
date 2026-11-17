"""Self-contained model backbones for Round-1 few-shot virtual staining on Workstation GPU 2."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Codec(nn.Module):
    """4x spatial autoencoder mapping [B, 3, 256, 256] <-> [B, 12, 64, 64]."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(48, 12, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(24, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class Block(nn.Module):
    """4-head Transformer block over concatenated state/condition latent tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4).unbind()
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(x)


class SwadAccumulator:
    """Running weight averager over the late flat-loss trajectory."""

    def __init__(self, modules: dict[str, nn.Module], start_step: int, every: int = 10):
        self.modules = modules
        self.start_step = start_step
        self.every = every
        self.count = 0
        self.sums = {
            name: {k: torch.zeros_like(v, dtype=torch.float32) for k, v in mod.state_dict().items()}
            for name, mod in modules.items()
        }

    def maybe_update(self, step: int, total_steps: int) -> None:
        if step >= self.start_step and (step % self.every == 0 or step == total_steps):
            self.count += 1
            for name, mod in self.modules.items():
                for k, v in mod.state_dict().items():
                    self.sums[name][k].add_(v.detach().float())

    def apply(self) -> None:
        if self.count == 0:
            return
        for name, mod in self.modules.items():
            avg = {
                k: (s / self.count).to(dtype=v.dtype)
                for (k, s), (_, v) in zip(self.sums[name].items(), mod.state_dict().items())
            }
            mod.load_state_dict(avg)


class ModularUniFlow(nn.Module):
    """Unified rectified-flow velocity network supporting four controlled prior/head layouts.

    Args:
        dim: Token hidden dimension (divisible by 4, e.g., 128 or 192).
        uni_mode: 'none' | '1x1' | 'up64_only'.
        head_mode: 'pixel_shuffle' | 'overlap_conv'.
    """

    def __init__(self, dim: int = 128, uni_mode: str = '1x1', head_mode: str = 'overlap_conv'):
        super().__init__()
        if uni_mode not in ('none', '1x1', 'up64_only'):
            raise ValueError(f'Unknown uni_mode: {uni_mode}')
        if head_mode not in ('pixel_shuffle', 'overlap_conv'):
            raise ValueError(f'Unknown head_mode: {head_mode}')

        patch = 4
        self.dim = dim
        self.uni_mode = uni_mode
        self.head_mode = head_mode
        self.use_uni_2d = uni_mode != 'none'

        self.state_proj = nn.Conv2d(12, dim, patch, stride=patch)
        self.pi_proj = nn.Conv2d(12, dim, patch, stride=patch)
        self.position = nn.Parameter(torch.randn(1, 16 * 16, dim) * 0.02)
        self.roles = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('frequency', torch.exp(torch.linspace(0.0, 6.0, 16)))
        self.blocks = nn.ModuleList(Block(dim) for _ in range(4))

        if uni_mode == '1x1':
            self.uni_proj = nn.Sequential(nn.Conv2d(1024, dim, 1), nn.SiLU())
            self.uni_zero1 = nn.Conv2d(dim, dim, 1)
            self.uni_zero2 = nn.Conv2d(dim, dim, 1)
            nn.init.zeros_(self.uni_zero1.weight)
            nn.init.zeros_(self.uni_zero1.bias)
            nn.init.zeros_(self.uni_zero2.weight)
            nn.init.zeros_(self.uni_zero2.bias)
        elif uni_mode == 'up64_only':
            hi_dim = dim // 2
            self.uni_hi_1x1 = nn.Sequential(nn.Conv2d(1024, hi_dim, 1), nn.SiLU())
            self.uni_hi_zero = nn.Conv2d(hi_dim, 12, 1)
            nn.init.zeros_(self.uni_hi_zero.weight)
            nn.init.zeros_(self.uni_hi_zero.bias)

        self.head_norm = nn.LayerNorm(dim)
        if head_mode == 'pixel_shuffle':
            self.head_proj = nn.Linear(dim, 12 * patch * patch)
            nn.init.zeros_(self.head_proj.weight)
            nn.init.zeros_(self.head_proj.bias)
        else:
            mid = dim // 2
            self.up32_conv = nn.Sequential(
                nn.Conv2d(dim, mid, 3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
            )
            self.up64_conv = nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
            )
            self.head_out = nn.Conv2d(mid, 12, 1)
            nn.init.zeros_(self.head_out.weight)
            nn.init.zeros_(self.head_out.bias)

    def _decode_head(self, state_tokens: torch.Tensor) -> torch.Tensor:
        if self.head_mode == 'pixel_shuffle':
            v_flat = self.head_proj(self.head_norm(state_tokens))
            return F.pixel_shuffle(v_flat.transpose(1, 2).reshape(-1, 192, 16, 16), 4)
        feat16 = self.head_norm(state_tokens).transpose(1, 2).reshape(-1, self.dim, 16, 16)
        feat32 = self.up32_conv(F.interpolate(feat16, size=(32, 32), mode='bilinear', align_corners=False))
        feat64 = self.up64_conv(F.interpolate(feat32, size=(64, 64), mode='bilinear', align_corners=False))
        return self.head_out(feat64)

    def forward(
        self,
        current: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
        uni_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        angle = t[:, None] * self.frequency[None]
        time_emb = self.time(torch.cat((angle.sin(), angle.cos()), dim=-1))[:, None]
        a = self.state_proj(current).flatten(2).transpose(1, 2) + self.position + self.roles[0] + time_emb
        b = self.pi_proj(source).flatten(2).transpose(1, 2) + self.position + self.roles[1] + time_emb
        a = self.blocks[0](a)
        b = self.blocks[1](b)
        hidden = torch.cat((a, b), dim=1)
        n_tok = a.shape[1]

        u_14 = None
        if self.use_uni_2d and uni_tokens is not None:
            u_14 = uni_tokens.transpose(1, 2).reshape(uni_tokens.shape[0], 1024, 14, 14)

        u16 = None
        if self.uni_mode == '1x1' and u_14 is not None:
            u16 = self.uni_proj(F.interpolate(u_14, size=(16, 16), mode='bilinear', align_corners=False))
            u1 = self.uni_zero1(u16).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :n_tok] + u1, hidden[:, n_tok:]), dim=1)
        hidden = self.blocks[2](hidden)
        if self.uni_mode == '1x1' and u16 is not None:
            u2 = self.uni_zero2(u16).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :n_tok] + u2, hidden[:, n_tok:]), dim=1)
        hidden = self.blocks[3](hidden)

        vel = self._decode_head(hidden[:, :n_tok])
        if self.uni_mode == 'up64_only' and u_14 is not None:
            u_hi = F.interpolate(self.uni_hi_1x1(u_14), size=(64, 64), mode='bilinear', align_corners=False)
            vel = vel + self.uni_hi_zero(u_hi)
        return vel


ARCH_SPECS = {
    'u0_base': {'uni_mode': 'none', 'head_mode': 'pixel_shuffle'},
    'u1_uni2d_1x1': {'uni_mode': '1x1', 'head_mode': 'pixel_shuffle'},
    'u1_up64_only': {'uni_mode': 'up64_only', 'head_mode': 'pixel_shuffle'},
    'u1_overlap': {'uni_mode': '1x1', 'head_mode': 'overlap_conv'},
}


def build_flow_model(arch_name: str, dim: int = 128) -> ModularUniFlow:
    if arch_name not in ARCH_SPECS:
        raise KeyError(f'Unsupported architecture: {arch_name}; expected one of {list(ARCH_SPECS)}')
    return ModularUniFlow(dim=dim, **ARCH_SPECS[arch_name])
