"""CAT-Q ternarization core: learnable modulation (LM) + softened ternarization (ST).

Recipe (CAT-Q, arXiv:2606.26650): weights are flattened into non-overlapping
groups of `group_size` (128). Per group, three learnable factors modulate the
normalization statistics:

    mu_eff    = mu0 + tanh(d_mu)    * alpha0       # -1 < d_mu < 1 effective
    alpha_eff = alpha0 * (1 + tanh(d_alpha))
    delta_eff = delta0 * (1 + tanh(d_delta)), clamped to [1e-2, 1.5]

normalized weights w_hat = (w - mu_eff) / alpha_eff feed either:

    stage 1 (t <= gamma): soft ternarization
        f(w; s, delta) = [tanh(s*(w - delta)) + tanh(s*(w + delta))] / (2*tanh(s))
        with sharpness s ramped linearly 2 -> s0 across the stage;
    stage 2 (t > gamma): hard ternary codes via round-half-up threshold with
        straight-through gradients.

The final reconstruction is W ~ alpha_out * T with alpha_out the per-group
least-squares scale (no mean/zero-point: CAT-Q drops mu in the output form).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

S_START = 2.0  # soft-stage sharpness ramp start


def group_view(w: torch.Tensor, group_size: int, return_pad: bool = False):
    """[..., N] -> [..., N/group_size, group_size], zero-padding the tail."""
    n = w.shape[-1]
    pad = (-n) % group_size
    if pad:
        w = F.pad(w, (0, pad))
    g = w.reshape(*w.shape[:-1], -1, group_size)
    return (g, pad) if return_pad else g


def mu_alpha(g: torch.Tensor):
    """Per-group mean and std used as modulation anchors (last dim = group)."""
    mu = g.mean(dim=-1, keepdim=True)
    alpha = g.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return mu, alpha


def hard_ternary(w: torch.Tensor, delta: torch.Tensor | float) -> torch.Tensor:
    """sign(w) * [|w| >= delta]; codes in {-1, 0, +1}, round-half-up."""
    return torch.sign(w) * (w.abs() >= delta).to(w.dtype)


def soft_ternary(w: torch.Tensor, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Differentiable ternarization; |f| <= 1, f -> sign(w)*[|w|>0] as s -> inf."""
    return (torch.tanh(s * (w - delta)) + torch.tanh(s * (w + delta))) / (2 * torch.tanh(s))


def fit_output_scale(w_groups: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Per-group least-squares scale alpha minimizing ||w - alpha*codes||^2."""
    denom = codes.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (w_groups * codes).sum(dim=-1, keepdim=True) / denom


class CatqParams(nn.Module):
    """Learnable modulation factors for one weight tensor, laid out per group."""

    def __init__(self, shape, group_size: int, gamma: float, s0: float, delta0: float):
        super().__init__()
        self.group_size, self.gamma, self.s0, self.delta0 = group_size, gamma, s0, delta0
        n_groups = -(-shape[-1] // group_size) * int(torch.tensor(shape[:-1]).prod() or 1)
        self.d_mu = nn.Parameter(torch.zeros(n_groups))
        self.d_alpha = nn.Parameter(torch.zeros(n_groups))
        self.d_delta = nn.Parameter(torch.zeros(n_groups))
        self._shape = tuple(shape)

    @classmethod
    def for_weight(cls, w: torch.Tensor, group_size: int = 128, gamma: float = 0.8,
                   s0: float = 30.0, delta0: float = 0.5) -> "CatqParams":
        return cls(tuple(w.shape), group_size, gamma, s0, delta0).to(w.device, w.dtype)

    def _grouped_factors(self, w: torch.Tensor):
        g, pad = group_view(w, self.group_size, return_pad=True)
        mu0, alpha0 = mu_alpha(g.detach())
        shape = g.shape[:-2] + (g.shape[-2], 1)
        d_mu = self.d_mu.reshape(shape)
        d_alpha = self.d_alpha.reshape(shape)
        d_delta = self.d_delta.reshape(shape)
        mu_eff = mu0 + torch.tanh(d_mu) * alpha0
        alpha_eff = alpha0 * (1 + torch.tanh(d_alpha))
        delta_eff = (self.delta0 * (1 + torch.tanh(d_delta))).clamp(1e-2, 1.5)
        return g, pad, mu_eff, alpha_eff, delta_eff


def sharpness(t: float, gamma: float, s0: float) -> float:
    """Linear ramp S_START -> s0 across the soft stage (t in [0, gamma])."""
    return S_START + (s0 - S_START) * min(max(t / gamma, 0.0), 1.0)


def ternarize_weight(w: torch.Tensor, params: CatqParams, t: float,
                     return_parts: bool = False):
    """Reconstruct w as alpha*T using the CAT-Q LM + ST recipe at progress t."""
    g, pad, mu_eff, alpha_eff, delta_eff = params._grouped_factors(w)
    w_hat = (g - mu_eff) / alpha_eff
    if t <= params.gamma:  # differentiable ternarization
        codes = soft_ternary(w_hat, torch.as_tensor(sharpness(t, params.gamma, params.s0),
                                                    dtype=w.dtype, device=w.device), delta_eff)
    else:  # hard ternary with straight-through gradients
        hard = hard_ternary(w_hat, delta_eff)
        codes = w_hat + (hard - w_hat).detach()
    alpha = fit_output_scale(g, codes)
    recon = (codes * alpha).reshape(*g.shape[:-2], -1)
    if pad:
        recon = recon[..., :-pad]
    if return_parts:
        return recon, codes, alpha
    return recon
