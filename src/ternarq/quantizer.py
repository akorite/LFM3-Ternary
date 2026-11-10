"""CAT-Q ternarization core: learnable modulation + softened ternarization.

Recipe: arXiv:2606.26650.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

S_START = 2.0


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
    """Learnable modulation factors for one weight tensor, laid out per group.

    The anchors (mean/std per group) are captured at construction, so callers
    must build these from the weight they will ternarize and not mutate it.
    """

    def __init__(self, group_size: int, gamma: float, s0: float, delta0: float,
                 mu0: torch.Tensor, alpha0: torch.Tensor):
        super().__init__()
        self.group_size, self.gamma, self.s0, self.delta0 = group_size, gamma, s0, delta0
        self.register_buffer("mu0", mu0.clone())
        self.register_buffer("alpha0", alpha0.clone())
        n_groups = mu0.numel()
        self.d_mu = nn.Parameter(torch.zeros(n_groups))
        self.d_alpha = nn.Parameter(torch.zeros(n_groups))
        self.d_delta = nn.Parameter(torch.zeros(n_groups))

    @classmethod
    def for_weight(cls, w: torch.Tensor, group_size: int = 128, gamma: float = 0.8,
                   s0: float = 30.0, delta0: float = 0.5) -> "CatqParams":
        g = group_view(w.detach(), group_size)
        mu0, alpha0 = mu_alpha(g)
        return cls(group_size, gamma, s0, delta0, mu0, alpha0).to(w.device, w.dtype)

    def _grouped_factors(self, w: torch.Tensor):
        g, pad = group_view(w, self.group_size, return_pad=True)
        shape = g.shape[:-2] + (g.shape[-2], 1)
        d_mu = self.d_mu.reshape(shape)
        d_alpha = self.d_alpha.reshape(shape)
        d_delta = self.d_delta.reshape(shape)
        mu_eff = self.mu0 + torch.tanh(d_mu) * self.alpha0
        alpha_eff = self.alpha0 * (1 + torch.tanh(d_alpha))
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
    if t <= params.gamma:
        codes = soft_ternary(w_hat, torch.as_tensor(sharpness(t, params.gamma, params.s0),
                                                    dtype=w.dtype, device=w.device), delta_eff)
    else:
        hard = hard_ternary(w_hat, delta_eff)
        codes = w_hat + (hard - w_hat).detach()
    alpha = fit_output_scale(g, codes)
    recon = (codes * alpha).reshape(*g.shape[:-2], -1)
    if pad:
        recon = recon[..., :-pad]
    if return_parts:
        return recon, codes, alpha
    return recon
