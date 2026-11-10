"""QAT rung: non-learnable absmean g=128 ternary fake-quant + STE + teacher KD.

Recipe borrowed from the Ternary-Mamba QAT result (arXiv:2606.18114): the scale
is recomputed per group every forward as mean(|w|) — learnable raw scales in
QAT-from-pretrained inflate and collapse ~90% of weights to zero; absmean
self-regulates the zero fraction near ~26-31%. Frozen BF16 teacher supplies
KD on logits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfm3_adapter import decoder_layers, layer_tensor_name, quantizable_linears
from .quantizer import group_view


def _absmean_codes(w: torch.Tensor, group_size: int):
    """Grouped view, pad and ternary codes at the absmean scale."""
    g, pad = group_view(w, group_size, return_pad=True)
    alpha = g.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return g, pad, alpha, torch.round(g / alpha).clamp(-1.0, 1.0)


def absmean_scale(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Per-group mean|w| scale, shape [..., n_groups, 1]. Non-learnable."""
    return group_view(w.detach(), group_size).abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)


def absmean_ternary(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Fake-quantize w to {-alpha,0,+alpha} per group with straight-through grad."""
    g, pad, alpha, codes = _absmean_codes(w, group_size)
    q = (codes * alpha).reshape(*g.shape[:-2], -1)
    if pad:
        q = q[..., :-pad]
    return w + (q - w).detach()


def zero_ratio(w: torch.Tensor, group_size: int = 128) -> float:
    """Fraction of weights that ternarize to 0 — the collapse sentinel."""
    return (_absmean_codes(w.detach(), group_size)[3] == 0).float().mean().item()


class FakeQuantLinear(nn.Module):

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        self.linear = linear
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, absmean_ternary(self.linear.weight, self.group_size),
                        self.linear.bias)


def install_qat(model: nn.Module, group_size: int = 128) -> list[str]:
    """Replace every ternary-scoped Linear with FakeQuantLinear; returns names."""
    wrapped = []
    for idx, layer in enumerate(decoder_layers(model)):
        for name, linear in quantizable_linears(layer).items():
            parent, attr = _resolve_parent(layer, name)
            setattr(parent, attr, FakeQuantLinear(linear, group_size))
            wrapped.append(layer_tensor_name(idx, name))
    return wrapped


def uninstall_qat(model: nn.Module) -> None:
    for layer in decoder_layers(model):
        for name, mod in layer.named_modules():
            if isinstance(mod, FakeQuantLinear):
                parent, attr = _resolve_parent(layer, name)
                setattr(parent, attr, mod.linear)


def _resolve_parent(layer: nn.Module, name: str):
    head, _, attr = name.rpartition(".")
    return (getattr(layer, head) if head else layer), attr


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 2.0) -> torch.Tensor:
    """KL(student || teacher) on softened logits, scaled by T^2."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * temperature * temperature


def qat_loss(student_logits: torch.Tensor, input_ids: torch.Tensor,
             teacher_logits: torch.Tensor, temperature: float = 2.0,
             lam: float = 0.5) -> torch.Tensor:
    """Next-token CE plus KD, both shifted so logit i predicts token i+1."""
    s, t_logits = student_logits[:, :-1], teacher_logits[:, :-1]
    targets = input_ids[:, 1:]
    ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), targets.reshape(-1))
    return ce + lam * kd_loss(s, t_logits, temperature)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer, step: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
