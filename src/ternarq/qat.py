"""QAT rung: non-learnable absmean g=128 ternary fake-quant + STE + teacher KD.

Recipe borrowed from the Ternary-Mamba QAT result (arXiv:2606.18114): the scale
is recomputed per group every forward as mean(|w|) — learnable raw scales in
QAT-from-pretrained inflate and collapse ~90% of weights to zero; absmean
self-regulates the zero fraction near ~26-31%. Frozen BF16 teacher supplies
KD on logits.
"""
from __future__ import annotations

import os
import pickle
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


def bake_ternary(model: nn.Module, group_size: int = 128) -> dict[str, dict]:
    """Replace every wrapped weight by its ternary value and return codes+scales.

    The codes are taken from the full-precision weights; exporting that manifest
    reproduces the baked values exactly. Re-deriving absmean codes from the
    baked weights does not: mean|w| of a ternary tensor is smaller than the
    group scale by mean|codes|, so the scales would shrink.
    """
    manifest = {}
    with torch.no_grad():
        for idx, layer in enumerate(decoder_layers(model)):
            for name, mod in layer.named_modules():
                if not isinstance(mod, FakeQuantLinear):
                    continue
                w = mod.linear.weight
                g, pad, alpha, codes = _absmean_codes(w.detach(), group_size)
                manifest[layer_tensor_name(idx, name)] = {
                    "codes": codes.reshape(-1, codes.shape[-2], codes.shape[-1]).to(torch.int8),
                    "alpha": alpha.reshape(-1, alpha.shape[-2], alpha.shape[-1]).float(),
                }
                q = (codes * alpha).reshape(*g.shape[:-2], -1)
                if pad:
                    q = q[..., :-pad]
                w.copy_(q)
    return manifest


def save_ternary_manifest(out_dir: str | Path, manifest: dict,
                          group_size: int = 128) -> Path:
    """Persist baked codes/scales in the layout `export_gguf --manifest` reads."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, out_dir / "ternary_codes.pt")
    meta = {"group_size": group_size,
            "tensors": {name: {"shape": None} for name in manifest}}
    path = out_dir / "ternary_manifest.json"
    path.write_text(json.dumps(meta, indent=1))
    return path


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
    """Per-token KL(student || teacher) on softened logits, scaled by T^2."""
    vocab = student_logits.shape[-1]
    s = F.log_softmax(student_logits.reshape(-1, vocab) / temperature, dim=-1)
    t = F.softmax(teacher_logits.reshape(-1, vocab) / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * temperature * temperature


def qat_loss(student_logits: torch.Tensor, input_ids: torch.Tensor,
             teacher_logits: torch.Tensor, temperature: float = 2.0,
             lam: float = 0.5) -> torch.Tensor:
    """Next-token CE plus KD, both shifted so logit i predicts token i+1."""
    s, t_logits = student_logits[:, :-1], teacher_logits[:, :-1]
    targets = input_ids[:, 1:]
    ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), targets.reshape(-1))
    return ce + lam * kd_loss(s, t_logits, temperature)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer, step: int,
                    keep_last: int | None = None) -> Path:
    """Write a resumable checkpoint atomically, rotating older ones if asked."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, tmp)
    os.replace(tmp, path)
    if keep_last is not None:
        older = sorted(path.parent.glob("step*.pt"),
                       key=lambda f: int(f.stem.removeprefix("step") or 0))
        for stale in older[:-keep_last]:
            stale.unlink()
    return path


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise SystemExit(
            f"checkpoint {path} is corrupt or truncated ({exc}); resume from an earlier step"
        ) from exc
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]
