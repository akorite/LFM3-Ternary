"""Sliding-layer reconstruction: each layer is ternarized against
activations of the already-quantized prefix.

shortcut: within a layer all linears use inputs captured at the same pass
(entry and exit linears alike) — finer per-position re-capture would need
block-level partial forwards; revisit if rung-1 evals underdeliver.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfm3_adapter import decoder_layers, layer_tensor_name, quantizable_linears
from .quantizer import CatqParams, ternarize_weight


class CatqReconstructor:
    def __init__(self, model: nn.Module, group_size: int = 128, gamma: float = 0.8,
                 s0: float = 30.0, delta0: float = 0.5, lr: float = 5e-3,
                 epochs: int = 60):
        self.model = model
        self.group_size, self.gamma, self.s0, self.delta0 = group_size, gamma, s0, delta0
        self.lr, self.epochs = lr, epochs
        self.manifest: dict[str, dict] = {"group_size": group_size, "tensors": {}}

    @torch.no_grad()
    def _capture(self, layer: nn.Module, batches: Iterable,
                 forward_fn: Callable[[nn.Module, object], object]) -> dict[str, list[torch.Tensor]]:
        """One forward pass over `batches`; returns linear-name -> input chunks."""
        linears = quantizable_linears(layer)
        captured = {name: [] for name in linears}
        hooks = []
        for name, mod in linears.items():
            def _hook(_m, args, _name=name):
                captured[_name].append(args[0].detach().to("cpu"))
            hooks.append(mod.register_forward_pre_hook(_hook))
        try:
            for batch in batches:
                forward_fn(self.model, batch)
        finally:
            for h in hooks:
                h.remove()
        return captured

    def _fit_linear(self, name: str, linear: nn.Linear,
                    inputs: list[torch.Tensor]) -> dict:
        w = linear.weight
        params = CatqParams.for_weight(
            w, self.group_size, self.gamma, self.s0, self.delta0)
        opt = torch.optim.AdamW(params.parameters(), lr=self.lr)
        epochs = self.epochs
        denom_w = w.detach()
        with torch.no_grad():
            targets = []
            for x in inputs:
                xf = x.reshape(-1, x.shape[-1]).to(w.device, w.dtype)
                y = F.linear(xf, denom_w)
                targets.append((xf, y, y.pow(2).mean().clamp_min(1e-8)))
        for epoch in range(epochs):
            t = epoch / max(epochs - 1, 1)
            for xf, y, norm in targets:
                w_hat = ternarize_weight(denom_w, params, t)
                loss = (F.linear(xf, w_hat) - y).pow(2).mean() / norm
                opt.zero_grad()
                loss.backward()
                opt.step()
        recon, codes, alpha = ternarize_weight(denom_w, params, 1.0, return_parts=True)
        with torch.no_grad():
            w.copy_(recon)
        return {"shape": list(w.shape),
                "codes": codes.reshape(-1, codes.shape[-2], codes.shape[-1]).to(torch.int8),
                "alpha": alpha.reshape(-1, alpha.shape[-2], alpha.shape[-1]).float()}

    def run(self, batches: Iterable,
            forward_fn: Callable[[nn.Module, object], object]) -> dict:
        """Quantize every decoder layer in place; returns the ternary manifest."""
        for idx, layer in enumerate(decoder_layers(self.model)):
            captured = self._capture(layer, batches, forward_fn)
            for name, linear in quantizable_linears(layer).items():
                qual = layer_tensor_name(idx, name)
                self.manifest["tensors"][qual] = self._fit_linear(qual, linear, captured[name])
        return self.manifest

    def save_manifest(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        codes = {name: {"codes": e["codes"], "alpha": e["alpha"]}
                 for name, e in self.manifest["tensors"].items()}
        torch.save(codes, out_dir / "ternary_codes.pt")
        meta = {"group_size": self.manifest["group_size"],
                "tensors": {name: {"shape": e["shape"]}
                            for name, e in self.manifest["tensors"].items()}}
        (out_dir / "ternary_manifest.json").write_text(json.dumps(meta, indent=1))
        return out_dir / "ternary_manifest.json"
