"""Sliding-layer reconstruction for CAT-Q on LFM3.

Sequential over decoder layers: each layer's quantizable linears are ternarized
against activations captured with the already-fake-quantized prefix active, then
the weights are overwritten in place so the next layer sees real activations of
the quantized prefix (the "sliding" in SliderQuant).

Per-linear objective: activation-weighted reconstruction, minimizing
||X @ W^T - X @ W_hat^T||^2 normalized by ||X @ W^T||^2, over the learnable
modulation factors. BitTern ships no training code; this is the re-derived
loop. shortcut: within a layer all linears use inputs captured at the same
pass (entry and exit linears alike) — finer per-position re-capture would need
block-level partial forwards; revisit if rung-1 evals underdeliver.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

from .lfm3_adapter import decoder_layers, quantizable_linears
from .quantizer import CatqParams, ternarize_weight


class CatqReconstructor:
    def __init__(self, model: nn.Module, group_size: int = 128, gamma: float = 0.8,
                 s0: float = 30.0, delta0: float = 0.5, lr: float = 5e-3,
                 epochs: int = 60):
        self.model = model
        self.cfg = dict(group_size=group_size, gamma=gamma, s0=s0,
                        delta0=delta0, lr=lr, epochs=epochs)
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
                captured[_name].append(args[0].detach().to("cpu", torch.float32))
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
            w, self.cfg["group_size"], self.cfg["gamma"], self.cfg["s0"], self.cfg["delta0"])
        params = params.to(w.device)
        opt = torch.optim.AdamW(params.parameters(), lr=self.cfg["lr"])
        epochs = self.cfg["epochs"]
        denom_w = w.detach()
        for epoch in range(epochs):
            t = epoch / max(epochs - 1, 1)
            for x in inputs:
                xf = x.reshape(-1, x.shape[-1]).to(w.device, w.dtype)
                with torch.no_grad():
                    y = xf @ denom_w.T
                    norm = y.pow(2).mean().clamp_min(1e-8)
                w_hat = ternarize_weight(denom_w, params, t)
                loss = (xf @ w_hat.T - y).pow(2).mean() / norm
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
                qual = f"model.layers.{idx}.{name}"
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
