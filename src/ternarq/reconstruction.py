"""Sliding-layer reconstruction: each layer is ternarized against
activations of the already-quantized prefix.

shortcut: within a layer all linears use inputs captured at the same pass
(entry and exit linears alike) — finer per-position re-capture would need
block-level partial forwards; revisit if rung-1 evals underdeliver.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfm3_adapter import decoder_layers, layer_tensor_name, quantizable_linears
from .quantizer import CatqParams, ternarize_weight

logger = logging.getLogger(__name__)


class CatqReconstructor:
    def __init__(self, model: nn.Module, group_size: int = 128, gamma: float = 0.8,
                 s0: float = 30.0, delta0: float = 0.5, lr: float = 5e-3,
                 epochs: int = 60, resume_dir: str | Path | None = None,
                 fit_dtype: torch.dtype | None = None,
                 spool_dir: str | Path | None = None):
        self.model = model
        self.group_size, self.gamma, self.s0, self.delta0 = group_size, gamma, s0, delta0
        self.lr, self.epochs = lr, epochs
        self.resume_dir = Path(resume_dir) if resume_dir else None
        # fp16 on a tensor-core card runs these GEMMs ~16x faster than emulated
        # bf16; None keeps the model's own dtype (the A100 path)
        self.fit_dtype = fit_dtype
        self.spool_dir = Path(spool_dir) if spool_dir else None
        self.manifest: dict[str, dict] = {"group_size": group_size, "tensors": {}}
        if self.resume_dir is not None and (self.resume_dir / "tensors").is_dir():
            for f in sorted((self.resume_dir / "tensors").glob("*.pt")):
                self.manifest["tensors"][f.stem] = torch.load(
                    f, map_location="cpu", weights_only=True)
            if self.manifest["tensors"]:
                logger.info("resuming: %d tensors already fitted",
                            len(self.manifest["tensors"]))

    @torch.no_grad()
    def _capture(self, layer: nn.Module, batches: Iterable,
                 forward_fn: Callable[[nn.Module, object], object]) -> dict[str, list]:
        """One forward pass; returns linear-name -> per-batch input handles.

        Handles are CPU tensors when `spool_dir` is None (small model or
        plenty of RAM) and Paths to torch.save'd chunks when it is set — a
        256-batch layer otherwise holds ~21GB of activations on top of the
        model, which OOM-killed a 16GB-RAM host at layer 2."""
        linears = quantizable_linears(layer)
        captured = {name: [] for name in linears}
        shared: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        spool = self.spool_dir
        if spool is not None:
            spool.mkdir(parents=True, exist_ok=True)

        def _copy(x: torch.Tensor) -> torch.Tensor:
            """CPU copy of `x`, shared by every projection that reads it."""
            key = id(x)
            if key not in shared:
                # holding the source keeps its id unique: a freed tensor's id can
                # be reused by the next allocation mid-forward
                shared[key] = (x, x.detach().to("cpu"))
            return shared[key][1]

        hooks = []
        for name, mod in linears.items():
            def _hook(_m, args, _name=name):
                captured[_name].append(_copy(args[0]))
            hooks.append(mod.register_forward_pre_hook(_hook))
        try:
            for bi, batch in enumerate(batches):
                shared.clear()
                forward_fn(self.model, batch)
                if spool is not None:
                    for name, chunks in captured.items():
                        if len(chunks) > bi:        # this hook fired this batch
                            path = spool / f"{name.replace('.', '_')}__{bi:05d}.pt"
                            torch.save(chunks[bi], path)
                            chunks[bi] = path
        finally:
            for h in hooks:
                h.remove()
        return captured

    def _fit_linear(self, name: str, linear: nn.Linear,
                    inputs: list[torch.Tensor]) -> dict:
        if not inputs:
            raise RuntimeError(f"no captured inputs for {name}")
        w = linear.weight
        first = inputs[0]
        if isinstance(first, Path):
            first = torch.load(first, map_location="cpu", weights_only=True)
        if first.shape[-1] != w.shape[-1]:
            raise ValueError(
                f"{name}: captured inputs are {first.shape[-1]}-wide but the weight "
                f"expects {w.shape[-1]}")
        params = CatqParams.for_weight(
            w, self.group_size, self.gamma, self.s0, self.delta0)
        if self.fit_dtype is not None:
            # anchors and learnable factors stay fp32: fp16 would underflow
            # alpha0 dead groups back into the 0-division this rung already hit
            params.float()
        opt = torch.optim.AdamW(params.parameters(), lr=self.lr)
        skipped = 0
        epochs = self.epochs
        denom_w = w.detach()
        denom_fit = denom_w.to(self.fit_dtype) if self.fit_dtype else denom_w
        for epoch in range(epochs):
            t = epoch / max(epochs - 1, 1)
            for x in inputs:
                if isinstance(x, Path):
                    x = torch.load(x, map_location="cpu", weights_only=True)
                xf = x.reshape(-1, x.shape[-1]).to(w.device, w.dtype)
                xf_fit = xf.to(self.fit_dtype) if self.fit_dtype else xf
                # y is recomputed per epoch rather than cached: a 512-sample
                # corpus holds ~20 GB of targets for w1/w3, which no 24GB card
                # survives alongside the model.
                with torch.no_grad():
                    y = F.linear(xf_fit, denom_fit)
                    # fp32 accumulation: fp16 y^2 sums overflow at this width
                    norm = y.float().pow(2).mean().clamp_min(1e-8)
                w_hat = ternarize_weight(denom_w, params, t)
                if self.fit_dtype is not None:
                    pred = F.linear(xf_fit, w_hat.to(self.fit_dtype)).float()
                    loss = (pred - y.float()).pow(2).mean() / norm
                else:
                    loss = (F.linear(xf, w_hat) - y).pow(2).mean() / norm
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                opt.zero_grad()
                loss.backward()
                if not all(torch.isfinite(pp.grad).all()
                           for pp in params.parameters() if pp.grad is not None):
                    skipped += 1
                    continue
                opt.step()
        if skipped:
            logger.warning("%s: skipped %d non-finite optimizer steps", name, skipped)
        with torch.no_grad():
            recon, codes, alpha = ternarize_weight(denom_w, params, 1.0, return_parts=True)
            codes = codes.reshape(-1, codes.shape[-2], codes.shape[-1])
            alpha = alpha.reshape(-1, alpha.shape[-2], alpha.shape[-1])
            if not (torch.isfinite(alpha).all() and torch.isfinite(codes).all()):
                raise RuntimeError(f"{name}: fit produced non-finite codes or alpha")
            w.copy_(recon)
        return {"shape": list(w.shape),
                # the STE identity can round a code off {-1,0,1} in bf16; the
                # stored codes are the exact ternary values it approximates
                "codes": codes.round().clamp(-1, 1).to(torch.int8),
                "alpha": alpha.float()}

    def run(self, batches: Iterable,
            forward_fn: Callable[[nn.Module, object], object]) -> dict:
        """Quantize every decoder layer in place; returns the ternary manifest."""
        total = len(decoder_layers(self.model))
        for idx, layer in enumerate(decoder_layers(self.model)):
            logger.info("layer %d/%d: capturing %d batches", idx + 1, total, len(batches)
                        if hasattr(batches, "__len__") else -1)
            captured = self._capture(layer, batches, forward_fn)
            for name, linear in quantizable_linears(layer).items():
                qual = layer_tensor_name(idx, name)
                if qual in self.manifest["tensors"]:
                    self._apply_saved(linear, self.manifest["tensors"][qual])
                    continue
                entry = self._fit_linear(qual, linear, captured[name])
                self.manifest["tensors"][qual] = entry
                self._persist(qual, entry)
                for c in captured[name]:
                    if isinstance(c, Path):
                        c.unlink(missing_ok=True)
                logger.info("layer %d/%d: fit %s %s", idx + 1, total, qual,
                            tuple(linear.weight.shape))
            if self.spool_dir is not None:
                shutil.rmtree(self.spool_dir, ignore_errors=True)
        return self.manifest

    def _apply_saved(self, linear: nn.Linear, entry: dict) -> None:
        """Re-splice a previously fitted tensor's codes into the live weight."""
        w = linear.weight
        recon = (entry["codes"].float() * entry["alpha"].float()).reshape(w.shape)
        with torch.no_grad():
            w.copy_(recon.to(w.dtype))

    def _persist(self, name: str, entry: dict) -> None:
        """Drop one fitted tensor's codes on disk; survives a lost session.
        KAGGLE_OUTPUT mirrors each file into the kernel's returned output dir,
        so a 12h-capped kernel hands back every fit it finished."""
        import os
        dirs = [self.resume_dir / "tensors"] if self.resume_dir is not None else []
        ko = os.environ.get("KAGGLE_OUTPUT")
        if ko:
            dirs.append(Path(ko))
        for parts in dirs:
            parts.mkdir(parents=True, exist_ok=True)
            tmp = parts / f"{name}.tmp"
            torch.save(entry, tmp)
            tmp.rename(parts / f"{name}.pt")

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
