"""LFM3 block-type dispatch for CAT-Q.

LFM3 decoder layers carry either `self_attn` (full/sliding attention) or `conv`
(Lfm3ShortConv); every layer carries `feed_forward` (w1/w2/w3) and norms.
Quantized tensors: attention q/k/v/o + gate_proj; conv in_proj/out_proj;
MLP w1/w2/w3. Everything else — the depthwise conv kernel, all norms, tied
embeddings/LM head, cache state — stays BF16 by exclusion.
"""
from __future__ import annotations

import torch.nn as nn

ATTENTION_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")
CONV_LINEARS = ("in_proj", "out_proj")
MLP_LINEARS = ("w1", "w2", "w3")

_BLOCK = {"attention": ("self_attn", ATTENTION_LINEARS),
          "conv": ("conv", CONV_LINEARS)}



def block_kind(layer: nn.Module) -> str:
    if getattr(layer, "self_attn", None) is not None:
        return "attention"
    if getattr(layer, "conv", None) is not None:
        return "conv"
    raise ValueError(f"layer has neither self_attn nor conv: {type(layer).__name__}")


def quantizable_linears(layer: nn.Module) -> dict[str, nn.Linear]:
    """name -> nn.Linear for every ternary-scoped matmul in one decoder layer."""
    out: dict[str, nn.Linear] = {}
    prefix, names = _BLOCK[block_kind(layer)]
    op = getattr(layer, prefix)
    for n in names:
        mod = getattr(op, n, None)
        if isinstance(mod, nn.Linear):
            out[f"{prefix}.{n}"] = mod
    for n in MLP_LINEARS:
        mod = getattr(layer.feed_forward, n, None)
        if isinstance(mod, nn.Linear):
            out[f"feed_forward.{n}"] = mod
    return out


def layer_tensor_name(idx: int, name: str) -> str:
    """Qualified tensor name shared by the quantizer manifest and the exporter."""
    return f"model.layers.{idx}.{name}"


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Lfm3Model.layers; accepts the ForCausalLM wrapper or the inner model."""
    inner = getattr(model, "model", model)
    return inner.layers

