"""Adapter scope tests: ternary reaches every matmul and nothing else."""
import torch
import torch.nn as nn

from ternarq.lfm3_adapter import (
    block_kind,
    decoder_layers,
    quantizable_linears,
    tied_lm_head_uses_embedding,
)


class _Attn(nn.Module):
    def __init__(self, d=32, heads=4, kv=2):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv * (d // heads), bias=False)
        self.v_proj = nn.Linear(d, kv * (d // heads), bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, heads, bias=False)
        self.q_norm = nn.LayerNorm(d // heads)
        self.k_norm = nn.LayerNorm(d // heads)


class _ShortConv(nn.Module):
    def __init__(self, d=32, k=7):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=k, groups=d, bias=False)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)


class _MLP(nn.Module):
    def __init__(self, d=32, h=96):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)


class _Layer(nn.Module):
    def __init__(self, kind, d=32):
        super().__init__()
        if kind == "attention":
            self.self_attn = _Attn(d)
        else:
            self.conv = _ShortConv(d)
        self.feed_forward = _MLP(d)
        self.operator_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        self.post_operator_norm = nn.LayerNorm(d)
        self.post_ffn_norm = nn.LayerNorm(d)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)
        self.embedding_norm = nn.LayerNorm(32)
        self.layers = nn.ModuleList(
            [_Layer("conv") if i % 2 else _Layer("attention") for i in range(4)]
        )


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.lm_head = nn.Linear(32, 100, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # tied


def test_attention_block_linears():
    layer = _Layer("attention")
    lin = quantizable_linears(layer)
    assert sorted(lin) == [
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        "self_attn.gate_proj", "self_attn.k_proj", "self_attn.o_proj",
        "self_attn.q_proj", "self_attn.v_proj",
    ]


def test_conv_block_linears():
    layer = _Layer("conv")
    lin = quantizable_linears(layer)
    assert sorted(lin) == [
        "conv.in_proj", "conv.out_proj",
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    ]
    # depthwise kernel and norms never appear in the quantizable set
    assert all("conv.conv" not in n and "norm" not in n for n in lin)


def test_block_kind_dispatch():
    assert block_kind(_Layer("attention")) == "attention"
    assert block_kind(_Layer("conv")) == "conv"
    with __import__("pytest").raises(ValueError):
        block_kind(nn.Linear(4, 4))


def test_decoder_layers_and_tied_head():
    model = _CausalLM()
    assert len(decoder_layers(model)) == 4
    assert tied_lm_head_uses_embedding(model)


def test_scope_keeps_everything_else_bf16():
    model = _CausalLM()
    quantized_names = set()
    for i, layer in enumerate(decoder_layers(model)):
        quantized_names |= {f"model.layers.{i}.{n}.weight" for n in quantizable_linears(layer)}
    state_names = {n for n, _ in model.named_parameters() if n.endswith("weight")}
    untouched = state_names - quantized_names
    # embed, all norms, conv kernels stay out of the ternary set
    assert any("embed_tokens" in n for n in untouched)
    assert any("norm" in n for n in untouched)
    assert any("conv.conv" in n for n in untouched)
