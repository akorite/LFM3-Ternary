"""Reconstruction pipeline tests on a toy LFM3-shaped stack."""
import torch
import torch.nn as nn

from ternarq.lfm3_adapter import quantizable_linears
from ternarq.reconstruction import CatqReconstructor

from test_lfm3_adapter import _CausalLM


class ToyStack(_CausalLM):
    def forward(self, x):
        h = x if x.is_floating_point() else self.model.embed_tokens.weight[x]
        for layer in self.model.layers:
            if hasattr(layer, "self_attn"):
                a = layer.self_attn
                gated = a.q_proj(h) * torch.sigmoid(a.gate_proj(h).sum(-1, keepdim=True))
                h = h + a.o_proj(gated) + a.k_proj(h).sum(-1, keepdim=True) + a.v_proj(h).sum(-1, keepdim=True)
            else:
                c = layer.conv
                h = h + c.out_proj(c.in_proj(h).chunk(3, dim=-1)[-1])
            mlp = layer.feed_forward
            h = h + mlp.w2(torch.nn.functional.silu(mlp.w1(h)) * mlp.w3(h))
        return h


def _batches(n=4, d=32):
    torch.manual_seed(0)
    return [torch.randn(2, 8, d) for _ in range(n)]


def test_run_quantizes_all_linears_and_writes_manifest(tmp_path):
    model = ToyStack()
    recon = CatqReconstructor(model, group_size=8, epochs=3, s0=10.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    manifest = recon.run(_batches(), lambda m, b: m(b))

    names = set()
    for i, layer in enumerate(model.model.layers):
        for ln in quantizable_linears(layer):
            names.add(f"model.layers.{i}.{ln}")
    assert set(manifest["tensors"]) == names
    assert manifest["group_size"] == 8

    modules = dict(model.named_modules())
    for qual, entry in manifest["tensors"].items():
        mod = modules[qual]
        codes = entry["codes"].reshape(*entry["shape"][:-1], -1, 8).float()
        alpha = entry["alpha"]
        expect = (codes * alpha).reshape(entry["shape"])
        assert torch.allclose(mod.weight, expect, atol=1e-5), qual
        assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}

    for n, p in model.named_parameters():
        if n not in {f"{q}.weight" for q in names}:
            assert torch.equal(p, before[n]), n

    path = recon.save_manifest(tmp_path)
    assert path.exists() and (tmp_path / "ternary_codes.pt").exists()
    loaded = torch.load(tmp_path / "ternary_codes.pt", weights_only=True)
    assert set(loaded) == names


def test_second_layer_sees_quantized_first_layer():
    """Sliding property: layer 1's inputs change once layer 0 is ternarized."""
    model = ToyStack()
    batches = _batches()
    seen = []

    def record_w1(_m, args):
        seen.append(args[0].detach().clone())

    hook = model.model.layers[1].feed_forward.w1.register_forward_pre_hook(record_w1)
    try:
        with torch.no_grad():   # baseline: layer 0 still BF16
            for batch in batches:
                model(batch)
        before = torch.cat([c.reshape(-1, c.shape[-1]) for c in seen])
        seen.clear()
        CatqReconstructor(model, group_size=8, epochs=1).run(batches, lambda m, b: m(b))
        seen.clear()
        with torch.no_grad():   # layer 0 now holds ternary weights
            for batch in batches:
                model(batch)
    finally:
        hook.remove()
    after = torch.cat([c.reshape(-1, c.shape[-1]) for c in seen])
    assert not torch.equal(before, after)
