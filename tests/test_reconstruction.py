"""Reconstruction pipeline tests on a toy LFM3-shaped stack."""
import torch
import torch.nn as nn

from ternarq.lfm3_adapter import quantizable_linears
from ternarq.quantizer import CatqParams, ternarize_weight
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

    # training must have moved at least one tensor away from plain ternarization
    moved = []
    for qual in manifest["tensors"]:
        mod = modules[qual]
        untrained = ternarize_weight(mod.weight.detach(),
                                     CatqParams.for_weight(mod.weight.detach(), 8), t=1.0)
        moved.append(not torch.allclose(mod.weight, untrained, atol=1e-6))
    assert any(moved)

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
    finally:
        hook.remove()
    # layer 1's capture pass runs after layer 0 is ternarized, so its inputs move
    during = torch.cat([c.reshape(-1, c.shape[-1]) for c in seen])
    tail = during[-before.shape[0]:]
    assert not torch.equal(before, tail)


def test_capture_keeps_inputs_paired_with_their_own_linear():
    """Every captured chunk must be as wide as the weight that consumes it."""
    model = ToyStack()
    batches = _batches()
    recon = CatqReconstructor(model, group_size=8, epochs=1)
    for layer in model.model.layers:
        captured = recon._capture(layer, batches, lambda m, b: m(b))
        for name, linear in quantizable_linears(layer).items():
            for chunk in captured[name]:
                assert chunk.shape[-1] == linear.weight.shape[-1], name


def test_shared_input_copies_do_not_alias_across_allocations():
    """A freed tensor's id can be reused; the cache must not hand back its copy."""
    cache = {}

    def copy(x):
        key = id(x)
        if key not in cache:
            cache[key] = (x, x.detach().to("cpu"))
        return cache[key][1]

    first = torch.ones(2, 4)
    c1 = copy(first)
    del first
    second = torch.zeros(2, 4)     # likely reuses the freed id
    c2 = copy(second)
    assert c2 is not c1
    assert c2.sum() == 0 and c1.sum() == 8


def test_resume_skips_fitted_tensors_and_reproduces_weights(tmp_path):
    """A killed run resumes at the first unfinished tensor, weights identical."""
    model_a, model_b = ToyStack(), ToyStack()
    model_b.load_state_dict(model_a.state_dict())
    batches = _batches()
    ra = CatqReconstructor(model_a, group_size=8, epochs=1, resume_dir=tmp_path)
    ra.run(batches, lambda m, b: m(b))
    weights_a = {n: p.detach().clone() for n, p in model_a.named_parameters()
                 if "weight" in n}
    rb = CatqReconstructor(model_b, group_size=8, epochs=1, resume_dir=tmp_path)
    fitted = []
    orig = rb._fit_linear
    rb._fit_linear = lambda *a, **k: (fitted.append(1), orig(*a, **k))[1]
    rb.run(batches, lambda m, b: m(b))
    assert not fitted, "resume re-fitted tensors that were already on disk"
    for n, w in weights_a.items():
        assert torch.equal(model_b.get_parameter(n), w), n
    # manifest survives too
    assert rb.manifest["tensors"].keys() == ra.manifest["tensors"].keys()
