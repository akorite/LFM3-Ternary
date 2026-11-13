"""QAT rung tests: absmean ternary, STE, zero-ratio sentinel, resume."""
import json

import torch
import torch.nn as nn

from ternarq.qat import (
    FakeQuantLinear,
    absmean_scale,
    absmean_ternary,
    bake_ternary,
    install_qat,
    kd_loss,
    load_checkpoint,
    qat_loss,
    save_checkpoint,
    save_ternary_manifest,
    uninstall_qat,
    zero_ratio,
)
from ternarq.quantizer import group_view

from test_lfm3_adapter import _CausalLM


def test_absmean_scale_is_group_mean():
    w = torch.ones(2, 256)
    a = absmean_scale(w, 128)
    assert a.shape == (2, 2, 1) and torch.allclose(a, torch.ones_like(a))
    w2 = torch.full((1, 128), 4.0)
    assert absmean_scale(w2, 128).item() == 4.0


def test_absmean_ternary_dequantizes_to_codes():
    torch.manual_seed(0)
    w = torch.randn(4, 256)
    q = absmean_ternary(w, 128)
    g = group_view(q, 128)
    a = absmean_scale(w, 128)
    ratio = g / a
    assert torch.allclose(ratio, torch.round(ratio), atol=1e-5)
    assert set(torch.round(ratio).unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_ste_gradient():
    torch.manual_seed(1)
    w = torch.randn(4, 256, requires_grad=True)
    absmean_ternary(w, 128).sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_zero_ratio_band():
    torch.manual_seed(2)
    w = torch.randn(64, 256)
    r = zero_ratio(w, 128)
    # absmean self-regulates near ~0.3; the collapse signature is ~0.9
    assert 0.1 < r < 0.6 and r < 0.85


def test_fake_quant_linear_forward():
    torch.manual_seed(3)
    lin = nn.Linear(256, 8, bias=False)
    qlin = FakeQuantLinear(lin, 128)
    x = torch.randn(2, 256)
    w_q = absmean_ternary(lin.weight.detach(), 128)
    assert torch.allclose(qlin(x), x @ w_q.T, atol=1e-6)


def test_install_uninstall_roundtrip():
    model = _CausalLM()
    orig = dict(model.named_modules())
    wrapped = install_qat(model, 8)
    assert len(wrapped) == 2 * (5 + 3) + 2 * (2 + 3)
    uninstall_qat(model)
    assert not any(isinstance(m, FakeQuantLinear) for m in model.modules())
    assert model.model.layers[0].feed_forward.w1 is orig["model.layers.0.feed_forward.w1"]
    assert model.model.layers[2].self_attn.q_proj is orig["model.layers.2.self_attn.q_proj"]


def test_checkpoint_resume_restores_optimizer_state(tmp_path):
    torch.manual_seed(4)
    m1 = _CausalLM()
    install_qat(m1, 8)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    path = tmp_path / "ck.pt"

    x = torch.randn(1, 4, 32)
    for _ in range(3):
        loss = m1.model.layers[0].feed_forward.w1(x).pow(2).sum()
        opt1.zero_grad()
        loss.backward()
        opt1.step()
    save_checkpoint(path, m1, opt1, step=3)

    torch.manual_seed(4)
    m2 = _CausalLM()
    install_qat(m2, 8)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    assert load_checkpoint(path, m2, opt2) == 3
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
    # AdamW moments must survive, or the resumed trajectory silently differs
    s1, s2 = opt1.state_dict()["state"], opt2.state_dict()["state"]
    assert s1 and s1.keys() == s2.keys()
    for k in s1:
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(s1[k][moment], s2[k][moment])
        assert s1[k]["step"] == s2[k]["step"]


def test_kd_and_qat_loss():
    torch.manual_seed(5)
    s = torch.randn(2, 8, 50)
    t_logits = torch.randn(2, 8, 50)
    assert kd_loss(s, s).abs() < 1e-6
    targets = torch.randint(0, 50, (2, 8))
    loss = qat_loss(s, targets, t_logits)
    assert torch.isfinite(loss) and loss > 0
    assert qat_loss(torch.randn(1, 9, 50), torch.randint(0, 50, (1, 9)),
                    torch.randn(1, 9, 50)).isfinite()


def test_bake_ternary_writes_codes_scales_and_matches_manifest(tmp_path):
    torch.manual_seed(6)
    model = _CausalLM()
    install_qat(model, 8)
    manifest = bake_ternary(model, 8)
    assert len(manifest) == 26
    assert "model.layers.0.feed_forward.w1" in manifest

    modules = dict(model.named_modules())
    for name, entry in manifest.items():
        mod = modules[name]
        codes = entry["codes"].reshape(*entry["codes"].shape[:1], -1, 8).float()
        baked = (codes * entry["alpha"]).reshape(mod.linear.weight.shape)
        assert torch.allclose(mod.linear.weight, baked, atol=1e-5)

    path = save_ternary_manifest(tmp_path, manifest, 8)
    meta = json.loads(path.read_text())
    assert meta["group_size"] == 8 and set(meta["tensors"]) == set(manifest)
