"""Tests for the CAT-Q ternarization core (LM factors + softened ternarization)."""
import torch

from ternarq.quantizer import (
    CatqParams,
    fit_output_scale,
    group_view,
    hard_ternary,
    soft_ternary,
    ternarize_weight,
)


def test_group_view_shapes():
    w = torch.randn(4, 256)  # 2 groups of 128 per row
    g = group_view(w, 128)
    assert g.shape == (4, 2, 128)
    # non-divisible dim pads with zeros and reports the pad
    g2, pad = group_view(torch.randn(2, 200), 128, return_pad=True)
    assert g2.shape == (2, 2, 128) and pad == 56


def test_hard_ternary_known_block():
    w = torch.tensor([[0.9, 0.8, -0.7, 0.05]])
    codes = hard_ternary(w, delta=0.5)
    assert codes.tolist() == [[1.0, 1.0, -1.0, 0.0]]
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_hard_ternary_threshold_boundary():
    w = torch.tensor([[0.49, 0.5, 0.51]])
    codes = hard_ternary(w, delta=0.5)
    assert codes.tolist() == [[0.0, 1.0, 1.0]]


def test_soft_ternary_limits():
    torch.manual_seed(0)
    w = torch.randn(2, 128)
    out = soft_ternary(w, torch.tensor(30.0), torch.tensor(0.5))
    assert out.abs().max().item() <= 1.0
    big = soft_ternary(torch.tensor([[5.0, -5.0]]), torch.tensor(30.0), torch.tensor(0.5))
    assert big[0, 0].item() > 0.99 and big[0, 1].item() < -0.99


def test_fit_output_scale_lstsq():
    torch.manual_seed(1)
    w = torch.randn(2, 256)
    g = group_view(w, 128)
    codes = hard_ternary(g / g.std(dim=-1, keepdim=True), 0.5)
    alpha = fit_output_scale(g, codes)
    assert alpha.shape == (2, 2, 1)
    assert (codes * alpha - g).pow(2).mean() < (codes - g).pow(2).mean()


def test_ternarize_weight_hard_stage_codes():
    torch.manual_seed(2)
    w = torch.randn(8, 256) * 0.03
    params = CatqParams.for_weight(w, group_size=128)
    recon, codes, alpha = ternarize_weight(w, params, t=1.0, return_parts=True)
    assert recon.shape == w.shape
    # stage-2 (hard): codes in {-1,0,1}, recon == codes*alpha regrouped
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    g_recon, pad = group_view(recon, 128, return_pad=True)
    assert torch.allclose(g_recon, codes * alpha)


def test_learnable_factors_shift_result():
    torch.manual_seed(3)
    w = torch.randn(4, 128)
    params = CatqParams.for_weight(w, group_size=128)
    base = ternarize_weight(w, params, t=0.0)
    with torch.no_grad():
        params.d_mu.fill_(2.0)
        params.d_alpha.fill_(2.0)
    shifted = ternarize_weight(w, params, t=0.0)
    assert not torch.allclose(base, shifted)


def test_ste_gradient_flows():
    torch.manual_seed(4)
    w = torch.randn(4, 128, requires_grad=True)
    params = CatqParams.for_weight(w.detach(), group_size=128)
    ternarize_weight(w, params, t=1.0).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert w.grad.abs().sum() > 0
    # LM factors receive gradients too
    assert params.d_mu.grad is not None


def test_stage_schedule():
    torch.manual_seed(5)
    w = torch.randn(2, 128)
    params = CatqParams.for_weight(w, group_size=128, gamma=0.8)
    _, codes, _ = ternarize_weight(w, params, t=0.9, return_parts=True)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    # soft stage output is continuous (not restricted to codes)
    soft = ternarize_weight(w, params, t=0.4)
    vals = group_view(soft, 128).flatten()
    assert not set(torch.round(vals / vals.abs().max() * 4).unique().tolist()) <= {-4.0, 0.0, 4.0}
