import torch

from mxlstm.models.baseline_xlstm_transformer import XLSTMTransformer
from mxlstm.models.diffusion_rul import DiffusionRUL
from mxlstm.models.fusion import ConcatFusion, CrossAttentionFusion, GatedFusion
from mxlstm.models.liquid_wave_rul import LiquidWaveRUL
from mxlstm.models.mamba_blocks import BidirectionalMamba, MambaStack, _VanillaMamba3
from mxlstm.models.mamba_rul import MambaRUL
from mxlstm.models.mamba_xlstm_net import MambaXLSTMConfig, MambaXLSTMNet
from mxlstm.models.patch_tst_rul import PatchTSTRUL
from mxlstm.models.nbeats_rul import NBeatsRUL
from mxlstm.models.physics_nbeats_rul import NBeatsXLSTMRUL, PhysicsNBeatsRUL
from mxlstm.models.phase_moe_xlstm_rul import PhaseMoExLSTMRUL
from mxlstm.models.sparse_gate_tcn_rul import SparseGateTCNRUL
from mxlstm.models.vanilla_xlstm_rul import VanillaXLSTMRUL
from mxlstm.models.xlstm_blocks import VanillaMLSTMBlock, VanillaSLSTMCell, XLSTMStack


def test_vanilla_mamba3_forward_shape():
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16, n_groups=1)
    x = torch.randn(2, 10, 32)
    y = blk(x)
    assert y.shape == (2, 10, 32)


def test_vanilla_mamba3_multi_group_forward_shape():
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16, n_groups=2)
    x = torch.randn(2, 7, 32)
    assert blk(x).shape == (2, 7, 32)


def test_vanilla_mamba3_partial_rope_default():
    """Default rope_fraction=0.5 must rotate exactly half of d_state in pairs."""
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16)
    assert blk.split_tensor_size == 4  # half of 8
    assert blk.num_rope_angles == 2    # 4 / 2
    assert blk.rotary_dim_divisor == 4


def test_vanilla_mamba3_full_rope():
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16, rope_fraction=1.0)
    assert blk.split_tensor_size == 8
    assert blk.num_rope_angles == 4
    x = torch.randn(2, 6, 32)
    assert blk(x).shape == x.shape


def test_vanilla_mamba3_bc_bias_initialised_to_ones():
    """Per the official source, B_bias and C_bias init to ones."""
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16)
    assert torch.allclose(blk.B_bias, torch.ones_like(blk.B_bias))
    assert torch.allclose(blk.C_bias, torch.ones_like(blk.C_bias))
    assert blk.B_bias.shape == (blk.nheads, blk.d_state)


def test_vanilla_mamba3_has_bcnorm():
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16)
    assert hasattr(blk, "B_norm") and hasattr(blk, "C_norm")


def test_vanilla_mamba3_data_dependent_A_no_A_log_param():
    """A is data-dependent in Mamba-3, so no A_log parameter should exist."""
    blk = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16)
    assert not hasattr(blk, "A_log")


def test_vanilla_mamba3_outproj_norm_toggle():
    blk_off = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16, is_outproj_norm=False)
    blk_on = _VanillaMamba3(d_model=32, d_state=8, expand=2, headdim=16, is_outproj_norm=True)
    assert not hasattr(blk_off, "norm")
    assert hasattr(blk_on, "norm")
    x = torch.randn(2, 4, 32)
    assert blk_off(x).shape == blk_on(x).shape == x.shape


def test_vanilla_mamba3_accepts_mimo_kwargs_silently():
    blk = _VanillaMamba3(
        d_model=32, d_state=8, expand=2, headdim=16,
        is_mimo=True, mimo_rank=4, chunk_size=16, is_outproj_norm=True,
    )
    x = torch.randn(2, 6, 32)
    assert blk(x).shape == x.shape


def test_vanilla_mamba3_gradient_flows_through_all_params():
    """Smoke check: backward pass touches every learnable Mamba-3 component."""
    blk = _VanillaMamba3(d_model=16, d_state=8, expand=2, headdim=8, n_groups=1)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = blk(x).sum()
    y.backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_bidirectional_mamba_forward_shape():
    blk = BidirectionalMamba(d_model=32, d_state=8, headdim=16, backend="vanilla")
    x = torch.randn(2, 10, 32)
    assert blk(x).shape == x.shape


def test_mamba_stack_forward_shape():
    stk = MambaStack(
        d_model=32, num_blocks=2, d_state=8, headdim=16,
        bidirectional=True, backend="vanilla",
    )
    x = torch.randn(2, 10, 32)
    assert stk(x).shape == x.shape


def test_vanilla_mlstm_forward_shape():
    blk = VanillaMLSTMBlock(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    assert blk(x).shape == x.shape


def test_vanilla_slstm_forward_shape():
    blk = VanillaSLSTMCell(d_model=32)
    x = torch.randn(2, 10, 32)
    assert blk(x).shape == x.shape


def test_xlstm_stack_forward_shape_force_fallback():
    stk = XLSTMStack(d_model=32, context_length=10, num_blocks=3, slstm_positions=[1], n_heads=4, force_fallback=True)
    x = torch.randn(2, 10, 32)
    assert stk(x).shape == x.shape


def test_baseline_xlstm_transformer_forward():
    model = XLSTMTransformer(n_features=8, d_model=32, n_heads=4, encoder_layers=1, xlstm_blocks=2, slstm_positions=[1], context_length=10)
    x = torch.randn(2, 10, 8)
    y = model(x)
    assert y.shape == (2,)
    assert (y >= 0).all() and (y <= 1).all()


def test_mamba_xlstm_net_forward():
    cfg = MambaXLSTMConfig(
        n_features=8, d_model=32, context_length=10,
        xlstm_blocks=2, slstm_positions=[1], xlstm_heads=4, xlstm_force_fallback=True,
        mamba_blocks=1, mamba_d_state=8, mamba_headdim=16, mamba_backend="vanilla",
        fusion="gated",
    )
    model = MambaXLSTMNet(cfg)
    x = torch.randn(2, 10, 8)
    y = model(x)
    assert y.shape == (2,)
    assert (y >= 0).all() and (y <= 1).all()


def test_mamba_xlstm_net_return_hidden():
    cfg = MambaXLSTMConfig(
        n_features=8, d_model=32, context_length=10,
        xlstm_blocks=2, slstm_positions=[1], xlstm_heads=4, xlstm_force_fallback=True,
        mamba_blocks=1, mamba_d_state=8, mamba_headdim=16, mamba_backend="vanilla",
        fusion="gated",
    )
    model = MambaXLSTMNet(cfg)
    x = torch.randn(2, 10, 8)
    y, hidden = model(x, return_hidden=True)
    assert y.shape == (2,)
    assert hidden["fused"].shape == (2, 10, 32)
    assert hidden["gate"].shape == (2, 10, 32)


def _assert_scalar_rul_model_has_gradients(model, x):
    y = model(x)
    assert y.shape == (x.size(0),)
    assert (y >= 0).all() and (y <= 1).all()
    loss = y.mean()
    loss.backward()
    bad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and (param.grad is None or not torch.isfinite(param.grad).all())
    ]
    assert not bad, f"bad gradients for {bad}"


def test_liquid_wave_rul_forward_and_gradients():
    model = LiquidWaveRUL(n_features=8, n_bands=4, hidden_dim=16, attn_heads=4, ltc_unfolds=2)
    x = torch.randn(2, 10, 8)
    _assert_scalar_rul_model_has_gradients(model, x)
    explanation = model.explain(x)
    assert explanation["rul"].shape == (2,)
    assert explanation["band_weights"].shape == (2, 10, 4)


def test_nbeats_rul_forward_and_gradients():
    model = NBeatsRUL(
        context_length=10,
        n_features=8,
        hidden_dim=32,
        trend_blocks=1,
        wear_blocks=1,
        shock_blocks=1,
        n_harmonics=3,
        n_shock_basis=8,
    )
    x = torch.randn(2, 10, 8)
    _assert_scalar_rul_model_has_gradients(model, x)
    explanation = model.explain(x)
    assert explanation["rul"].shape == (2,)
    assert explanation["trend_contribution"].shape == (2,)


def test_diffusion_rul_forward_and_gradients():
    """v5 contract: ``score_head`` is trained via ``compute_loss`` (denoising
    auxiliary), not via leak into the RUL logit. Verify (a) forward shape and
    bounds, (b) every parameter -- including ``score_head`` -- receives a
    finite gradient through ``compute_loss``."""
    model = DiffusionRUL(
        n_features=8,
        context_length=10,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_noise_levels=2,
        film_num_embeddings=2,
    )
    x = torch.randn(2, 10, 8)
    y = model(x)
    assert y.shape == (2,)
    assert (y >= 0).all() and (y <= 1).all()
    target = torch.tensor([0.7, 0.2])
    cids = torch.tensor([0, 1], dtype=torch.long)
    loss = model.compute_loss(x, target, condition_ids=cids)
    loss.backward()
    bad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and (param.grad is None or not torch.isfinite(param.grad).all())
    ]
    assert not bad, f"bad gradients for {bad}"
    explanation = model.explain(x, condition_ids=cids)
    assert explanation["rul"].shape == (2,)
    assert explanation["feature_score"].shape == (2, 8)


def test_diffusion_rul_v5_inference_deterministic_when_scale_zero():
    """``inference_noise_scale=0`` must reproduce deterministic eval output."""
    model = DiffusionRUL(
        n_features=4,
        context_length=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_noise_levels=3,
        inference_noise_scale=0.0,
        n_inference_samples=2,
    )
    model.eval()
    x = torch.randn(3, 8, 4)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_diffusion_rul_v5_inference_stochastic_changes_output():
    """With ``inference_noise_scale>0`` two eval calls must differ."""
    model = DiffusionRUL(
        n_features=4,
        context_length=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_noise_levels=3,
        inference_noise_scale=0.5,
        n_inference_samples=2,
    )
    model.eval()
    x = torch.randn(3, 8, 4)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert not torch.allclose(a, b, atol=1e-6)


def test_sparse_gate_tcn_rul_forward_non_default_feature_count():
    """HI pipelines may use F≠16; placeholder feature names must align."""
    model = SparseGateTCNRUL(
        n_features=9,
        tcn_channels=(16, 16),
        attn_d_model=16,
        attn_heads=2,
        head_hidden=32,
    )
    x = torch.randn(2, 10, 9)
    y = model(x)
    assert y.shape == (2,)
    (model.compute_loss(x, torch.tensor([0.5, 0.4]))).backward()


def test_sparse_gate_tcn_rul_forward_and_pinball_loss():
    model = SparseGateTCNRUL(
        n_features=16,
        tcn_channels=(16, 16),
        attn_d_model=16,
        attn_heads=2,
        head_hidden=32,
    )
    x = torch.randn(2, 10, 16)
    y = model(x)
    assert y.shape == (2,) and (y >= 0).all() and (y <= 1).all()
    loss = model.compute_loss(x, torch.tensor([0.45, 0.55]))
    loss.backward()
    bad = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not bad, f"bad gradients for {bad}"


def test_phase_moe_xlstm_rul_forward_and_dense_loss():
    model = PhaseMoExLSTMRUL(n_features=8, d_model=32, n_phases=3, dropout=0.1)
    x = torch.randn(2, 10, 8)
    rul_window = torch.linspace(0.95, 0.15, 10).unsqueeze(0).expand(2, 10).clone()
    y = rul_window[:, -1].clone()
    pred = model(x)
    assert pred.shape == (2,)
    assert (pred >= 0).all() and (pred <= 1).all()
    loss = model.compute_loss(x, y, rul_window=rul_window)
    loss.backward()


def test_phase_moe_xlstm_rul_fallback_mse_when_no_rul_window():
    model = PhaseMoExLSTMRUL(n_features=8, d_model=32, n_phases=3, dropout=0.1)
    x = torch.randn(2, 10, 8)
    y = torch.tensor([0.5, 0.4])
    (model.compute_loss(x, y, rul_window=None)).backward()


def test_fusion_modules():
    a = torch.randn(2, 5, 16)
    b = torch.randn(2, 5, 16)
    assert GatedFusion(16)(a, b).shape == a.shape
    assert CrossAttentionFusion(16, n_heads=4)(a, b).shape == a.shape
    assert ConcatFusion(16)(a, b).shape == a.shape


def test_physics_nbeats_rul_forward_and_loss():
    model = PhysicsNBeatsRUL(
        10,
        8,
        "phm2012",
        film_num_embeddings=4,
        kurt_index=2,
        hidden_dim=32,
        trend_blocks=1,
        wear_blocks=1,
        shock_blocks=1,
    )
    x = torch.randn(3, 10, 8)
    c = torch.tensor([1, 2, 3])
    y = model(x, c)
    assert y.shape == (3,) and (y >= 0).all() and (y <= 1).all()
    loss = model.compute_loss(x, torch.tensor([0.4, 0.5, 0.6]), condition_ids=c)
    loss.backward()


def test_nbeats_xlstm_rul_forward_and_loss():
    model = NBeatsXLSTMRUL(
        10,
        8,
        "xjtusy",
        xlstm_d_model=32,
        xlstm_heads=4,
        film_num_embeddings=4,
        kurt_index=2,
        hidden_dim=32,
        trend_blocks=1,
        wear_blocks=1,
        shock_blocks=1,
    )
    x = torch.randn(2, 10, 8)
    c = torch.tensor([1, 2])
    y = model(x, c)
    assert y.shape == (2,)
    model.compute_loss(x, torch.tensor([0.5, 0.4]), condition_ids=c).backward()


def test_patch_tst_rul_forward():
    model = PatchTSTRUL(
        n_features=16,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        patch_len=8,
        stride=4,
        dropout=0.0,
        ffn_dim=128,
    )
    x = torch.randn(2, 32, 16, requires_grad=True)
    y = model(x)
    assert y.shape == (2,) and (y >= 0).all() and (y <= 1).all()
    y.sum().backward()


def test_mamba_rul_forward():
    model = MambaRUL(
        n_features=8,
        context_length=24,
        d_model=32,
        mamba_blocks=1,
        mamba_d_state=16,
        mamba_expand=2,
        mamba_headdim=16,
        mamba_backend="vanilla",
        head_hidden=16,
        dropout=0.0,
    )
    x = torch.randn(3, 24, 8, requires_grad=True)
    y = model(x)
    assert y.shape == (3,) and (y >= 0).all() and (y <= 1).all()
    y.sum().backward()


def test_vanilla_xlstm_rul_forward():
    model = VanillaXLSTMRUL(
        n_features=12,
        context_length=20,
        d_model=32,
        num_blocks=2,
        slstm_positions=[1],
        n_heads=4,
        head_hidden=16,
        dropout=0.0,
        xlstm_force_fallback=True,
    )
    x = torch.randn(4, 20, 12, requires_grad=True)
    y = model(x)
    assert y.shape == (4,) and (y >= 0).all() and (y <= 1).all()
    y.sum().backward()
