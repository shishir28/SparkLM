import torch

from sparklm.attention.self_attention_linear import SelfAttentionLinear_v1, SelfAttentionLinear_v2


def _example_inputs():
    return torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ],
        dtype=torch.float32,
    )


def test_self_attention_linear_v1_output_shape():
    torch.manual_seed(123)
    x = _example_inputs()
    module = SelfAttentionLinear_v1(d_in=3, d_out=2)

    out = module(x)

    assert out.shape == torch.Size([6, 2])
    assert torch.isfinite(out).all()


def test_self_attention_linear_v2_output_shape():
    torch.manual_seed(123)
    x = _example_inputs()
    module = SelfAttentionLinear_v2(d_in=3, d_out=2, qkv_bias=False)

    out = module(x)

    assert out.shape == torch.Size([6, 2])
    assert torch.isfinite(out).all()


def test_self_attention_linear_v1_parameters_receive_gradients():
    torch.manual_seed(123)
    x = _example_inputs()
    module = SelfAttentionLinear_v1(d_in=3, d_out=2)

    loss = module(x).sum()
    loss.backward()

    assert module.W_query.grad is not None
    assert module.W_key.grad is not None
    assert module.W_value.grad is not None
    assert torch.isfinite(module.W_query.grad).all()
    assert torch.isfinite(module.W_key.grad).all()
    assert torch.isfinite(module.W_value.grad).all()


def test_self_attention_linear_v2_parameters_receive_gradients():
    torch.manual_seed(123)
    x = _example_inputs()
    module = SelfAttentionLinear_v2(d_in=3, d_out=2, qkv_bias=False)

    loss = module(x).sum()
    loss.backward()

    assert module.W_query.weight.grad is not None
    assert module.W_key.weight.grad is not None
    assert module.W_value.weight.grad is not None
    assert torch.isfinite(module.W_query.weight.grad).all()
    assert torch.isfinite(module.W_key.weight.grad).all()
    assert torch.isfinite(module.W_value.weight.grad).all()


def test_self_attention_linear_v1_and_v2_match_when_weights_are_aligned():
    torch.manual_seed(123)
    x = _example_inputs()
    v1 = SelfAttentionLinear_v1(d_in=3, d_out=2)
    v2 = SelfAttentionLinear_v2(d_in=3, d_out=2, qkv_bias=False)

    with torch.no_grad():
        v2.W_query.weight.copy_(v1.W_query.T)
        v2.W_key.weight.copy_(v1.W_key.T)
        v2.W_value.weight.copy_(v1.W_value.T)

    assert torch.allclose(v1(x), v2(x), atol=1e-6)
