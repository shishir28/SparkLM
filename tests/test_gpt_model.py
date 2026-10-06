import torch

from sparklm.dummy_gpt_model import (
    DummyGPTModel,
    DummyLayerNorm,
    DummyTransformerBlock,
)


def tiny_config(**overrides):
    config = {
        "vocab_size": 31,
        "context_length": 8,
        "embedding_dim": 12,
        "num_layers": 2,
        "drop_rate": 0.0,
    }
    config.update(overrides)
    return config


def test_dummy_components_leave_their_input_unchanged():
    inputs = torch.randn(2, 4, 12)

    block_output = DummyTransformerBlock(tiny_config())(inputs)
    norm_output = DummyLayerNorm(12)(inputs)

    assert torch.equal(block_output, inputs)
    assert torch.equal(norm_output, inputs)


def test_forward_returns_one_logit_per_vocabulary_item():
    model = DummyGPTModel(tiny_config(vocab_size=50_257))
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits = model(token_ids)

    assert logits.shape == (2, 4, 50_257)
    assert torch.isfinite(logits).all()


def test_changing_a_future_token_does_not_change_earlier_logits():
    torch.manual_seed(123)
    model = DummyGPTModel(tiny_config()).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]])
    changed_token_ids = token_ids.clone()
    changed_token_ids[0, -1] = 5

    logits = model(token_ids)
    changed_logits = model(changed_token_ids)

    assert torch.equal(logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(logits[:, -1], changed_logits[:, -1])


def test_backward_reaches_embeddings_and_output_head():
    torch.manual_seed(123)
    model = DummyGPTModel(tiny_config())
    token_ids = torch.tensor([[1, 2, 3], [3, 4, 5]])

    model(token_ids).sum().backward()

    parameters = (
        model.token_embedding.weight,
        model.position_embedding.weight,
        model.out_head.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
