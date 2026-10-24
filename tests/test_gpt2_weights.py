from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparklm.gpt2_weights import assign, load_weights_into_gpt
from sparklm.model import GPTModel


TINY_CONFIG = {
    "vocab_size": 7,
    "context_length": 5,
    "embedding_dim": 4,
    "num_layers": 1,
    "num_heads": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


def values(shape, start):
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def fake_params():
    embedding_dim = TINY_CONFIG["embedding_dim"]
    expanded_dim = 4 * embedding_dim
    block = {
        "attn": {
            "c_attn": {
                "w": values((embedding_dim, 3 * embedding_dim), 100),
                "b": values((3 * embedding_dim,), 200),
            },
            "c_proj": {
                "w": values((embedding_dim, embedding_dim), 300),
                "b": values((embedding_dim,), 400),
            },
        },
        "mlp": {
            "c_fc": {
                "w": values((embedding_dim, expanded_dim), 500),
                "b": values((expanded_dim,), 600),
            },
            "c_proj": {
                "w": values((expanded_dim, embedding_dim), 700),
                "b": values((embedding_dim,), 800),
            },
        },
        "ln_1": {
            "g": values((embedding_dim,), 900),
            "b": values((embedding_dim,), 910),
        },
        "ln_2": {
            "g": values((embedding_dim,), 920),
            "b": values((embedding_dim,), 930),
        },
    }
    return {
        "wte": values(
            (TINY_CONFIG["vocab_size"], embedding_dim),
            0,
        ),
        "wpe": values(
            (TINY_CONFIG["context_length"], embedding_dim),
            50,
        ),
        "blocks": [block],
        "g": values((embedding_dim,), 940),
        "b": values((embedding_dim,), 950),
    }


def test_assign_rejects_shape_mismatch():
    destination = torch.empty(2, 3)
    source = np.empty((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        assign(destination, source)


def test_load_weights_assigns_each_model_parameter_once():
    model = GPTModel(TINY_CONFIG)
    parameter_count = len(dict(model.named_parameters()))

    with patch("sparklm.gpt2_weights.assign", side_effect=assign) as mocked_assign:
        load_weights_into_gpt(model, fake_params())

    assert mocked_assign.call_count == parameter_count


def test_load_weights_splits_and_transposes_qkv_parameters():
    model = GPTModel(TINY_CONFIG)
    params = fake_params()

    load_weights_into_gpt(model, params)

    q_w, k_w, v_w = np.split(params["blocks"][0]["attn"]["c_attn"]["w"], 3, axis=-1)
    q_b, k_b, v_b = np.split(params["blocks"][0]["attn"]["c_attn"]["b"], 3, axis=-1)
    attention = model.transformer_blocks[0].att
    assert torch.equal(attention.W_query.weight, torch.tensor(q_w.T))
    assert torch.equal(attention.W_key.weight, torch.tensor(k_w.T))
    assert torch.equal(attention.W_value.weight, torch.tensor(v_w.T))
    assert torch.equal(attention.W_query.bias, torch.tensor(q_b))
    assert torch.equal(attention.W_key.bias, torch.tensor(k_b))
    assert torch.equal(attention.W_value.bias, torch.tensor(v_b))


def test_output_head_copies_embedding_values_without_tying_parameters():
    model = GPTModel(TINY_CONFIG)
    load_weights_into_gpt(model, fake_params())

    assert torch.equal(model.out_head.weight, model.tok_embedding.weight)
    assert model.out_head.weight.data_ptr() != model.tok_embedding.weight.data_ptr()

    model.out_head.weight = model.tok_embedding.weight
    assert model.out_head.weight.data_ptr() == model.tok_embedding.weight.data_ptr()
