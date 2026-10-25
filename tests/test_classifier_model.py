import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from sparklm.classification import classifier_model
from sparklm.device import move_model_to_device
from sparklm.model import GPTModel


def test_base_config_matches_pretrained_gpt2_small():
    assert classifier_model.CHOOSE_MODEL == "gpt2-small (124M)"
    assert classifier_model.BASE_CONFIG == {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "embedding_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
    }


def test_initialize_pretrained_gpt2_loads_weights_and_sets_eval(
    monkeypatch,
):
    events = []
    params = object()

    class FakeModel:
        def __init__(self, config):
            events.append(("build", config.copy()))

        def eval(self):
            events.append(("eval", self))
            return self

    def fake_download(model_size, models_dir):
        events.append(("download", model_size, models_dir))
        return {}, params

    def fake_load(model, loaded_params):
        events.append(("load", model, loaded_params))

    fake_download_module = ModuleType("sparklm.gpt_download")
    fake_download_module.download_and_load_gpt2 = fake_download
    monkeypatch.setitem(
        sys.modules,
        "sparklm.gpt_download",
        fake_download_module,
    )
    monkeypatch.setattr(classifier_model, "GPTModel", FakeModel)
    monkeypatch.setattr(classifier_model, "load_weights_into_gpt", fake_load)

    models_dir = object()
    model = classifier_model.initialize_pretrained_gpt2(models_dir)

    assert events == [
        ("download", "124M", models_dir),
        ("build", classifier_model.BASE_CONFIG),
        ("load", model, params),
        ("eval", model),
    ]


def test_run_pretrained_sanity_checks_uses_both_prompts(
    monkeypatch,
    capsys,
):
    generated_calls = []
    model = object()
    tokenizer = object()

    def fake_generate(
        generated_model,
        idx,
        max_new_tokens,
        context_size,
    ):
        generated_calls.append(
            (generated_model, idx, max_new_tokens, context_size)
        )
        return [max_new_tokens]

    monkeypatch.setattr(
        classifier_model,
        "text_to_token_ids",
        lambda prompt, selected_tokenizer: prompt,
    )
    monkeypatch.setattr(
        classifier_model,
        "generate_text_simple",
        fake_generate,
    )
    monkeypatch.setattr(
        classifier_model,
        "token_ids_to_text",
        lambda token_ids, selected_tokenizer: f"decoded-{token_ids[0]}",
    )

    generated_texts = classifier_model.run_pretrained_sanity_checks(
        model,
        tokenizer,
    )

    assert [call[1] for call in generated_calls] == [
        "Every effort moves you",
        classifier_model.CLASSIFICATION_PROMPT,
    ]
    assert [call[2] for call in generated_calls] == [15, 23]
    assert all(call[0] is model for call in generated_calls)
    assert all(
        call[3] == classifier_model.BASE_CONFIG["context_length"]
        for call in generated_calls
    )
    assert generated_texts == ("decoded-15", "decoded-23")
    output = capsys.readouterr().out
    assert "decoded-15" in output
    assert "decoded-23" in output


@pytest.fixture
def tiny_config():
    return {
        "vocab_size": 32,
        "context_length": 8,
        "embedding_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


def test_classification_head_has_expected_shape_and_trainable_parameters(
    tiny_config,
):
    model = classifier_model.add_classification_head(GPTModel(tiny_config))

    assert model.out_head.in_features == tiny_config["embedding_dim"]
    assert model.out_head.out_features == 2
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_names = {
        name for name, _ in model.named_parameters()
        if name.startswith(
            (
                "transformer_blocks.1.",
                "final_layer_norm.",
                "out_head.",
            )
        )
    }
    assert trainable_names == expected_names


def test_classifier_forward_returns_two_logits_per_position(tiny_config):
    model = classifier_model.add_classification_head(GPTModel(tiny_config))
    input_ids = torch.randint(
        0,
        tiny_config["vocab_size"],
        (3, 5),
    )

    logits = model(input_ids)

    assert logits.shape == (3, 5, 2)


def test_optimizer_step_changes_only_trainable_parameters(tiny_config):
    model = classifier_model.add_classification_head(GPTModel(tiny_config))
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    input_ids = torch.randint(
        0,
        tiny_config["vocab_size"],
        (2, 5),
    )
    targets = torch.tensor([0, 1])

    optimizer.zero_grad()
    loss = F.cross_entropy(model(input_ids)[:, -1, :], targets)
    loss.backward()
    optimizer.step()

    current_parameters = dict(model.named_parameters())
    assert all(
        torch.equal(current_parameters[name], original)
        for name, original in frozen_before.items()
    )
    assert any(
        not torch.equal(current_parameters[name], original)
        for name, original in trainable_before.items()
    )


def test_all_parameters_follow_model_to_selected_device(tiny_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = classifier_model.add_classification_head(GPTModel(tiny_config))

    selected_device = move_model_to_device(model, device)

    assert selected_device == device
    assert all(parameter.device == device for parameter in model.parameters())
    assert all(buffer.device == device for buffer in model.buffers())


@pytest.mark.parametrize("trainable_last_blocks", [0, -1, 3])
def test_rejects_invalid_trainable_last_block_count(
    tiny_config,
    trainable_last_blocks,
):
    with pytest.raises(ValueError, match="between 1 and 2"):
        classifier_model.add_classification_head(
            GPTModel(tiny_config),
            trainable_last_blocks=trainable_last_blocks,
        )
