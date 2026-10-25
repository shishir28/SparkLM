import sys
from types import ModuleType

from scripts import finetune_spam


def test_base_config_matches_pretrained_gpt2_small():
    assert finetune_spam.CHOOSE_MODEL == "gpt2-small (124M)"
    assert finetune_spam.BASE_CONFIG == {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "embedding_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
    }


def test_initialize_pretrained_gpt2_loads_and_runs_both_sanity_checks(
    monkeypatch,
    capsys,
):
    events = []
    generated_calls = []
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

    def fake_generate(
        model,
        idx,
        max_new_tokens,
        context_size,
    ):
        generated_calls.append(
            (model, idx, max_new_tokens, context_size)
        )
        return [max_new_tokens]

    fake_download_module = ModuleType("sparklm.gpt_download")
    fake_download_module.download_and_load_gpt2 = fake_download
    monkeypatch.setitem(
        sys.modules,
        "sparklm.gpt_download",
        fake_download_module,
    )
    monkeypatch.setattr(finetune_spam, "GPTModel", FakeModel)
    monkeypatch.setattr(finetune_spam, "load_weights_into_gpt", fake_load)
    monkeypatch.setattr(
        finetune_spam,
        "text_to_token_ids",
        lambda prompt, tokenizer: prompt,
    )
    monkeypatch.setattr(finetune_spam, "generate_text_simple", fake_generate)
    monkeypatch.setattr(
        finetune_spam,
        "token_ids_to_text",
        lambda token_ids, tokenizer: f"decoded-{token_ids[0]}",
    )

    tokenizer = object()
    model = finetune_spam.initialize_pretrained_gpt2(tokenizer)

    assert events[0] == (
        "download",
        "124M",
        finetune_spam.GPT2_MODELS_DIR,
    )
    assert events[1] == ("build", finetune_spam.BASE_CONFIG)
    assert events[2] == ("load", model, params)
    assert events[3] == ("eval", model)
    assert [call[1] for call in generated_calls] == [
        "Every effort moves you",
        finetune_spam.CLASSIFICATION_PROMPT,
    ]
    assert [call[2] for call in generated_calls] == [15, 23]
    assert all(
        call[3] == finetune_spam.BASE_CONFIG["context_length"]
        for call in generated_calls
    )
    output = capsys.readouterr().out
    assert "decoded-15" in output
    assert "decoded-23" in output
