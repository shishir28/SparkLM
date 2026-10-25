import pandas as pd
import pytest
import torch

from sparklm.classification.classifier_model import add_classification_head
from sparklm.classification.inference import classify_review
from sparklm.classification.spam_dataset import SpamDataset
from sparklm.model import GPTModel


PAD_TOKEN_ID = 50256
TINY_CONFIG = {
    "vocab_size": 50257,
    "context_length": 8,
    "embedding_dim": 16,
    "num_layers": 2,
    "num_heads": 2,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


class CharTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]


def make_classifier():
    torch.manual_seed(0)
    return add_classification_head(GPTModel(TINY_CONFIG))


def force_prediction(model, label):
    with torch.no_grad():
        model.out_head.weight.zero_()
        bias = [1.0, 0.0] if label == 0 else [0.0, 1.0]
        model.out_head.bias.copy_(torch.tensor(bias))
    return model


def capture_model_inputs(model):
    captured = []
    model.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    return captured


def dataset_ids(tmp_path, text, max_length):
    csv_path = tmp_path / "rows.csv"
    pd.DataFrame({"Label": [0], "Text": [text]}).to_csv(csv_path, index=False)
    dataset = SpamDataset(csv_path, CharTokenizer(), max_length=max_length)
    return dataset[0][0]


@pytest.mark.parametrize(("label", "expected"), [(0, "not spam"), (1, "spam")])
def test_maps_predicted_label_to_text(label, expected):
    model = force_prediction(make_classifier(), label)
    prediction = classify_review("hello", model, CharTokenizer(), "cpu", max_length=6)

    assert prediction == expected


@pytest.mark.parametrize(
    "text",
    ["hi", "exactly", "this text is longer than the context"],
)
def test_input_ids_match_spam_dataset_preprocessing(tmp_path, text):
    model = make_classifier()
    captured = capture_model_inputs(model)

    classify_review(text, model, CharTokenizer(), "cpu", max_length=7)

    assert torch.equal(captured[0], dataset_ids(tmp_path, text, 7).unsqueeze(0))


def test_short_text_is_right_padded_with_pad_token():
    model = make_classifier()
    captured = capture_model_inputs(model)

    classify_review("ab", model, CharTokenizer(), "cpu", max_length=5)

    assert captured[0].tolist() == [[97, 98, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]]


def test_long_text_is_truncated_to_max_length():
    model = make_classifier()
    captured = capture_model_inputs(model)

    classify_review("abcdefgh", model, CharTokenizer(), "cpu", max_length=3)

    assert captured[0].tolist() == [[97, 98, 99]]


def test_missing_max_length_raises():
    with pytest.raises(ValueError, match="max_length must be specified"):
        classify_review("hi", make_classifier(), CharTokenizer(), "cpu")


def test_max_length_beyond_context_raises():
    too_long = TINY_CONFIG["context_length"] + 1

    with pytest.raises(ValueError, match="exceeds model's context length"):
        classify_review(
            "hi", make_classifier(), CharTokenizer(), "cpu", max_length=too_long
        )


def test_switches_model_to_eval_mode():
    model = make_classifier()
    model.train()

    classify_review("hi", model, CharTokenizer(), "cpu", max_length=4)

    assert not model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_moves_input_to_model_device():
    model = force_prediction(make_classifier(), 1).to("cuda")

    assert classify_review("hi", model, CharTokenizer(), "cuda", max_length=4) == "spam"


def test_save_and_reload_preserves_logits_and_predictions(tmp_path):
    model = make_classifier()
    with torch.no_grad():
        model.out_head.bias.copy_(torch.tensor([0.0, 0.5]))
    checkpoint_path = tmp_path / "review_classifier.pth"
    torch.save(model.state_dict(), checkpoint_path)

    reloaded = add_classification_head(GPTModel(TINY_CONFIG))
    reloaded.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    input_ids = torch.tensor([[1, 2, 3, PAD_TOKEN_ID]])
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(model(input_ids), reloaded(input_ids))
    for text in ("hi", "free prize now"):
        assert classify_review(
            text, reloaded, CharTokenizer(), "cpu", max_length=6
        ) == classify_review(text, model, CharTokenizer(), "cpu", max_length=6)
