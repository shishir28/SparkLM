from unittest.mock import patch

import pytest
import torch
from torch import nn

from sparklm.generation import generate, generate_text_simple


class IncrementingModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_inputs = []
        self.grad_states = []

    def forward(self, token_ids):
        self.seen_inputs.append(token_ids.clone())
        self.grad_states.append(torch.is_grad_enabled())
        next_ids = (token_ids + 1) % self.vocab_size
        logits = torch.zeros(*token_ids.shape, self.vocab_size)
        return logits.scatter(2, next_ids.unsqueeze(-1), 1.0)


def test_generation_preserves_prompt_and_appends_greedy_tokens():
    model = IncrementingModel(vocab_size=7)
    prompt = torch.tensor([[1, 3], [2, 4]])

    generated = generate_text_simple(
        model, prompt, max_new_tokens=3, context_size=4
    )

    expected = torch.tensor([[1, 3, 4, 5, 6], [2, 4, 5, 6, 0]])
    assert torch.equal(generated, expected)
    assert torch.equal(generated[:, : prompt.shape[1]], prompt)


def test_generation_crops_each_model_input_to_context_size():
    model = IncrementingModel(vocab_size=10)
    prompt = torch.tensor([[1, 2, 3, 4]])

    generate_text_simple(model, prompt, max_new_tokens=2, context_size=2)

    assert torch.equal(model.seen_inputs[0], torch.tensor([[3, 4]]))
    assert torch.equal(model.seen_inputs[1], torch.tensor([[4, 5]]))


def test_generation_disables_gradient_tracking():
    model = IncrementingModel(vocab_size=7)

    generate_text_simple(
        model, torch.tensor([[1, 2]]), max_new_tokens=2, context_size=4
    )

    assert model.grad_states == [False, False]


def test_zero_new_tokens_returns_unchanged_prompt():
    model = IncrementingModel(vocab_size=7)
    prompt = torch.tensor([[1, 2, 3]])

    generated = generate_text_simple(
        model, prompt, max_new_tokens=0, context_size=3
    )

    assert torch.equal(generated, prompt)
    assert model.seen_inputs == []


def test_generate_temperature_zero_matches_greedy_reference():
    model = IncrementingModel(vocab_size=7)
    prompt = torch.tensor([[1, 3]])

    expected = generate_text_simple(model, prompt.clone(), 3, context_size=4)
    actual = generate(model, prompt.clone(), 3, context_size=4, temperature=0.0)

    assert torch.equal(actual, expected)


def test_generate_sampling_is_repeatable_with_same_seed():
    class FixedLogitModel(nn.Module):
        def forward(self, token_ids):
            logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
            return logits.expand(*token_ids.shape, -1)

    model = FixedLogitModel()
    prompt = torch.tensor([[0]])

    torch.manual_seed(123)
    first = generate(model, prompt.clone(), 5, context_size=4, temperature=1.0)
    torch.manual_seed(123)
    second = generate(model, prompt.clone(), 5, context_size=4, temperature=1.0)

    assert torch.equal(first, second)


def test_generate_top_k_keeps_k_finite_logits():
    class FixedLogitModel(nn.Module):
        def forward(self, token_ids):
            logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
            return logits.expand(*token_ids.shape, -1)

    with (
        patch("sparklm.generation.torch.multinomial", return_value=torch.tensor([[0]])),
        patch("sparklm.generation.torch.softmax", wraps=torch.softmax) as softmax,
    ):
        generate(
            FixedLogitModel(),
            torch.tensor([[0]]),
            1,
            context_size=4,
            temperature=1.0,
            top_k=2,
        )

    filtered_logits = softmax.call_args.args[0]
    assert torch.isfinite(filtered_logits).sum().item() == 2


def test_generate_rejects_negative_temperature():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        generate(IncrementingModel(vocab_size=7), torch.tensor([[1]]), 1, 4, temperature=-1.0)


def test_generate_stops_before_appending_eos_token():
    class EosModel(nn.Module):
        def forward(self, token_ids):
            logits = torch.zeros(*token_ids.shape, 3)
            logits[..., 2] = 1.0
            return logits

    prompt = torch.tensor([[1]])
    generated = generate(EosModel(), prompt, 5, context_size=4, eos_id=2)

    assert torch.equal(generated, prompt)
