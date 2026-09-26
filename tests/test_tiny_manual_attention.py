import torch

from sparklm.attention.tiny_manual import tiny_manual_attention_example


def test_tiny_manual_outputs_have_expected_keys_and_shapes():
    out = tiny_manual_attention_example()

    for key in (
        "Q",
        "K",
        "V",
        "scores_loop",
        "weights_loop",
        "context_loop",
        "scores_mat",
        "weights_mat",
        "context_mat",
    ):
        assert key in out

    assert out["scores_loop"].shape == out["scores_mat"].shape
    assert out["weights_loop"].shape == out["weights_mat"].shape
    assert out["context_loop"].shape == out["context_mat"].shape



def test_loop_path_matches_row_normalized_scores_and_context_formula():
    out = tiny_manual_attention_example()

    scores_loop = out["scores_loop"]
    weights_loop = out["weights_loop"]
    v = out["V"]

    expected_weights_loop = scores_loop / scores_loop.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights_loop, expected_weights_loop, atol=1e-6)

    expected_context_loop = weights_loop @ v
    assert torch.allclose(out["context_loop"], expected_context_loop, atol=1e-6)



def test_matrix_path_matches_softmax_scores_and_context_formula():
    out = tiny_manual_attention_example()

    scores_mat = out["scores_mat"]
    weights_mat = out["weights_mat"]
    v = out["V"]

    expected_weights_mat = torch.softmax(scores_mat, dim=-1)
    assert torch.allclose(weights_mat, expected_weights_mat, atol=1e-6)

    expected_context_mat = weights_mat @ v
    assert torch.allclose(out["context_mat"], expected_context_mat, atol=1e-6)



def test_custom_inputs_are_passed_through_in_output_dict():
    q = torch.tensor([2.0, 0.0])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    out = tiny_manual_attention_example(q=q, k=k, v=v)

    assert torch.equal(out["Q"], q)
    assert torch.equal(out["K"], k)
    assert torch.equal(out["V"], v)



def test_weights_are_finite_and_rows_sum_to_one():
    out = tiny_manual_attention_example()
    w_loop = out["weights_loop"]
    w_mat = out["weights_mat"]

    assert torch.isfinite(w_loop).all()
    assert torch.isfinite(w_mat).all()

    assert torch.allclose(w_loop.sum(dim=-1), torch.ones_like(w_loop.sum(dim=-1)), atol=1e-6)
    assert torch.allclose(w_mat.sum(dim=-1), torch.ones_like(w_mat.sum(dim=-1)), atol=1e-6)
