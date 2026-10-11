import torch
"""Tiny manual attention examples (placeholders).

Provide one or more small helper functions that compute attention by loop
and by matrix operations for tiny tensors. Implementations go here.
"""
def tiny_manual_attention_example(q=None, k=None, v=None, return_intermediates=True) -> dict:
    """Return a dictionary with example tensors and intermediate values.

    Expected to return keys: 'Q', 'K', 'V', 'scores_loop', 'weights_loop',
    'context_loop', 'scores_mat', 'weights_mat', 'context_mat'.
    """
    inputs = torch.tensor(
       [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
     )
    attn_scores_loops = torch.empty((inputs.shape[0], inputs.shape[0]))
    attn_weights_loops = torch.empty((inputs.shape[0], inputs.shape[0]))
    attn_context_loops = torch.empty((inputs.shape[0], inputs.shape[1]))
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores_loops[i, j] = torch.dot(x_i, x_j)
        row = attn_scores_loops[i]
        attn_weights_loops[i] = row / row.sum()
        attn_context_loops[i] = torch.zeros(inputs.shape[1])
        for j, x_j in enumerate(inputs):
            attn_context_loops[i] += attn_weights_loops[i, j] * x_j
    attn_scores_mat = inputs @ inputs.T
    attn_weights_mat = torch.softmax(attn_scores_mat, dim=-1)
    attn_context_mat = attn_weights_mat @ inputs
    return {
        "Q": q if q is not None else inputs,
        "K": k if k is not None else inputs,
        "V": v if v is not None else inputs,
        "scores_loop": attn_scores_loops,
        "weights_loop": attn_weights_loops,
        "context_loop": attn_context_loops,
        "scores_mat": attn_scores_mat,
        "weights_mat": attn_weights_mat,
        "context_mat": attn_context_mat,
    }
