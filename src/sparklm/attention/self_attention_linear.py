"""Self-attention with explicit W_q/W_k/W_v (placeholder).

This module should define a small class that holds explicit weight
matrices (W_q, W_k, W_v) and computes attention in the simple form used
in the book's step-by-step progression.
"""

import torch
from torch import Tensor

import torch.nn as nn

class SelfAttentionLinear_v1(nn.Module):
    """Explicit self-attention using parameter matrices W_q, W_k, W_v.

    - __init__(d_in, d_out=None)
    - forward(x: Tensor) -> Tensor
    """

    def __init__(self, d_in, d_out=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        self.W_query = nn.Parameter(torch.randn(d_in, self.d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, self.d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, self.d_out))

    def forward(self, x) -> Tensor:
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T
        attn_weights =  torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vectors = attn_weights @ values
        return context_vectors
    

class SelfAttentionLinear_v2(nn.Module):
    """Explicit self-attention using parameter matrices W_q, W_k, W_v.

    - __init__(d_in, d_out=None)
    - forward(x: Tensor) -> Tensor
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        self.W_query = nn.Linear(d_in, self.d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, self.d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, self.d_out, bias=qkv_bias)

    def forward(self, x) -> Tensor:
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights =  torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vectors = attn_weights @ values
        return context_vectors