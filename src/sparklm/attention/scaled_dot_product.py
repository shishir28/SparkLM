import torch
from torch import Tensor

import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_in, d_out=None):
       super().__init__()

    def forward(self, q, k, v, mask=None, dropout=None):       
        if q.dim() < 2 or k.dim() < 2 or v.dim() < 2:
            raise ValueError("q, k, v must have the same number of dimensions and at least 2D")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k must have the same final dimension d_k")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError("k and v must have the same token/key length")      
        scores = q @ k.transpose(-2, -1) 
        attn_scores = scores / (q.shape[-1] ** 0.5)
        if mask is not None:
            if mask.shape != attn_scores.shape:
                raise ValueError("mask must be broadcastable to the attention score shape")
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        context = attn_weights @ v
        return context, attn_weights    