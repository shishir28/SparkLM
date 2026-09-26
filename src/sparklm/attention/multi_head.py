"""Multi-head attention scaffolds and implementation notes.

Chapter mapping:
- Raschka ch03.ipynb section 3.6: "Extending single-head attention to
  multi-head attention".

This file contains two learning implementations:

1. MultiHeadAttentionStacked
   - Book-first/simple version.
   - Creates multiple independent causal attention heads.
   - Runs each head separately.
   - Concatenates head outputs.
   - Optionally applies a final output projection.

2. MultiHeadAttentionCombined
   - More compact/efficient version.
   - Uses one combined projection for Q/K/V.
   - Reshapes tensors into multiple heads.
   - Computes all heads in parallel.
   - Recombines heads and applies output projection.

Both implementations should preserve input/output shape:
- input x:  (batch_size, num_tokens, d_model)
- output:   (batch_size, num_tokens, d_model)

Do not use torch.nn.MultiheadAttention or torch.nn.Transformer here.
The goal is to keep tensor transformations explicit and inspectable.
"""

import torch
from torch import Tensor

import torch.nn as nn

from sparklm.attention.causal_self_attention import CausalSelfAttention
class MultiHeadAttentionStacked(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalSelfAttention(d_in, d_out, context_length, qkv_bias=qkv_bias, attention_dropout=dropout)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_out * num_heads, d_out)
      

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(concat)
        return output


class MultiHeadAttentionCombined(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.attention_dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))
     

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        if num_tokens > self.causal_mask.shape[0]:
            raise ValueError(f"Sequence length {num_tokens} exceeds maximum {self.causal_mask.shape[0]}")
        keys  = self.W_key(x)
        queries = self.W_query(x)        
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b, num_tokens,self.num_heads,self.head_dim)
        values = values.view(b, num_tokens,self.num_heads,self.head_dim)
        
        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2) 
        
        attn_scores = queries @ keys.transpose(2, 3) 
        mask_bool = self.causal_mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_scores = attn_scores / (keys.shape[-1]**0.5)
        attn_weights = torch.softmax(attn_scores,    dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        context = (attn_weights @ values).transpose(1, 2)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        context = self.out_proj(context)
        return context