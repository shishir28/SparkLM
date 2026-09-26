import torch
from torch import Tensor

import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """Causal self-attention placeholder.

    Manual implementation target:
    - __init__(d_in, d_out=None, max_seq_len=1024, qkv_bias=False, attention_dropout=0.0)
    - forward(x) -> Tensor
    """

    def __init__(self, d_in, d_out=None, max_seq_len=1024, qkv_bias=False, attention_dropout=0.0):
        #qkv_bias: whether Q/K/V linear projections have bias.
        #attention_dropout: dropout probability applied to attention weights.
        super().__init__()
        self.d_in = d_in #input embedding dimension.
        self.d_out = d_out if d_out is not None else d_in #projected Q/K/V dimension. If None, default to d_in.
        self.max_seq_len = max_seq_len
        self.W_query = nn.Linear(d_in, self.d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, self.d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, self.d_out, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.register_buffer('causal_mask', torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)) # New
        

    def forward(self, x, attn_mask=None):
        b,num_tokens,d_in = x.shape #New batch dimension support
        if num_tokens > self.max_seq_len:
            raise ValueError(f"Sequence length {num_tokens} exceeds maximum {self.max_seq_len}")
        keys = self.W_key(x) # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values = self.W_value(x) # (b, num_tokens, d_out)
        attn_scores = queries @ keys.transpose(1, 2) # (b, num_tokens, num_tokens)
        
        attn_scores = attn_scores.masked_fill(self.causal_mask[:num_tokens, :num_tokens],-torch.inf) # (b, num_tokens, num_tokens)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) 
        attn_weights = self.attention_dropout(attn_weights)
        context_vectors = attn_weights @ values # (b, num_tokens, d_out)
        return context_vectors