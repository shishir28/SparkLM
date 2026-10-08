import torch
from torch import Tensor

def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token_id), dim=1)
    return idx