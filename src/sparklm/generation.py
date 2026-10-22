import torch
from torch import Tensor

def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        with torch.no_grad(): # I only want to use the model. Do not prepare for backpropagation
            logits = model(idx_cond)
            logits = logits[:, -1, :] #Focuses only on the last time step,so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
            probs = torch.softmax(logits, dim=-1) #probas has shape (batch, vocab_size)
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token_id), dim=1) # cOncatenate the new token to the existing sequence
    return idx

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # Focus on the last time step

        # New: Filter logits with top_k sampling
        if top_k is not None:
            top_logits, top_indices = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(float('-inf')).to(logits.device), logits)

        # apply temperature scaling if temperature > 0.0
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        # otherwise, if temperature is 0.0, use greedy selection
        elif temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy selection
        else:
            raise ValueError("Temperature must be greater than or equal to 0.0.")

        # Check for EOS token and break if found
        if eos_id is not None and (idx_next == eos_id).all():
            break

        # Concatenate the new token to the existing sequence
        idx = torch.cat((idx, idx_next), dim=1)  # Concatenate the new token to the existing sequence
    return idx

def softmax_with_temperature(logits, temperature):
    if(temperature <= 0.0):
        raise ValueError("Temperature must be greater than 0.0.")

    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)
