import torch


def classify_review(
    text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    if max_length is None:
        raise ValueError("max_length must be specified.")

    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_embedding.weight.shape[0]

    if max_length > supported_context_length:
        raise ValueError(
            f"max_length {max_length} exceeds model's context length "
            f"{supported_context_length}."
        )

    input_ids = input_ids[:max_length]
    input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
