import torch

from sparklm.generation import generate_text_simple
from sparklm.loss import calc_loss_batch, calc_loss_loader
from sparklm.text_utils import text_to_token_ids ,token_ids_to_text

def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for evaluation
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Set the model back to training mode
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # Set the model to evaluation mode
    context_size = model.pos_embedding.num_embeddings
    token_ids = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model,
            token_ids,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Generated text: {decoded_text}")
    model.train()  # Set the model back to training mode


def train_model_simple(model, train_loader, val_loader,
                      optimizer, device, num_epochs,
                      eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            tokens_seen += input_batch.numel()  # Update the count of tokens seen
            global_step += 1

            if(global_step % eval_freq == 0):
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Step {global_step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
