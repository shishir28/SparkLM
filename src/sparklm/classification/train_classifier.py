import torch

from sparklm.classification.metrics import (
    calc_accuracy_loader,
    calc_loss_batch,
    calc_loss_loader,
)


def train_classifier_simple(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.size(0)
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Step {global_step}: Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}"
                )

        train_acc = calc_accuracy_loader(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_acc = calc_accuracy_loader(
            val_dataloader, model, device, num_batches=eval_iter
        )
        model.train()

        print(
            f"Epoch {epoch + 1}: Train Accuracy = {train_acc:.2f}, "
            f"Val Accuracy = {val_acc:.2f}"
        )
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
