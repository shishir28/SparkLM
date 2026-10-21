import torch

def calc_loss_batch(input_batch,target_batch,model,device):
    """
    Calculate the loss for a batch of input and target sequences.

    Args:
        input_batch (torch.Tensor): Input tensor of shape (batch_size, seq_length).
        target_batch (torch.Tensor): Target tensor of shape (batch_size, seq_length).
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to run the computation on.

    Returns:
        torch.Tensor: The calculated loss for the batch.
    """
    # Move input and target batches to the specified device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass through the model
    logits = model(input_batch)

    logits = logits.flatten(0,1) # Reshape outputs and targets for loss calculation
    target_batch = target_batch.flatten()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target_batch)

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a data loader.

    Args:
        data_loader (DataLoader): DataLoader providing batches of input and target sequences.
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to run the computation on.
        num_batches (int, optional): Number of batches to evaluate. If None, evaluate all batches.

    Returns:
        float: The average loss over the evaluated batches.
    """
    total_loss = 0.0
    if(len(data_loader) == 0):
        return float('nan')  # Return NaN for an empty loader
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    average_loss = total_loss / num_batches

    return average_loss
