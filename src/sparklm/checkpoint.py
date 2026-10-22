"""Save and restore SparkLM training checkpoints."""

import torch


def save_checkpoint(path, model, optimizer, config, metadata):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["metadata"]
