from pathlib import Path

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, path):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Train Loss", color="blue")
    ax1.plot(epochs_seen, val_losses, label="Val Loss", color="red")
    ax1.set_xlabel("Epochs Seen")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0.0)
    ax2.set_xlabel("Tokens Seen")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
