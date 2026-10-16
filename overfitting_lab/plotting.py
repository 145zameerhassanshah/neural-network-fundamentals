from pathlib import Path
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, best_epoch, title, output_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    if best_epoch > 0:
        plt.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.2,
            label=f"Best Validation Epoch = {best_epoch}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
