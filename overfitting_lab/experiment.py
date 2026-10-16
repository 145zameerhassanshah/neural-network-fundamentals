import torch

try:
    from .data import make_dataset
    from .models import LargeNet, SmallNet
    from .training import train_model
    from .plotting import plot_losses
except ImportError:
    from data import make_dataset
    from models import LargeNet, SmallNet
    from training import train_model
    from plotting import plot_losses

def run_experiment(name, model, x_train, y_train, x_val, y_val, **kwargs):
    print("\n" + "=" * 78)
    print(name.upper())
    print("=" * 78)

    result = train_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        **kwargs,
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )

    plot_losses(
        result["train_losses"],
        result["val_losses"],
        result["best_epoch"],
        name,
        f"overfitting_lab/plots/{safe_name}.png",
    )

    print(f"Epochs ran:            {result['epochs_ran']}")
    print(f"Best validation epoch: {result['best_epoch']}")
    print(f"Best validation loss:  {result['best_val_loss']:.6f}")
    print(f"Final training loss:   {result['final_train_loss']:.6f}")
    print(f"Final validation loss: {result['final_val_loss']:.6f}")
    print(f"Final train-val gap:   {result['final_gap']:.6f}")

    return result

def main():
    torch.manual_seed(7)

    x_train, y_train, x_val, y_val = make_dataset(
        seed=7,
        n_train=80,
        n_val=300,
        noise=0.22,
    )

    results = {}

    results["Baseline"] = run_experiment(
        "Baseline Large Network",
        LargeNet(dropout=0.0),
        x_train, y_train, x_val, y_val,
        epochs=700,
        learning_rate=0.01,
    )

    torch.manual_seed(7)
    results["Early Stopping"] = run_experiment(
        "Early Stopping",
        LargeNet(dropout=0.0),
        x_train, y_train, x_val, y_val,
        epochs=700,
        learning_rate=0.01,
        early_stopping=True,
        patience=40,
    )

    torch.manual_seed(7)
    results["Dropout"] = run_experiment(
        "Dropout",
        LargeNet(dropout=0.40),
        x_train, y_train, x_val, y_val,
        epochs=700,
        learning_rate=0.01,
    )

    torch.manual_seed(7)
    results["Smaller Network"] = run_experiment(
        "Smaller Network",
        SmallNet(),
        x_train, y_train, x_val, y_val,
        epochs=700,
        learning_rate=0.01,
    )

    torch.manual_seed(7)
    results["L2"] = run_experiment(
        "L2 Regularization",
        LargeNet(dropout=0.0),
        x_train, y_train, x_val, y_val,
        epochs=700,
        learning_rate=0.01,
        weight_decay=1e-3,
    )

    print("\n" + "=" * 96)
    print("COMPARISON SUMMARY")
    print("=" * 96)
    print(
        f"{'Experiment':<20}"
        f"{'Best Epoch':>12}"
        f"{'Best Val':>14}"
        f"{'Final Train':>16}"
        f"{'Final Val':>14}"
        f"{'Gap':>12}"
    )

    for name, result in results.items():
        print(
            f"{name:<20}"
            f"{result['best_epoch']:>12}"
            f"{result['best_val_loss']:>14.6f}"
            f"{result['final_train_loss']:>16.6f}"
            f"{result['final_val_loss']:>14.6f}"
            f"{result['final_gap']:>12.6f}"
        )

    print(
        "\nOpen overfitting_lab/plots/ to inspect each loss curve."
    )

if __name__ == "__main__":
    main()
