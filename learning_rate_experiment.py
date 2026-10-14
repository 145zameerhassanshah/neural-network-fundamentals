from train_tiny_network import train

learning_rates = [0.1, 0.5, 1.0, 2.0]
epochs = 1500

print("=" * 72)
print("LEARNING RATE EXPERIMENT")
print("=" * 72)

results = []

for lr in learning_rates:
    _, history = train(
        learning_rate=lr,
        epochs=epochs,
        verbose=False,
    )

    results.append(
        (
            lr,
            history[0],
            history[-1],
        )
    )

for lr, initial_loss, final_loss in results:
    print(
        f"LR={lr:<3} | "
        f"Initial Loss={initial_loss:.6f} | "
        f"Final Loss={final_loss:.6f}"
    )

print(
    "\nLearning point: smaller learning rates make slower progress; "
    "larger rates can learn faster, but excessively large rates can "
    "overshoot or become unstable."
)
