import numpy as np

from loss import mse_loss
from backprop_intuition import (
    forward_single_neuron,
    numerical_gradient_for_weight,
)

def run_learning_cycle():
    print("=" * 65)
    print("FORWARD PASS -> LOSS -> GRADIENT -> UPDATE")
    print("=" * 65)

    inputs = np.array([2.0, 1.5, -0.5])
    weights = np.array([0.8, -0.4, 0.6], dtype=float)
    bias = 0.2
    target = 1.0
    learning_rate = 0.1

    for epoch in range(1, 9):
        _, prediction = forward_single_neuron(inputs, weights, bias)
        loss = mse_loss(prediction, target)

        gradients = np.zeros_like(weights)
        for i in range(len(weights)):
            gradients[i] = numerical_gradient_for_weight(
                inputs, weights, bias, target, i
            )

        weights -= learning_rate * gradients

        print(
            f"Epoch {epoch:02d} | "
            f"Prediction: {prediction:.6f} | "
            f"Loss: {loss:.6f}"
        )

if __name__ == "__main__":
    run_learning_cycle()
