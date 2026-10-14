import numpy as np

from tiny_network import TinyNetwork
from numerical_gradients import numerical_gradients
from loss import binary_cross_entropy

# XOR dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = np.array([0.0, 1.0, 1.0, 0.0])


def dataset_loss(network):
    losses = []

    for features, target in zip(X, y):
        prediction, _ = network.forward(features)
        losses.append(binary_cross_entropy(prediction, target))

    return float(np.mean(losses))


def train(learning_rate=1.0, epochs=3000, verbose=True):
    network = TinyNetwork(seed=0)
    history = []

    for epoch in range(1, epochs + 1):
        total_grads = {
            "w1": np.zeros_like(network.w1),
            "b1": np.zeros_like(network.b1),
            "w2": np.zeros_like(network.w2),
            "b2": 0.0,
        }

        # Full-batch numerical gradients
        for features, target in zip(X, y):
            grads = numerical_gradients(
                network,
                features,
                target,
            )

            total_grads["w1"] += grads["w1"]
            total_grads["b1"] += grads["b1"]
            total_grads["w2"] += grads["w2"]
            total_grads["b2"] += grads["b2"]

        n = len(X)

        # Gradient-descent update
        network.w1 -= learning_rate * total_grads["w1"] / n
        network.b1 -= learning_rate * total_grads["b1"] / n
        network.w2 -= learning_rate * total_grads["w2"] / n
        network.b2 -= learning_rate * total_grads["b2"] / n

        loss = dataset_loss(network)
        history.append(loss)

        if verbose and (
            epoch == 1
            or epoch % 300 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:04d} | "
                f"BCE Loss: {loss:.6f}"
            )

    if verbose:
        print("\nFinal predictions:")

        for features, target in zip(X, y):
            prediction, _ = network.forward(features)

            predicted_class = int(prediction >= 0.5)

            print(
                f"Input {features} | "
                f"Target {int(target)} | "
                f"Probability {prediction:.4f} | "
                f"Class {predicted_class}"
            )

    return network, history


if __name__ == "__main__":
    train(
        learning_rate=1.0,
        epochs=3000,
    )
