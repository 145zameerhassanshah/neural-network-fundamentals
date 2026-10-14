import numpy as np
from activations import sigmoid
from loss import binary_cross_entropy

class TinyNetwork:
    """
    Tiny XOR network:
        2 inputs -> 2 sigmoid hidden neurons -> 1 sigmoid output
    """

    def __init__(self, seed=0):
        # Seed/scale chosen to avoid the flat ~0.5 solution seen with seed=42,
        # scale=0.5 in this tiny educational XOR example.
        rng = np.random.default_rng(seed)

        self.w1 = rng.normal(0.0, 1.0, size=(2, 2))
        self.b1 = np.zeros(2)

        self.w2 = rng.normal(0.0, 1.0, size=2)
        self.b2 = 0.0

    def forward(self, x):
        x = np.asarray(x, dtype=float)

        z1 = x @ self.w1.T + self.b1
        a1 = sigmoid(z1)

        z2 = a1 @ self.w2 + self.b2
        y_hat = float(sigmoid(np.array([z2]))[0])

        return y_hat, {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "y_hat": y_hat,
        }

    def loss(self, x, y):
        prediction, _ = self.forward(x)
        return float(binary_cross_entropy(prediction, y))
