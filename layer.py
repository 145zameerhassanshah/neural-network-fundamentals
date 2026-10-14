import numpy as np

class DenseLayer:
    def __init__(self, weights, biases, activation):
        self.weights = np.asarray(weights, dtype=float)
        self.biases = np.asarray(biases, dtype=float)
        self.activation = activation

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        z = np.dot(inputs, self.weights.T) + self.biases
        output = self.activation(z)
        return {
            "weighted_sums": z,
            "outputs": output,
        }
