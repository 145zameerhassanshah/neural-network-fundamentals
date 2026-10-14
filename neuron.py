import numpy as np

class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.activation = activation

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        z = np.dot(inputs, self.weights) + self.bias
        output = self.activation(z)
        return {
            "inputs": inputs,
            "weights": self.weights,
            "bias": self.bias,
            "weighted_sum": z,
            "output": output,
        }
