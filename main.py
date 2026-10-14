import numpy as np

from activations import relu, sigmoid
from neuron import Neuron
from layer import DenseLayer
from loss import mse_loss

print("SINGLE NEURON / FORWARD PASS")

inputs = np.array([2.0, 1.5, -0.5])
neuron = Neuron([0.8, -0.4, 0.6], 0.2, sigmoid)

result = neuron.forward(inputs)

print("Inputs:", result["inputs"])
print("Weighted sum:", result["weighted_sum"])
print("Prediction:", result["output"])

target = 1.0
loss = mse_loss(result["output"], target)

print("Target:", target)
print("MSE loss:", loss)

print("\nSMALL LAYER")

layer = DenseLayer(
    weights=[
        [0.2, 0.8, -0.5],
        [0.5, -0.3, 0.7],
        [-0.6, 0.4, 0.9],
    ],
    biases=[0.1, -0.2, 0.3],
    activation=relu,
)

layer_result = layer.forward([1.0, 2.0, -1.0])

print("Weighted sums:", layer_result["weighted_sums"])
print("ReLU outputs:", layer_result["outputs"])
