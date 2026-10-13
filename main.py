import numpy as np
from activations import relu, sigmoid
from neuron import Neuron
from layer import DenseLayer

inputs = np.array([2.0, 1.5, -0.5])
neuron = Neuron([0.8, -0.4, 0.6], 0.2, sigmoid)
z, output = neuron.forward(inputs)

print("SINGLE NEURON")
print("Inputs:", inputs)
print("Weighted sum:", z)
print("Sigmoid output:", output)

layer = DenseLayer(
    weights=[
        [0.2, 0.8, -0.5],
        [0.5, -0.3, 0.7],
        [-0.6, 0.4, 0.9],
    ],
    biases=[0.1, -0.2, 0.3],
    activation=relu,
)

z_layer, outputs = layer.forward([1.0, 2.0, -1.0])

print("\nSMALL LAYER")
print("Weighted sums:", z_layer)
print("ReLU outputs:", outputs)
