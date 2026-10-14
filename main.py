import numpy as np
from tiny_network import TinyNetwork
from loss import binary_cross_entropy

network = TinyNetwork(seed=42)
x = np.array([0.0, 1.0])
target = 1.0

prediction, cache = network.forward(x)
loss = binary_cross_entropy(prediction, target)

print("HOW NETWORKS LEARN: FORWARD PASS + BCE")
print("Input:", x)
print("Hidden weighted sums:", cache["z1"])
print("Hidden activations:", cache["a1"])
print("Output weighted sum:", cache["z2"])
print("Prediction:", prediction)
print("Target:", target)
print("Binary Cross-Entropy loss:", loss)
