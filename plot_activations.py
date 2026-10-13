import numpy as np
import matplotlib.pyplot as plt
from activations import relu, sigmoid

x = np.linspace(-10, 10, 500)

plt.figure(figsize=(8, 5))
plt.plot(x, relu(x))
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("relu_plot.png", dpi=180)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("sigmoid_plot.png", dpi=180)
plt.show()
