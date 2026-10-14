import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))
