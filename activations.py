import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))
