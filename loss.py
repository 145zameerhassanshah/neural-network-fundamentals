import numpy as np

def mse_loss(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.mean((prediction - target) ** 2)

def binary_cross_entropy(prediction, target, epsilon=1e-8):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    return -np.mean(target*np.log(prediction) + (1-target)*np.log(1-prediction))
