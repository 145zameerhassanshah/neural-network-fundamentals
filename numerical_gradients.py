import numpy as np

def numerical_gradients(network, x, y, epsilon=1e-5):
    grads = {
        "w1": np.zeros_like(network.w1),
        "b1": np.zeros_like(network.b1),
        "w2": np.zeros_like(network.w2),
        "b2": 0.0,
    }

    for i in range(network.w1.shape[0]):
        for j in range(network.w1.shape[1]):
            original = network.w1[i, j]
            network.w1[i, j] = original + epsilon
            lp = network.loss(x, y)
            network.w1[i, j] = original - epsilon
            lm = network.loss(x, y)
            network.w1[i, j] = original
            grads["w1"][i, j] = (lp - lm) / (2 * epsilon)

    for i in range(network.b1.shape[0]):
        original = network.b1[i]
        network.b1[i] = original + epsilon
        lp = network.loss(x, y)
        network.b1[i] = original - epsilon
        lm = network.loss(x, y)
        network.b1[i] = original
        grads["b1"][i] = (lp - lm) / (2 * epsilon)

    for i in range(network.w2.shape[0]):
        original = network.w2[i]
        network.w2[i] = original + epsilon
        lp = network.loss(x, y)
        network.w2[i] = original - epsilon
        lm = network.loss(x, y)
        network.w2[i] = original
        grads["w2"][i] = (lp - lm) / (2 * epsilon)

    original = network.b2
    network.b2 = original + epsilon
    lp = network.loss(x, y)
    network.b2 = original - epsilon
    lm = network.loss(x, y)
    network.b2 = original
    grads["b2"] = (lp - lm) / (2 * epsilon)

    return grads
