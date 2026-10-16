import numpy as np
import torch

def make_dataset(seed=42, n_train=80, n_val=300, noise=0.22):
    rng = np.random.default_rng(seed)

    def build(n):
        x = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
        score = x[:, 0] ** 2 + x[:, 1] ** 2
        y = (score > 2.0).astype(np.float32)
        flips = rng.random(n) < noise
        y = np.where(flips, 1.0 - y, y).astype(np.float32)
        return x, y.reshape(-1, 1)

    x_train, y_train = build(n_train)
    x_val, y_val = build(n_val)

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
