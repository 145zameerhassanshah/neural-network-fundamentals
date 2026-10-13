# Neural Network Fundamentals Lab

A from-scratch NumPy implementation of a single neuron, a small dense layer, and ReLU/Sigmoid plots.

## Requirements Covered
- Inputs
- Weights
- Bias
- Weighted sum
- ReLU
- Sigmoid
- Output
- Single neuron in NumPy
- Small layer in NumPy
- ReLU and Sigmoid plots

## Core Formula

Single neuron:

z = x · w + b

output = activation(z)

Small dense layer:

Z = X @ W.T + b

A = activation(Z)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
python plot_activations.py
```

## Limitations

This task covers only forward propagation. It does not yet implement loss functions, backpropagation, gradient descent, or training.
