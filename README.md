# Neural Network Fundamentals from Scratch

This existing project now covers two stages:

1. Neural-network building blocks
2. How networks learn: forward pass, loss, and backpropagation intuition

## Concepts Covered

- Inputs
- Weights
- Bias
- Weighted sums
- ReLU
- Sigmoid
- Single neuron
- Dense layer
- Forward pass
- Prediction
- Target
- Mean Squared Error
- Binary Cross Entropy
- Gradient intuition
- Numerical gradients
- Backpropagation intuition
- Learning rate
- Gradient descent
- Weight updates
- Learning rate
- Epochs
- Training vs inference

## Learning Cycle

```text
Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Gradient / Backpropagation
  ↓
Weight Update
  ↓
Repeat
```

## Run

```powershell
python main.py
python backprop_intuition.py
python training_demo.py
python plot_activations.py
```

## Important Note

The backpropagation demo uses numerical gradients to build intuition.
It is not yet a full analytical multi-layer backpropagation engine.

## Next Step

A future extension can add:
- analytical derivatives
- chain rule
- full backpropagation
- multi-layer training
- mini-batches
- train/test datasets
