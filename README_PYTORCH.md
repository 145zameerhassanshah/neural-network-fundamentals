# Introduction to PyTorch — XOR Network

Recreates the previous NumPy XOR network using PyTorch.

## Requirements Covered
- tensors
- nn.Module
- nn.Linear layers
- forward()
- activation functions
- BCEWithLogitsLoss
- optimizer selection
- training loop
- zero_grad()
- backward()
- optimizer.step()
- model.eval()
- torch.no_grad()
- XOR network recreation
- learning-rate experiment

## Architecture
2 inputs -> 2 hidden neurons -> 1 output

## Install
```powershell
python -m pip install torch
python -c "import torch; print(torch.__version__)"
```

## Run
```powershell
python pytorch_xor.py
python pytorch_learning_rate_experiment.py
```

## Core Training Loop
```python
logits = model(X)
loss = criterion(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Why BCEWithLogitsLoss?
It combines sigmoid + binary cross-entropy in a numerically stable way.

During inference:
```python
probabilities = torch.sigmoid(logits)
```

## NumPy vs PyTorch
NumPy version:
- manual matrix operations
- manual/numerical gradients
- manual parameter updates

PyTorch version:
- reusable layers
- automatic differentiation
- built-in optimizers
- cleaner training workflow
