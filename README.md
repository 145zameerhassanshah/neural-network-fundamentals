# Neural Network Fundamentals from Scratch

## Current Task
How Networks Learn: Forward Pass, Loss and Backpropagation Intuition

## Requirements Covered
- Forward propagation
- Binary Cross-Entropy loss
- Backpropagation intuition
- Tiny network in NumPy
- Numerical gradient estimation
- Gradient-descent training loop
- Weight updates
- Epochs
- Learning-rate experiment

## Tiny Network
2 inputs -> 2 hidden neurons -> 1 sigmoid output

## Run
```powershell
python main.py
python train_tiny_network.py
python learning_rate_experiment.py
```

## Learning Loop
Input -> Forward Pass -> Prediction -> BCE Loss -> Gradients -> Weight Update -> Repeat

## Important Note
Numerical gradients are used here for intuition without heavy calculus. Real frameworks use analytical backpropagation/autodiff.
