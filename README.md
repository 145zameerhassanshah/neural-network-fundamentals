Neural Network Fundamentals from Scratch to PyTorch
A hands-on neural-network learning project that progresses from core neural-network mathematics in NumPy to training with PyTorch, and then to diagnosing overfitting through training/validation loss curves and regularization experiments.
The project demonstrates how neural networks move from manual calculations and gradient-based learning to framework-based training, automatic differentiation, model evaluation, and generalization analysis.
Learning Progression
Stage 1 — Neural Network Fundamentals
Implemented the basic building blocks of a neural network from scratch using NumPy:
- Inputs
- Weights
- Bias
- Weighted sums
- ReLU activation
- Sigmoid activation
- Single neuron
- Small dense layer
- Forward propagation
- Activation-function visualization
Stage 2 — How Networks Learn
Implemented the neural-network learning cycle conceptually using NumPy:
- Forward propagation
- Prediction
- Binary Cross-Entropy loss
- Numerical gradient estimation
- Backpropagation intuition
- Gradient descent
- Weight updates
- Learning rate
- Epochs
- Training vs inference
- XOR binary classification
- Learning-rate experiments
The NumPy XOR network successfully learned the nonlinear XOR pattern by repeatedly adjusting its weights to reduce prediction error.
Stage 3 — Introduction to PyTorch
Recreated the previous XOR neural network using PyTorch.
Concepts covered:
- PyTorch tensors
- torch.nn
- nn.Module
- nn.Linear
- forward()
- Sigmoid activation
- Logits
- BCEWithLogitsLoss
- torch.optim.SGD
- PyTorch Autograd
- optimizer.zero_grad()
- loss.backward()
- optimizer.step()
- Training loops
- model.eval()
- torch.no_grad()
- Model inference
The PyTorch XOR model successfully reduced BCE loss from approximately 0.748 to 0.0026 and correctly classified all four XOR combinations.
Stage 4 — Overfitting in Deep Networks and Reading Loss Curves
Extended the PyTorch work to study model generalization and overfitting.
The experiment used a small nonlinear binary-classification dataset and compared training loss with validation loss across multiple neural-network configurations.
Concepts covered:
- Training loss
- Validation loss
- Generalization
- Underfitting vs. overfitting
- Train-validation gap
- Loss curves
- Best validation epoch
- Divergence between training and validation behavior
- Early stopping
- Dropout
- Smaller network capacity
- L2 regularization using weight_decay
- model.train()
- model.eval()
- torch.no_grad()
- Model-capacity trade-offs
Neural Network Architecture
XOR Network
The NumPy and PyTorch XOR implementations use the same small architecture:
2 Input Features
       ↓
2 Hidden Neurons
       ↓
Sigmoid Activation
       ↓
1 Output Neuron
       ↓
Binary Prediction
XOR dataset:
Input      Target
[0, 0]  ->   0
[0, 1]  ->   1
[1, 0]  ->   1
[1, 1]  ->   0
Neural Network Learning Cycle
Input Data
    ↓
Forward Pass
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradients
    ↓
Optimizer / Gradient Descent
    ↓
Weight Update
    ↓
Repeat
Overfitting Experiment
The overfitting lab compares five configurations:
1. Baseline large network
2. Early stopping
3. Dropout
4. Smaller network
5. L2 regularization
A typical overfitting pattern is:
Training loss   -> keeps decreasing
Validation loss -> improves initially, then stops improving or increases
The epoch with the lowest validation loss is treated as the best-validation point.
Experimental Results
Experiment	Best Epoch	Best Validation Loss	Final Training Loss	Final Validation Loss	Final Train-Val Gap
Baseline	13	0.627974	0.009006	4.681148	4.672142
Early Stopping	13	0.627974	0.413534	1.010808	0.597274
Dropout	18	0.627878	0.389648	2.975877	2.586228
Smaller Network	59	0.622844	0.459826	0.943791	0.483965
L2 Regularization	16	0.624741	0.124339	1.735670	1.611332


Result Interpretation
The baseline large network showed severe overfitting. Its training loss dropped to 0.009006, while validation loss increased to 4.681148, producing a very large train-validation gap of 4.672142.
Early stopping reduced unnecessary training and stopped after only 53 epochs instead of 700.
Dropout reduced overfitting compared with the baseline, although the final validation gap remained noticeable.
The smaller network produced the strongest generalization in this experiment, achieving the lowest best validation loss (0.622844) and the smallest final train-validation gap (0.483965).
L2 regularization also improved generalization compared with the baseline by discouraging overly large weights.
Loss Curve Visualizations
Baseline Large Network

The baseline demonstrates severe overfitting: training loss keeps decreasing while validation loss worsens after the best validation region.
Early Stopping

Early stopping prevents continued training once validation performance stops improving.
Dropout

Dropout reduces dependence on individual neurons and can improve generalization.
Smaller Network

Reducing model capacity produced the best generalization in this experiment.
L2 Regularization

L2 regularization uses weight decay to discourage excessively large weights and reduce overfitting.
Key Learning from the Overfitting Lab
The experiment demonstrates that the model with the lowest training loss is not necessarily the best model.
A model should be evaluated on unseen validation or test data because the real objective is generalization, not memorization.
In this experiment:
Baseline:
Training Loss = 0.009006
Validation Loss = 4.681148
Despite excellent training performance, the model generalized poorly.
The smaller model had a higher training loss but much better validation behavior, demonstrating that lower model capacity can improve generalization.
NumPy vs PyTorch
NumPy
The NumPy implementation manually demonstrates:
- weights and biases
- matrix operations
- activation functions
- forward propagation
- Binary Cross-Entropy loss
- numerical gradients
- gradient-descent updates
- learning-rate behavior
PyTorch
The PyTorch implementation provides:
- reusable neural-network layers
- automatic differentiation
- built-in loss functions
- optimizers
- training/evaluation modes
- dropout
- weight decay
- scalable deep-learning workflows
NumPy helps explain how neural networks work internally, while PyTorch provides the framework used to train and evaluate larger practical models.
Project Structure
neural-network-fundamentals/
│
├── activations.py
├── loss.py
├── tiny_network.py
├── numerical_gradients.py
├── train_tiny_network.py
├── learning_rate_experiment.py
├── pytorch_xor.py
├── pytorch_learning_rate_experiment.py
├── plot_activations.py
│
├── overfitting_lab/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── plotting.py
│   ├── experiment.py
│   └── plots/
│       ├── baseline_large_network.png
│       ├── early_stopping.png
│       ├── dropout.png
│       ├── smaller_network.png
│       └── l2_regularization.png
│
├── requirements.txt
├── README.md
└── .gitignore
Installation
Create and activate a virtual environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install dependencies:
python -m pip install numpy matplotlib torch
Verify PyTorch:
python -c "import torch; print(torch.__version__)"
Run the NumPy Experiments
python main.py
python train_tiny_network.py
python learning_rate_experiment.py
Run the PyTorch XOR Model
python pytorch_xor.py
Optional learning-rate experiment:
python pytorch_learning_rate_experiment.py
Run the Overfitting Lab
python overfitting_lab\experiment.py
The generated loss-curve images are saved in:
overfitting_lab/plots/
Technologies and Skills
- Python
- NumPy
- PyTorch
- Matplotlib
- Neural Networks
- Forward Propagation
- Binary Cross-Entropy
- Numerical Gradients
- Backpropagation
- Gradient Descent
- PyTorch Autograd
- SGD
- Adam
- Training Loops
- Validation
- Overfitting Analysis
- Loss Curves
- Early Stopping
- Dropout
- L2 Regularization
- Model Capacity
- Generalization
- Git
- GitHub
Learning Outcomes
Through this project I learned how to:
- implement neural-network fundamentals from scratch
- understand weights, biases, activations, and weighted sums
- implement forward propagation
- calculate Binary Cross-Entropy loss
- understand gradients and backpropagation
- update parameters using gradient descent
- train an XOR classifier
- experiment with learning rates
- define models using nn.Module
- create layers using nn.Linear
- use PyTorch Autograd
- choose loss functions and optimizers
- build complete PyTorch training loops
- distinguish training from validation behavior
- read and interpret loss curves
- detect overfitting
- identify the best validation epoch
- apply early stopping
- apply dropout
- reduce network capacity
- apply L2 regularization
- compare regularization strategies
- understand why generalization matters more than memorizing training data