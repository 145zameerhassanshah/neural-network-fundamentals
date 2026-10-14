import numpy as np

from activations import sigmoid
from loss import mse_loss

def forward_single_neuron(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    prediction = sigmoid(np.array([z]))[0]
    return z, prediction

def numerical_gradient_for_weight(
    inputs, weights, bias, target, weight_index, epsilon=1e-5
):
    weights_plus = weights.copy()
    weights_minus = weights.copy()

    weights_plus[weight_index] += epsilon
    weights_minus[weight_index] -= epsilon

    _, pred_plus = forward_single_neuron(inputs, weights_plus, bias)
    _, pred_minus = forward_single_neuron(inputs, weights_minus, bias)

    loss_plus = mse_loss(pred_plus, target)
    loss_minus = mse_loss(pred_minus, target)

    return (loss_plus - loss_minus) / (2.0 * epsilon)

def run_demo():
    print("=" * 65)
    print("BACKPROPAGATION INTUITION DEMO")
    print("=" * 65)

    inputs = np.array([2.0, 1.5, -0.5])
    weights = np.array([0.8, -0.4, 0.6], dtype=float)
    bias = 0.2
    target = 1.0
    learning_rate = 0.1

    _, prediction_before = forward_single_neuron(inputs, weights, bias)
    loss_before = mse_loss(prediction_before, target)

    gradient = numerical_gradient_for_weight(
        inputs, weights, bias, target, weight_index=0
    )

    updated_weights = weights.copy()
    updated_weights[0] -= learning_rate * gradient

    _, prediction_after = forward_single_neuron(
        inputs, updated_weights, bias
    )
    loss_after = mse_loss(prediction_after, target)

    print("Initial prediction:", prediction_before)
    print("Initial loss:", loss_before)
    print("Estimated gradient for weight[0]:", gradient)
    print("Updated weight[0]:", updated_weights[0])
    print("New prediction:", prediction_after)
    print("New loss:", loss_after)

    if loss_after < loss_before:
        print("Result: loss decreased after the weight update.")

if __name__ == "__main__":
    run_demo()
