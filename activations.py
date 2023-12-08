
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(x):
    return np.where(x < 0, 0, 1)


activation_dict = {
    "sigmoid": [sigmoid, sigmoid_derivative],
    "relu": [relu, relu_derivative]
}
