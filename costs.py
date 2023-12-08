import numpy as np


def log_loss(y, y_hat):
    m = len(y)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m


def log_loss_derivative(y, y_hat):

    y = y.reshape(-1, 1)
    result = -(y / y_hat) + (1 - y) / (1 - y_hat)
    return result
