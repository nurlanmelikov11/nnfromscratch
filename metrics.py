import numpy as np


def accuracy(yhat, y, threshold):
    labels = (yhat > threshold).astype('int')

    return (labels == y).sum()/y.size


def mean_squared_error(yhat, y):
    return np.sum((y-yhat)**2)/len(y)


def mean_absolute_error(yhat, y):
    return np.abs(yhat-y).sum()/len(y)


metric_dict = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "accuracy": accuracy
}
