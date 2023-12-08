
from costs import *
import numpy as np
import tqdm
from metrics import *


class Sequential:

    def __init__(self, layers: list):
        self.layers = layers
        self.cost_function = log_loss
        self.cost_function_derivative = log_loss_derivative
        self.metric = metric_dict['mse']

    def fit(self, x_train, y_train, epoch, batch_size, alpha):

        for i in tqdm.tqdm(range(epoch)):

            for batch_index in range(0, len(x_train), batch_size):
                batch_x, batch_y = x_train[batch_index:batch_index +
                                           batch_size], y_train[batch_index:batch_index+batch_size]  # reshape to contain shape info for count of training example per batch

                # forward
                for l in range(len(self.layers)):
                    a = self.layers[l].forward(batch_x)
                    batch_x = a

                # calculate error
                cost = log_loss(batch_y, a)

                print(cost, "cost per epoch : {}".format(i))
                dCost = self.cost_function_derivative(batch_y, a)

                grad = dCost

                # backpropagation
                for layer in self.layers[::-1]:
                    grad = layer.backward(grad)

                # update parameters
                for layer in self.layers:
                    if layer.back:
                        layer.update(alpha)

        return cost

    def compile(self, optimization='gd', metric="mse", early_stopping=False):
        self.metric = metric_dict[metric]

    def predict(self, x_test):
        a = x_test
        for layer in self.layers:
            a = layer.forward(a)

        return a
