import numpy as np
from activations import *
from utils import *


class Dense:
    back = True
    layer_code = "dense"

    def __init__(self, in_features, out_features, activation, initialization='normal', load_mode=False):
        if not load_mode:
            self.w = initializator(
                shape=(out_features, in_features), keyword=initialization)
            self.b = initializator(
                shape=(out_features, 1), keyword=initialization)
            self.activation = activation_dict[activation][0]
            self.dw = 0
            self.db = 0
            self.activation_derivative = activation_dict[activation][1]
        else:
            self.setParams()

    def setParams(self):
        pass

    def forward(self, a_in):
        self.a_in = a_in
        self.z = np.dot(self.a_in, self.w.T) + self.b.T
        self.a_out = sigmoid(self.z)
        return self.a_out

    def backward(self, grad):

        total_grad = grad
        dadz = self.activation_derivative(self.z)  # 4x4
        dz = total_grad * dadz  # elementwise
        dzdw = self.a_in

        self.dw = np.dot(dzdw.T, dz)
        self.db = dz.mean()
        da_in = np.dot(dz, self.w)

        return da_in

    def update(self, alpha):
        self.w = self.w - alpha * self.dw.T
        self.b = self.b - alpha * self.db


class Conv2d:
    back = True

    def __init__(self, input_channel, kernel_size: list, number_of_kernels: int, activation, stride=1, padding=0):

        # hyperparameters
        self.stride = stride
        self.padding = padding

        # .shapes
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self. number_of_kernels = number_of_kernels

        # parameters
        self.bias = np.random.randn(number_of_kernels)
        self.kernel = np.random.randn(number_of_kernels,
                                      kernel_size, kernel_size, input_channel)

        # activation stuff
        self.activation = activation_dict[activation][0]
        self.activation_derivative = activation_dict[activation][1]

    def forward(self, a_prev):

        self.out_a = []

        self.input = a_prev

        z, self.out_shape = correlation(self.input, self.kernel, self.bias,
                                        self.number_of_kernels, self.stride, self.padding)

        self.out_a = self.activation(z)

        return self.out_a

    def backward(self, grad_input):

        # preparing gradients
        grad_channels = grad_input.shape[-1]

        # activated to z
        dadz = self.activation_derivative(
            self.out_a)  # multiply with grad input

        self.dadz = dadz * grad_input

        # initialize zero matrix to fill with derivatives of loss w.r.t kernels
        self.dw_out = np.zeros(shape=(self.number_of_kernels, self.kernel_size, self.kernel_size,
                                      self.input_channel))

        # assure right shapes
        h_in, w_in = self.input.shape[1], self.input.shape[2]
        h_k, w_k = grad_input.shape[0], grad_input.shape[1]

        # calculate for each output gradient channel separately

        # derivative for w kernels;
        for g in range(grad_channels):

            gradient = self.dadz[:, :, :, g]

            for i in range(self.input_channel):

                for h in range(0, h_in-h_k, self.stride):
                    for w in range(0, w_in-w_k, self.stride):

                        correlated_slice = self.input[:,
                                                      w:w+w_k, h:h + h_k, i] * gradient

                        # assign derivatives

                        self.dw_out[g, w:w+w_k, h:h+h_k,
                                    i] += np.sum(correlated_slice)

        self.db = np.sum(self.dadz)
        # convolve output with rotated kernel
        self.dzda = np.zeros(self.input.shape)  # w.r.t input

        # padding for transpose conv
        pad_ = (self.input.shape[1] -
                self.kernel_size + self.out_a.shape[2] - 1)//2
        # pad_ = self.out_a.shape[2]-1

        # derivative for input
        for m in range(self.input.shape[0]):

            for k in range(self.number_of_kernels):
                # go along depth of kernel with 1 output channel
                for c in range(self.input_channel):

                    current_kernel = self.kernel[k, :, :, c]

                    rotated_kernel = np.flipud(
                        np.flipud(current_kernel))  # okay

                    # kth training example not to confues with kth chanel of output;

                    current_output = self.out_a[m, :, :, k]

                    # expand dimension to generalize;
                    rotated_kernel = np.expand_dims(
                        rotated_kernel, axis=[-1, 0])

                    current_output_exp = np.expand_dims(
                        current_output, axis=[0, -1])
                    result, _ = correlation(input=rotated_kernel, kernel=current_output_exp, bias=0,
                                            num_kernels=1, stride=1, pad_w=pad_)

                    self.dzda[m, :, :, c] = result[0, :, :, 0]

        # sum for same ones, product with prev grad

        return self.dzda

    def update(self, alpha):
        self.kernel = self.kernel - alpha * self.dw_out
        self.bias = self.bias - alpha * self.db


class MaxPool:

    back = True

    def __init__(self, pool_size, stride, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, x):
        h_in, w_in = x.shape[1], x.shape[2]
        self.a_in = x

        out_h = (h_in - self.pool_size)//self.stride + 1
        out_w = (w_in - self.pool_size)//self.stride + 1
        z_out = np.zeros(shape=(x.shape[0], out_h, out_w, x.shape[3]))
        print(z_out.shape, 'zourrr')
        for m in range(x.shape[0]):
            for c in range(x.shape[3]):
                channel = x[m, :, :, c]

                for h in range(0, out_h, self.stride):
                    for w in range(0, out_w, self.stride):
                        slice = channel[h:h+self.pool_size, w:w+self.pool_size]

                        if self.mode == 'max':
                            result = np.max(slice)
                        elif self.mode == 'avg':
                            result = np.average(slice)
                        z_out[m, h, w, c] = result

        return z_out

    def backward(self, grad_input):

        grad_flatten = grad_input.reshape(-1, 1)
        h_in, w_in = self.a_in.shape[1], self.a_in.shape[2]

        out_h = (h_in - self.pool_size)//self.stride + 1
        out_w = (w_in - self.pool_size)//self.stride + 1
        z_out = np.zeros(shape=(out_h, out_w, self.a_in.shape[2]))
        pool_window = np.zeros(shape=(self.pool_size, self.pool_size))
        out_index = 0
        for c in range(self.a_in.shape[2]):
            channel = self.a_in[:, :, c]

            for h in range(0, out_h, self.stride):
                for w in range(0, out_w, self.stride):

                    # slicing input
                    slice = channel[h:h+self.pool_size, w:w+self.pool_size]

                    if self.mode == 'max':
                        max_indices = np.unravel_index(
                            np.argmax(slice), slice.shape)
                        print(max_indices)
                        # pool_window[] = grad_flatten[out_index]

                        z_out[h, w] += pool_window
                        out_index += 1

                    elif self.mode == 'avg':
                        avg_grad = grad_flatten[out_index]/self.pool_size**2
                        slice[:] = avg_grad
                        out_index += 1
                        z_out[h, w] += slice

        return z_out

    def update(self, alpha):
        pass


class Flatten:
    back = False

    def __init__(self):
        self.x, self.shape = 0, 0

    def forward(self, x):
        self.x = x
        self.shape = x.shape
        out = reshape2d(x)
        self.out_shape = out.shape
        return out

    def backward(self, grad_input):
        grad_input = np.sum(grad_input, axis=0)

        out = grad_input.reshape(self.shape[1:])
        return out


class BatchNormalization:

    def __init__(self):
        self.betta = np.random.random()
        self.qamma = np.random.random()

    def forward(self, x):
        mean = np.mean(x, axis=0)
        std = np.mean(x, axis=0)
        self.z = (x-mean)/std
        self.a = self.qamma*self.z + self.betta
        return self.a  # m # training xs and by output node count for prev layer

    def backward(self, grad_input):

        sum_grad = np.sum(grad_input, axis=0)
        self.dq = sum_grad * self.z
        self.db = sum_grad
        self.da = sum_grad * self.qamma
        return self.da

    def update(self, alpha):
        self.qamma = self.qamma - alpha * self.dq
        self.betta = self.betta - alpha * self.db


layer_dict = {
    "dense": Dense,
    "conv2d": Conv2d
}
