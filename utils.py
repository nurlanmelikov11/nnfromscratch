import numpy as np


def initializator(shape, keyword='normal'):
    if keyword == "normal":
        return np.random.normal(size=shape)
    elif keyword == "uniform":
        return np.random.uniform(size=shape)


def pad(x, pad_w):
    if x.ndim == 2:
        padding = ((pad_w, 0), (0, pad_w))
    elif x.ndim == 3:
        padding = ((pad_w, 0), (0, pad_w), (0, 0))

    x = np.pad(x, pad_width=padding)
    return x


def correlation(input, kernel, bias, num_kernels, stride, pad_w):

    batch_size = input.shape[0]
    h_in, w_in = input.shape[1], input.shape[2]
    h_k, w_k = kernel.shape[1], kernel.shape[2]
    h_o = (h_in - h_k + 2*pad_w)//stride + 1
    w_o = (w_in - w_k + 2*pad_w)//stride + 1

    d_o = num_kernels
    z_out = np.zeros(shape=(batch_size, h_o, w_o, d_o))

    for m in range(batch_size):
        padded_input = pad(input[m], pad_w)

        for filter in range(kernel.shape[0]):
            current_kernel = kernel[filter, :, :, :]

            for i in range(current_kernel.shape[2]):  # depth iteration

                inp_channel = padded_input[:, :, i]

                for h in range(0, h_in-h_k, stride):
                    for w in range(0, w_in-w_k, stride):

                        correlated_slice = inp_channel[w:w+w_k, h:h +
                                                       h_k] * current_kernel[:, :, i] + bias[filter]

                        z_out[m, w:w+w_k, h:h+h_k,
                              i] += np.sum(correlated_slice)

    return z_out, z_out.shape


def pooling(x, pool_size, stride, mode='max'):
    h_in, w_in = x.shape[0], x.shape[1]

    out_h = (h_in - pool_size)//stride + 1
    out_w = (w_in - pool_size)//stride + 1
    z_out = np.zeros(shape=(out_h, out_w, x.shape[2]))

    for c in range(x.shape[2]):
        channel = x[:, :, c]

        for h in range(0, out_h, stride):
            for w in range(0, out_w, stride):
                slice = channel[h:h+pool_size, w:w+pool_size]

                if mode == 'max':
                    result = np.max(slice)
                elif mode == 'avg':
                    result = np.average(slice)
                z_out[h, w] = result

    return z_out


def max_finder(array):
    h, w = array.shape


def reshape2d(arr):
    N = np.prod(arr.shape[1:])

    reshaped_arr = arr.reshape(-1, N)

    return reshaped_arr
