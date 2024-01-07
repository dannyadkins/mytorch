import numpy as np 
from mytorch.nn.cython import relu_cython_naive as cython_relu_naive_impl
import torch 
from numba import cuda

@cuda.jit
def relu_cuda_kernel(t):
    x, y = cuda.grid(2)
    if x < t.shape[0] and y < t.shape[1]:
        if t[x, y] < 0:
            t[x, y] = 0

def relu_cuda(t):
    t_gpu = cuda.to_device(t)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(t.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(t.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    relu_cuda_kernel[blockspergrid, threadsperblock](t_gpu)
    return t_gpu.copy_to_host()

def relu_naive(t):
    # super naive implementation of relu from scratch:
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if t[i, j] < 0:
                t[i, j] = 0
    return t

def relu_naive_inplace(t):
    t[t < 0] = 0
    return t

def relu_vectorized_numpy(t):
    return np.maximum(t, 0)

def relu_cython_naive(t):
    nparr = t.numpy()
    cython_relu_naive_impl.relu_cython(nparr)
    return torch.from_numpy(nparr)

def bad_relu(t):
    return t 

def relu_naive_cuda(t):
    pass 

def relu_naive_triton(t):
    pass


### conv2d 

# currently assumes stride=1 and padding=0
def conv2d_naive(t, kernel, stride=1, padding=0):
    # get the shape of the input tensor and the kernel
    batch_size, num_channels, t_height, t_width = t.shape
    k_out_channels, k_in_channels, k_height, k_width = kernel.shape

    # calculate the output dimensions
    output_height = ((t_height - k_height + 2 * padding) // stride) + 1
    output_width = ((t_width - k_width + 2 * padding) // stride) + 1

    # initialize the output tensor with zeros, it's going to hold the convolved results
    output = torch.zeros((batch_size, k_out_channels, output_height, output_width))

    # now we're going to do the convolution operation
    # we loop over every element in the batch
    for b in range(batch_size):
        # for each output channel
        for k_out in range(k_out_channels):
            # and for each input channel
            for t_channel in range(num_channels):
                # slide the kernel over the input tensor
                for i in range(0, t_height - k_height + 1 + padding, stride):
                    for j in range(0, t_width - k_width + 1 + padding, stride):
                        # for each position, we do an element-wise multiplication
                        # and sum the results to get the convolved value
                        for h in range(k_height):
                            for w in range(k_width):
                                # make sure we're within the bounds of the input tensor
                                if (i + h < t_height) and (j + w < t_width):
                                    # accumulate the results in the output tensor
                                    output[b, k_out, i // stride, j // stride] += t[b, t_channel, i + h, j + w] * kernel[k_out, t_channel, h, w]

    return output

### maxpool2d

# input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
def max_pool2d_naive(input_tensor, pool_kernel_size, pool_stride, pool_padding):
    # get the shape of the input tensor
    batch_size, num_channels, input_height, input_width = input_tensor.shape

    # calculate the output dimensions
    output_height = ((input_height - pool_kernel_size[0] + 2 * pool_padding[0]) // pool_stride[0]) + 1
    output_width = ((input_width - pool_kernel_size[1] + 2 * pool_padding[1]) // pool_stride[1]) + 1

    # initialize the output tensor with negative infinity values for max pooling
    output = torch.full((batch_size, num_channels, output_height, output_width), float('-inf'))

    # loop over every element in the batch
    for batch_idx in range(batch_size):
        # for each channel
        for channel_idx in range(num_channels):
            # slide the pooling kernel over the input tensor
            for i in range(0, input_height - pool_kernel_size[0] + 1, pool_stride[0]):
                for j in range(0, input_width - pool_kernel_size[1] + 1, pool_stride[1]):
                    # for each position, find the maximum value
                    for h in range(pool_kernel_size[0]):
                        for w in range(pool_kernel_size[1]):
                            # ensure we're within the bounds of the input tensor
                            if (i + h < input_height) and (j + w < input_width):
                                # update the maximum value in the output tensor
                                output[batch_idx, channel_idx, i // pool_stride[0], j // pool_stride[1]] = max(output[batch_idx, channel_idx, i // pool_stride[0], j // pool_stride[1]], input_tensor[batch_idx, channel_idx, i + h, j + w])

    return output

### batchnorm2d

def batch_norm_naive(input_tensor, running_mean, running_var):
    # get the shape of the input tensor
    batch_size, num_channels, input_height, input_width = input_tensor.shape

    # initialize the output tensor
    output = torch.zeros((batch_size, num_channels, input_height, input_width))

    # loop over every element in the batch
    for batch_idx in range(batch_size):
        # for each channel
        for channel_idx in range(num_channels):
            # for each position
            for i in range(input_height):
                for j in range(input_width):
                    # normalize the input tensor
                    output[batch_idx, channel_idx, i, j] = (input_tensor[batch_idx, channel_idx, i, j] - running_mean) / torch.sqrt(running_var)

    return output