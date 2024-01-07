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


# conv2d methods
# currently assumes stride=1 and padding=0
def conv2d_naive(t, kernel):
    # print t and ke
    print("t: ", t)
    print("kernel: ", kernel)
    t_height, t_width = t.shape
    k_height, k_width = kernel.shape

    output_height = t_height - k_height + 1
    output_width = t_width - k_width + 1

    output = torch.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = torch.sum(t[i:i+k_height, j:j+k_width] * kernel)

    return output