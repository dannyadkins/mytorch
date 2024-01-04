import numpy as np 
from mytorch.nn.cython import relu_cython_naive as cython_relu_naive_impl

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
    return cython_relu_naive_impl.relu_cython(t)

def bad_relu(t):
    return t 

def relu_naive_cuda(t):
    pass 

def relu_naive_triton(t):
    pass