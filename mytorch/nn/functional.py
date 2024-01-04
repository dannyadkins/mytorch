def relu_naive(t):
    # super naive implementation of relu from scratch:
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if t[i, j] < 0:
                t[i, j] = 0
    return t

def bad_relu(t):
    return t 

def relu_naive_cuda(t):
    pass 

def relu_naive_triton(t):
    pass