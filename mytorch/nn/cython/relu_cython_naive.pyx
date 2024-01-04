# cython: language_level=3
import numpy as np
cimport numpy as np

def relu_cython(np.ndarray t):
    cdef Py_ssize_t i, j
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if t[i, j] < 0:
                t[i, j] = 0
    return t
