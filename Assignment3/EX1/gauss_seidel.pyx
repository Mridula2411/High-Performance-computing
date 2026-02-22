import numpy as np
cimport numpy as np

def gauss_seidel(np.ndarray[np.float64_t, ndim=2] f):
    cdef int i, j
    cdef int N = f.shape[0]

    for i in range(1, N-1):
        for j in range(1, N-1):
            f[i, j] = 0.25 * (
                f[i+1, j] + f[i-1, j] +
                f[i, j+1] + f[i, j-1]
            )

    return f