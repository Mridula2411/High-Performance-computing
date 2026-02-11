import numpy as np

def dgemm_numpy(A, B, C):
    """
    Perform DGEMM: C = C + A * B
    """
    N = A.shape[0]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

    return C
