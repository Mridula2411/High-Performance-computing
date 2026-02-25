import numpy as np
from array import array

def dgemm_lists(A, B, C, N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def dgemm_arrays(A, B, C, N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def dgemm_numpy(A, B, C):
    C += A @ B
    return C