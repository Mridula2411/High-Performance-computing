from array import array
import random

def initialize_arrays(N):
    A = [array('d', [random.random() for _ in range(N)]) for _ in range(N)]
    B = [array('d', [random.random() for _ in range(N)]) for _ in range(N)]
    C = [array('d', [0.0 for _ in range(N)]) for _ in range(N)]
    return A, B, C


def dgemm_arrays(A, B, C, N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C