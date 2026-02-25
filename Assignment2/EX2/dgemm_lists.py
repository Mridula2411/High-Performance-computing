import random

def dgemm_lists(A, B, C, N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


# Initialize matrices
def initialize_lists(N):
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
    C = [[0.0 for _ in range(N)] for _ in range(N)]
    return A, B, C