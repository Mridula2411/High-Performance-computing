import numpy as np


def initialize_grid(N):
    f = np.random.rand(N, N)
    f[0, :] = 0
    f[-1, :] = 0
    f[:, 0] = 0
    f[:, -1] = 0
    return f


@profile
def gauss_seidel(f):
    N = f.shape[0]
    for i in range(1, N-1):
        for j in range(1, N-1):
            f[i, j] = 0.25 * (
                f[i+1, j] + f[i-1, j] +
                f[i, j+1] + f[i, j-1]
            )
    return f


def run_solver(N, iterations=100):
    f = initialize_grid(N)
    for _ in range(iterations):
        f = gauss_seidel(f)
    return f


if __name__ == "__main__":
    run_solver(128)