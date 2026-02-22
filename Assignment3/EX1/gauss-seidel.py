import numpy as np
import matplotlib.pyplot as plt
import time

def gauss_seidel(f):
    N = f.shape[0]
    for i in range(1, N-1):
        for j in range(1, N-1):
            f[i, j] = 0.25 * (
                f[i+1, j] + f[i-1, j] +
                f[i, j+1] + f[i, j-1]
            )
    return f


def initialize_grid(N):
    f = np.random.rand(N, N)
    f[0, :] = 0
    f[-1, :] = 0
    f[:, 0] = 0
    f[:, -1] = 0
    return f

def measure_performance(N, iterations=1000):
    f = initialize_grid(N)

    start = time.time()
    for _ in range(iterations):
        f = gauss_seidel(f)
    end = time.time()

    return end - start

if __name__ == "__main__":
    grid_sizes = [32, 64, 96, 128]
    times = []

    for N in grid_sizes:
        t = measure_performance(N)
        times.append(t)
        print(f"N = {N}, Time = {t:.4f} s")

    # Plot results
    plt.figure()
    plt.plot(grid_sizes, times, marker='o')
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Execution Time (s)")
    plt.title("Gauss-Seidel Performance")
    plt.grid(True)
    plt.show()