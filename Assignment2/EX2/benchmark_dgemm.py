import numpy as np
import time
from dgemm import dgemm_numpy


def time_dgemm(A, B, C, repeats=5):
    times = []

    for _ in range(repeats):
        C_copy = C.copy()
        start = time.perf_counter()
        dgemm_numpy(A, B, C_copy)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times), min(times), max(times)


def main():
    np.random.seed(0)

    matrix_sizes = [32, 64, 96, 128]
    repeats = 5

    print(f"{'N':>6} | {'Mean (s)':>10} | {'Std (s)':>10} | {'Min (s)':>10} | {'Max (s)':>10}")
    print("-" * 60)

    for N in matrix_sizes:
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
        C = np.random.rand(N, N).astype(np.float64)

        mean_t, std_t, min_t, max_t = time_dgemm(A, B, C, repeats)

        print(f"{N:6d} | {mean_t:10.4f} | {std_t:10.4f} | {min_t:10.4f} | {max_t:10.4f}")


if __name__ == "__main__":
    main()
