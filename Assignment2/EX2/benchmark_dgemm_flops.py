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

    return np.mean(times), np.std(times)


def compute_flops(N):
    """
    DGEMM performs 2 * N^3 floating-point operations
    """
    return 2 * (N ** 3)


def main():
    np.random.seed(0)

    matrix_sizes = [32, 64, 96, 128]
    repeats = 5

    print(f"{'N':>6} | {'Mean Time (s)':>14} | {'Std (s)':>10} | {'FLOPs':>12} | {'FLOPs/s':>12}")
    print("-" * 75)

    for N in matrix_sizes:
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
        C = np.random.rand(N, N).astype(np.float64)

        mean_t, std_t = time_dgemm(A, B, C, repeats)

        flops = compute_flops(N)
        flops_per_sec = flops / mean_t

        print(
            f"{N:6d} | "
            f"{mean_t:14.4f} | "
            f"{std_t:10.4f} | "
            f"{flops:12.3e} | "
            f"{flops_per_sec:12.3e}"
        )


if __name__ == "__main__":
    main()
