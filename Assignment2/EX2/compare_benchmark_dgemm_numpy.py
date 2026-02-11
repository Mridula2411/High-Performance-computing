import numpy as np
import time
from dgemm import dgemm_numpy


def time_python_dgemm(A, B, C, repeats=3):
    times = []

    for _ in range(repeats):
        C_copy = C.copy()
        start = time.perf_counter()
        dgemm_numpy(A, B, C_copy)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def time_numpy_dgemm(A, B, C, repeats=10):
    times = []

    for _ in range(repeats):
        C_copy = C.copy()
        start = time.perf_counter()
        C_copy += A @ B
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def flops_dgemm(N):
    return 2 * (N ** 3)


def main():
    np.random.seed(0)

    matrix_sizes = [64, 128, 256]
    print(
        f"{'N':>6} | "
        f"{'Python DGEMM (s)':>18} | "
        f"{'NumPy BLAS (s)':>16} | "
        f"{'Speedup':>10} | "
        f"{'Python FLOPs/s':>16} | "
        f"{'BLAS FLOPs/s':>16}"
    )
    print("-" * 95)

    for N in matrix_sizes:
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
        C = np.random.rand(N, N).astype(np.float64)

        t_python = time_python_dgemm(A, B, C)
        t_numpy = time_numpy_dgemm(A, B, C)

        flops = flops_dgemm(N)

        flops_python = flops / t_python
        flops_numpy = flops / t_numpy
        speedup = t_python / t_numpy

        print(
            f"{N:6d} | "
            f"{t_python:18.4f} | "
            f"{t_numpy:16.6f} | "
            f"{speedup:10.1f} | "
            f"{flops_python:16.3e} | "
            f"{flops_numpy:16.3e}"
        )


if __name__ == "__main__":
    main()
