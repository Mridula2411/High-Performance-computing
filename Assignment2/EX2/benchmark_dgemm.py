import numpy as np
import time
from array import array
from statistics import mean, stdev

from dgemm import dgemm_lists, dgemm_arrays, dgemm_numpy


def benchmark(func, A, B, C, N=None, repeats=5):
    times = []

    for _ in range(repeats):
        C_copy = C.copy() if isinstance(C, np.ndarray) else [row[:] for row in C]

        start = time.perf_counter()

        if N is not None:
            func(A, B, C_copy, N)
        else:
            func(A, B, C_copy)

        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": mean(times),
        "std": stdev(times),
        "min": min(times),
        "max": max(times),
    }


def run_benchmarks():
    sizes = [50, 100, 200]   # Increase carefully (lists get very slow)
    repeats = 5

    for N in sizes:
        print(f"\nMatrix size: {N}x{N}")

        # NumPy matrices
        A_np = np.random.rand(N, N)
        B_np = np.random.rand(N, N)
        C_np = np.zeros((N, N))

        # Lists
        A_list = A_np.tolist()
        B_list = B_np.tolist()
        C_list = [[0.0]*N for _ in range(N)]

        # Arrays
        A_arr = [array('d', row) for row in A_np]
        B_arr = [array('d', row) for row in B_np]
        C_arr = [array('d', [0.0]*N) for _ in range(N)]

        # Run benchmarks
        print("Lists:", benchmark(dgemm_lists, A_list, B_list, C_list, N, repeats))
        print("Arrays:", benchmark(dgemm_arrays, A_arr, B_arr, C_arr, N, repeats))
        print("NumPy:", benchmark(dgemm_numpy, A_np, B_np, C_np, None, repeats))


if __name__ == "__main__":
    run_benchmarks()