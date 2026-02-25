import numpy as np
import time
from array import array
from statistics import mean, stdev


# =====================================
# DGEMM IMPLEMENTATIONS
# =====================================

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


def dgemm_numpy_at(A, B, C):
    C += A @ B
    return C


def dgemm_numpy_matmul(A, B, C):
    C += np.matmul(A, B)
    return C


# =====================================
# BENCHMARK FUNCTION
# =====================================

def benchmark(func, A, B, C, N=None, repeats=5):
    times = []

    for _ in range(repeats):

        # Copy C for fresh computation
        if isinstance(C, np.ndarray):
            C_copy = C.copy()
        else:
            C_copy = [row[:] for row in C]

        start = time.perf_counter()

        if N is not None:
            func(A, B, C_copy, N)
        else:
            func(A, B, C_copy)

        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": mean(times),
        "std": stdev(times)
    }


# =====================================
# FLOPS CALCULATION
# =====================================

def compute_gflops(N, time_seconds):
    total_flops = 2 * (N ** 3)
    return (total_flops / time_seconds) / 1e9


# =====================================
# MAIN
# =====================================

def main():

    sizes = [100, 200, 400]
    repeats = 5

    for N in sizes:

        print("=" * 60)
        print(f"Matrix size: {N} x {N}")

        # Generate matrices
        A_np = np.random.rand(N, N)
        B_np = np.random.rand(N, N)
        C_np = np.zeros((N, N))

        # Lists
        A_list = A_np.tolist()
        B_list = B_np.tolist()
        C_list = [[0.0] * N for _ in range(N)]

        # Arrays
        A_arr = [array('d', row) for row in A_np]
        B_arr = [array('d', row) for row in B_np]
        C_arr = [array('d', [0.0] * N) for _ in range(N)]

        # Run benchmarks
        results = {
            "Lists": benchmark(dgemm_lists, A_list, B_list, C_list, N, repeats),
            "Arrays": benchmark(dgemm_arrays, A_arr, B_arr, C_arr, N, repeats),
            "NumPy @": benchmark(dgemm_numpy_at, A_np, B_np, C_np, None, repeats),
            "NumPy matmul": benchmark(dgemm_numpy_matmul, A_np, B_np, C_np, None, repeats),
        }

        for name, stats in results.items():
            gflops = compute_gflops(N, stats["mean"])

            print(f"\n{name}:")
            print(f"  Mean Time: {stats['mean']:.6f} s")
            print(f"  Std Dev  : {stats['std']:.6f} s")
            print(f"  GFLOPS   : {gflops:.3f}")


if __name__ == "__main__":
    main()