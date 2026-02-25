import numpy as np
import time
from array import array
from statistics import mean, stdev


# ==============================
# DGEMM IMPLEMENTATIONS
# ==============================

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


# ==============================
# BENCHMARK FUNCTION
# ==============================

def benchmark(func, A, B, C, N=None, repeats=5):
    times = []

    for _ in range(repeats):

        # Copy C so each run starts fresh
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
        "std": stdev(times),
        "min": min(times),
        "max": max(times),
    }


# ==============================
# FLOPS CALCULATION
# ==============================

def compute_flops(N, time_seconds):
    total_flops = 2 * (N ** 3)
    flops_per_sec = total_flops / time_seconds
    gflops = flops_per_sec / 1e9
    return flops_per_sec, gflops


# ==============================
# MAIN BENCHMARK
# ==============================

def main():

    sizes = [50, 100, 200]
    repeats = 5

    # Change this to your CPU frequency
    cpu_freq_ghz = 3.0
    theoretical_peak_gflops = cpu_freq_ghz  # assuming 1 flop per cycle

    print(f"Theoretical Peak (per core, 1 flop/cycle): {theoretical_peak_gflops} GFLOPS\n")

    for N in sizes:

        print("=" * 50)
        print(f"Matrix size: {N} x {N}")

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
        stats_lists = benchmark(dgemm_lists, A_list, B_list, C_list, N, repeats)
        stats_arrays = benchmark(dgemm_arrays, A_arr, B_arr, C_arr, N, repeats)
        stats_numpy = benchmark(dgemm_numpy, A_np, B_np, C_np, None, repeats)

        for name, stats in [("Lists", stats_lists),
                            ("Arrays", stats_arrays),
                            ("NumPy", stats_numpy)]:

            flops, gflops = compute_flops(N, stats["mean"])

            print(f"\n{name}:")
            print(f"  Mean Time: {stats['mean']:.6f} s")
            print(f"  Std Dev  : {stats['std']:.6f} s")
            print(f"  GFLOPS   : {gflops:.6f}")
            print(f"  % of Peak: {(gflops / theoretical_peak_gflops) * 100:.4f} %")


if __name__ == "__main__":
    main()