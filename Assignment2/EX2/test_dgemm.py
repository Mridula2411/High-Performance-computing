import numpy as np
from array import array
import pytest
from dgemm import dgemm_lists, dgemm_arrays, dgemm_numpy


def generate_test_matrices(N):
    np.random.seed(0)
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))
    return A, B, C


def test_dgemm_lists():
    N = 3
    A_np, B_np, C_np = generate_test_matrices(N)

    # Convert to lists
    A = A_np.tolist()
    B = B_np.tolist()
    C = C_np.tolist()

    result = dgemm_lists(A, B, C, N)

    expected = A_np @ B_np

    assert np.allclose(result, expected)


def test_dgemm_arrays():
    N = 3
    A_np, B_np, C_np = generate_test_matrices(N)

    # Convert to arrays
    A = [array('d', row) for row in A_np]
    B = [array('d', row) for row in B_np]
    C = [array('d', row) for row in C_np]

    result = dgemm_arrays(A, B, C, N)

    # Convert result back to numpy for comparison
    result_np = np.array([list(row) for row in result])
    expected = A_np @ B_np

    assert np.allclose(result_np, expected)


def test_dgemm_correctness():
    np.random.seed(0)
    N = 4

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.random.rand(N, N).astype(np.float64)

    C_test = dgemm_numpy(A, B, C.copy())
    C_expected = C + A @ B

    assert np.allclose(C_test, C_expected)

@pytest.mark.parametrize("N", [1, 2, 5, 10])
def test_dgemm_multiple_sizes(N):
    np.random.seed(1)

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.random.rand(N, N).astype(np.float64)

    C_test = dgemm_numpy(A, B, C.copy())
    C_expected = C + A @ B

    assert np.allclose(C_test, C_expected)