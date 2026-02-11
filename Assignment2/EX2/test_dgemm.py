import numpy as np
from dgemm import dgemm_numpy

def test_dgemm_correctness():
    np.random.seed(0)
    N = 4

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.random.rand(N, N).astype(np.float64)

    C_test = dgemm_numpy(A, B, C.copy())
    C_expected = C + A @ B

    assert np.allclose(C_test, C_expected)
    print("DGEMM correctness test passed.")

import pytest

@pytest.mark.parametrize("N", [1, 2, 5, 10])
def test_dgemm_multiple_sizes(N):
    np.random.seed(1)

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.random.rand(N, N).astype(np.float64)

    C_test = dgemm_numpy(A, B, C.copy())
    C_expected = C + A @ B

    assert np.allclose(C_test, C_expected)