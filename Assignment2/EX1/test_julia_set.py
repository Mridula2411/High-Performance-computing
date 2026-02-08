from julia_set import calc_pure_python
import pytest


def test_julia_set_sum():
    """
    Test correctness of Julia set computation for a 1000x1000 grid
    with 300 iterations.
    """
    output = calc_pure_python(desired_width=1000, max_iterations=300)
    assert sum(output) == 33219980

@pytest.mark.parametrize(
    "width, iterations, expected_sum",
    [
        (100, 50, 78990 ),
        (500, 200, 5798200),
        (1000, 300, 33219980),
    ],
)
def test_julia_set_general(width, iterations, expected_sum):
    output = calc_pure_python(width, iterations)
    assert sum(output) == expected_sum