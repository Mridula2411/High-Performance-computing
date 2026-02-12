import JuliaSet
def test_juliaSet():
    output = JuliaSet.calc_pure_python(desired_width=1000, max_iterations=300)
    assert sum(output) == 33219980