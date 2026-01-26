# Bonus Profiler

A modular CPU profiler for Python applications.

## Installation for development

Install the package in editable mode from the project directory:

```bash
pip install -e ./Bonus-Profiler
```

Or from within the Bonus-Profiler directory:

```bash
pip install -e .
```

## Requirements

- Python >= 3.9
- psutil
- plotly
- tabulate
- ipywidgets

## Design Choices

This profiler was designed with modularity and ease of use in mind. The architecture separates concerns into distinct components:

**Decorator-based profiling:** The @profile_cpu decorator provides a non-intrusive way to profile functions without modifying their internal code, making it easy to add or remove profiling from any function

**Background monitoring:** CPU profiling runs in a separate thread, allowing it to collect metrics at regular intervals without blocking the main program execution. This ensures accurate resource measurements even during intensive computations

**Global profiler instance:** A singleton pattern tracks all decorated function calls and automatically generates a consolidated report at program exit using Python's atexit mechanism, eliminating the need for manual cleanup

**Modular utilities:** The visualization and table generation logic is separated into reusable utility functions (plot.py and table.py), making it straightforward to extend the profiler for other metrics (e.g., memory, GPU) while maintaining consistent output formatting

## Example usage

```python
from profilers import profile_cpu

@profile_cpu(interval=1)
def evolve(grid, dt, D=1.0):
    xmax, ymax = grid_shape
    new_grid = [[0.0] * ymax for x in range(xmax)]
    for i in range(xmax):
        for j in range(ymax):
            grid_xx = (
                grid[(i + 1) % xmax][j] + grid[(i - 1) % xmax][j] - 2.0 * grid[i][j]
            )
            grid_yy = (
                grid[i][(j + 1) % ymax] + grid[i][(j - 1) % ymax] - 2.0 * grid[i][j]
            )
            new_grid[i][j] = grid[i][j] + D * (grid_xx + grid_yy) * dt
    return new_grid
```
