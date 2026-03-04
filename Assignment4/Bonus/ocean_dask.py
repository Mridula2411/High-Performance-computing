import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import time

def laplacian(field):
    """Computes the discrete Laplacian of a 2D field using finite differences."""
    lap = (
        np.roll(field, shift=1, axis=0) +
        np.roll(field, shift=-1, axis=0) +
        np.roll(field, shift=1, axis=1) +
        np.roll(field, shift=-1, axis=1) -
        4 * field
    )
    return lap

def update_ocean(u, v, temperature, wind, alpha=0.1, beta=0.02):
    """Updates ocean velocity and temperature fields using a simplified flow model."""    
    u_lap = u.map_overlap(laplacian, depth=1, boundary='reflect')
    v_lap = v.map_overlap(laplacian, depth=1, boundary='reflect')
    t_lap = temperature.map_overlap(laplacian, depth=1, boundary='reflect')

    u_new = da.map_blocks(lambda u, ul, w: u + alpha * ul + beta * w, u, u_lap, wind)
    v_new = da.map_blocks(lambda v, vl, w: v + alpha * vl + beta * w, v, v_lap, wind)
    temperature_new = da.map_blocks(lambda t, tl: t + 0.01 * tl, temperature, t_lap)

    return u_new, v_new, temperature_new

# Grid size 
grid_sizes = [200, 2000, 5000]
TIME_STEPS = 100

for grid_size in grid_sizes:
    chunks = (grid_size/4,grid_size/4)
    # Init values
    temperature = da.random.uniform(5, 30, size=(grid_size, grid_size), chunks=chunks)
    u_velocity = da.random.uniform(-1, 1, size=(grid_size, grid_size), chunks=chunks)
    v_velocity = da.random.uniform(-1, 1, size=(grid_size, grid_size), chunks=chunks)
    wind = da.random.uniform(-0.5, 0.5, size=(grid_size, grid_size), chunks=chunks)

    # Run the simulation
    start = time.time()
    for t in range(TIME_STEPS):
        u_velocity, v_velocity, temperature = update_ocean(u_velocity, v_velocity, temperature, wind)
    u_final, v_final, t_final = da.compute(u_velocity, v_velocity, temperature)
    end = time.time()

    print(f"Simulation (grid size: {grid_size}) completed in {end - start} seconds.")