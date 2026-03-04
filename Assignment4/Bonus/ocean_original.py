import numpy as np
import matplotlib.pyplot as plt
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
    u_new = u + alpha * laplacian(u) + beta * wind
    v_new = v + alpha * laplacian(v) + beta * wind
    temperature_new = temperature + 0.01 * laplacian(temperature)  # Small diffusion
    return u_new, v_new, temperature_new

grid_sizes = [200, 2000, 5000]
TIME_STEPS = 100

for grid_size in grid_sizes:
    # Initialize temperature field (random values between 5C and 30C)
    temperature = np.random.uniform(5, 30, size=(grid_size, grid_size))

    # Initialize velocity fields (u: x-direction, v: y-direction)
    u_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))
    v_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))

    # Initialize wind influence (adds turbulence)
    wind = np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size))

    # Run the simulation
    start = time.time()
    for t in range(TIME_STEPS):
        u_velocity, v_velocity, temperature = update_ocean(u_velocity, v_velocity, temperature, wind)
    end = time.time()

    print(f"Simulation (grid size: {grid_size}) completed in {end - start} seconds.")