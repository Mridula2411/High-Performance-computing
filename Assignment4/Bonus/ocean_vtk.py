import numpy as np
import matplotlib.pyplot as plt
np.float_ = np.float64
np.int_ = np.int64
import pyvtk

# Grid size 
grid_size = 200  
TIME_STEPS = 100

# Initialize temperature field (random values between 5C and 30C)
temperature = np.random.uniform(5, 30, size=(grid_size, grid_size))

# Initialize velocity fields (u: x-direction, v: y-direction)
u_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))
v_velocity = np.random.uniform(-1, 1, size=(grid_size, grid_size))

# Initialize wind influence (adds turbulence)
wind = np.random.uniform(-0.5, 0.5, size=(grid_size, grid_size))

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

def save_ocean_vtk(filename, u, v, temp):
    nx, ny = temp.shape

    u_flat = u.T.flatten()
    v_flat = v.T.flatten()
    temp_flat = temp.T.flatten()

    vtk_data = pyvtk.VtkData(
        pyvtk.StructuredPoints([nx, ny, 1]), 
        pyvtk.PointData(
            pyvtk.Scalars(temp_flat, name="Temperature"),
            pyvtk.Vectors(np.column_stack((u_flat, v_flat, np.zeros_like(u_flat))), name="Velocity")
        )
    )
    vtk_data.tofile(filename)

# Run the simulation
for t in range(TIME_STEPS):
    u_velocity, v_velocity, temperature = update_ocean(u_velocity, v_velocity, temperature, wind)
    if t % 10 == 0 or t == TIME_STEPS - 1:
        fname = f"vtk_results/ocean_sim_{t:04d}.vtk"
        save_ocean_vtk(fname, u_velocity, v_velocity, temperature)
        print(f"Time Step {t}: Saved {fname}")

# Plot the velocity field
# plt.figure(figsize=(6, 5))
# plt.quiver(u_velocity[::10, ::10], v_velocity[::10, ::10])
# plt.title("Ocean Current Directions")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.show()

# # Plot temperature distribution
# plt.figure(figsize=(6, 5))
# plt.imshow(temperature, cmap='coolwarm', origin='lower')
# plt.colorbar(label="Temperature (°C)")
# plt.title("Ocean Temperature Distribution")
# plt.show()

# print("Simulation complete.")