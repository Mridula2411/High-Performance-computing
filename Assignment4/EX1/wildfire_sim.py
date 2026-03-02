import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

def save_to_vtk(grid, day):
    """Saves the forest grid as a VTK structured points file for ParaView."""
    nx, ny = grid.shape
    # Create a directory for VTK files if it doesn't exist
    if not os.path.exists("vtk_output"):
        os.makedirs("vtk_output")
        
    filename = f"vtk_output/wildfire_day_{day:03d}.vtk"
    
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Wildfire Simulation Grid\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {nx * ny}\n")
        f.write("SCALARS forest_state int 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        # Flatten and write the grid data
        for val in grid.flatten():
            f.write(f"{val}\n")


def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

def simulate_wildfire(save_vtk=False):
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):

        if save_vtk:
            save_to_vtk(forest, day)

        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            if save_vtk:
                save_to_vtk(forest, day + 1)  # Save final state
            break
    
    return fire_spread

def run_sim(num_runs=10):
    """Runs multiple wildfire simulations and averages the results."""
    all_spreads = []
    
    for i in range(num_runs):
        spread = simulate_wildfire()
        all_spreads.append(spread)
    
    # Pad results to the same length and average
    max_length = max(len(spread) for spread in all_spreads)
    padded_spreads = [spread + [0] * (max_length - len(spread)) for spread in all_spreads]
    
    average_spread = np.mean(padded_spreads, axis=0)
    
    return average_spread




if __name__ == "__main__":

    # avg_fire_spread_over_time = run_sim(num_runs=10)

    # # Plot results
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(len(avg_fire_spread_over_time)), avg_fire_spread_over_time, label="Burning Trees")
    # plt.xlabel("Days")
    # plt.ylabel("Number of Burning Trees")
    # plt.title("Wildfire Spread Over Time")
    # plt.legend()
    # plt.show()

    # Save final state as VTK for ParaView
    simulate_wildfire(save_vtk=True)