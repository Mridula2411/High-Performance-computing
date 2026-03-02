import numpy as np
import random
import dask
import dask.array as da
import matplotlib.pyplot as plt
from dask.distributed import Client, progress

# Constants
GRID_SIZE = 800
FIRE_SPREAD_PROB = 0.3  
BURN_TIME = 3  
DAYS = 60

# State definitions
EMPTY = 0    
TREE = 1     
BURNING = 2  
ASH = 3      

def initialize_forest():
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1
    return forest, burn_time

def get_neighbors(x, y):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

# Task 1.2: Convert to a Dask Delayed function
@dask.delayed
def simulate_wildfire_delayed(sim_id):
    """Simulates wildfire spread and returns a fixed-length array for Dask Array conversion."""
    forest, burn_time = initialize_forest()
    fire_spread = []
    
    for day in range(DAYS):
        new_forest = forest.copy()
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        if np.sum(forest == BURNING) == 0:
            break
            
    # Pad results with 0s to ensure consistent shape for Dask Array 
    padded_results = fire_spread + [0] * (DAYS - len(fire_spread))
    return np.array(padded_results)

def run_sim(num_simulations=10, num_workers=1):
    """Executes simulations using Dask Distributed and aggregates results."""
    # 1. Initialize Dask Client for Dashboard access 
    client = Client(processes=False, n_workers=num_workers) 
    print(f"Dask Dashboard is active at: {client.dashboard_link}")

    # 2. Create Delayed objects (Task Graph) 
    delayed_tasks = [simulate_wildfire_delayed(i) for i in range(num_simulations)]

    # 3. Convert results into a Dask Array for efficient aggregation 
    # We create a Dask array from the list of delayed objects
    dask_arrays = [da.from_delayed(task, shape=(DAYS,), dtype=int) for task in delayed_tasks]
    stacked_sims = da.stack(dask_arrays, axis=0)

    # 4. Compute the average across the simulation axis (axis 0)
    mean_spread_task = stacked_sims.mean(axis=0)

    # 5. Execute calculations
    print("Executing simulations in parallel via Dask...")
    avg_fire_spread = mean_spread_task.compute()
    
    client.close()
    return avg_fire_spread

if __name__ == "__main__":
    # Parameters
    NUM_SIMS = 10
    
    # Run parallel Dask execution
    avg_fire_spread = run_sim(num_simulations=NUM_SIMS)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(range(DAYS), avg_fire_spread, label="Average Burning Trees", color='orange')
    plt.xlabel("Days")
    plt.ylabel("Average Number of Burning Trees")
    plt.title(f"Dask-Parallelized Wildfire Simulation ({NUM_SIMS} Runs)")
    plt.legend()
    plt.grid(True)
    plt.show()