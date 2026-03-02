import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing

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

def simulate_wildfire(sim_id):
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):
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
            break
        
    
    return fire_spread



def run_sim(num_simulations=10, num_workers=1):
    """Runs multiple wildfire simulations in parallel and averages the results."""
    num_workers = num_workers 

    # Create a pool of workers and run simulations in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(simulate_wildfire, range(num_simulations))

    # Average the results across simulations
    max_days = max(len(result) for result in results)
    avg_fire_spread = np.zeros(max_days)

    for d in range(max_days):
        day_values = [result[d] for result in results if d < len(result)]
        avg_fire_spread[d] = np.mean(day_values)


    return avg_fire_spread

if __name__ == "__main__":
    
    avg_fire_spread = run_sim(num_simulations=10)


    
    # Plot averaged results
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(avg_fire_spread)), avg_fire_spread, label="Average Burning Trees")
    plt.xlabel("Days")
    plt.ylabel("Average Number of Burning Trees")
    plt.title("Average Wildfire Spread Over Time (10 Simulations)")
    plt.legend()
    plt.show()