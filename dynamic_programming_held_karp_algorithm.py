import numpy as np
from functools import lru_cache


# Define the adjacency matrix for the TSP graph
tsp_graph = np.array([
    [0, 12, 10, 8, 12, 3, 9],  # City 1
    [12, 0, 12, 11, 10, 11, 6],  # City 2
    [10, 12, 0, 9, 11, 10, 7],  # City 3
    [8, 11, 9, 0, 6, 7, 9],  # City 4
    [12, 10, 11, 6, 0, 9, 11],  # City 5
    [3, 11, 10, 7, 9, 0, 9],  # City 6
    [9, 6, 7, 9, 11, 9, 0]  # City 7
])

num_cities = tsp_graph.shape[0]


# Define the TSP solver using Dynamic Programming (Held-Karp)
@lru_cache(None)
def tsp(dp_mask, pos):
    if dp_mask == (1 << num_cities) - 1:
        return tsp_graph[pos][0]  # Return to starting city

    min_cost = float('inf')
    for city in range(num_cities):
        if (dp_mask >> city) & 1 == 0:  # If city not visited
            new_mask = dp_mask | (1 << city)
            cost = tsp_graph[pos][city] + tsp(new_mask, city)
            min_cost = min(min_cost, cost)

    return min_cost

# Start the TSP from City 1 (index 0)
optimal_tsp_distance = tsp(1, 0)
print("Optimal TSP Distance:", optimal_tsp_distance)
