from itertools import permutations
import sys

# Adjacency matrix representation of the graph
graph = [
    [0, 12, 10, 0, 0, 0, 12],
    [12, 0, 8, 12, 0, 0, 0],
    [10, 8, 0, 11, 3, 0, 9],
    [0, 12, 11, 0, 11, 10, 0],
    [0, 0, 3, 11, 0, 6, 7],
    [0, 0, 0, 10, 6, 0, 9],
    [12, 0, 9, 0, 7, 9, 0]
]

n = len(graph)  # Number of cities
INF = sys.maxsize  # Representing a large number for infinity

# Memoization table for storing subproblem results
dp = [[-1] * (1 << n) for _ in range(n)]

def tsp(mask, pos):
    """
    Solves TSP using Dynamic Programming with bitmasking.
    :param mask: Bitmask representing visited cities.
    :param pos: Current city.
    :return: Minimum tour cost from pos visiting unvisited cities.
    """
    # Base case: all cities visited, return to start (0)
    if mask == (1 << n) - 1:
        return graph[pos][0] if graph[pos][0] > 0 else INF
    
    # If already computed, return stored result
    if dp[pos][mask] != -1:
        return dp[pos][mask]
    
    min_cost = INF
    # Try visiting all cities that have not been visited yet
    for city in range(n):
        if (mask & (1 << city)) == 0 and graph[pos][city] > 0:
            new_cost = graph[pos][city] + tsp(mask | (1 << city), city)
            min_cost = min(min_cost, new_cost)
    
    # Store the computed result in the memoization table
    dp[pos][mask] = min_cost
    return min_cost

def find_tour():
    """
    Finds the optimal TSP tour and its cost.
    :return: Optimal path and its total cost
    """
    optimal_cost = tsp(1, 0)  # Start from city 0 with only it visited
    
    # Reconstruct the optimal path
    mask = 1  # Start with only the first city visited
    pos = 0  # Start at city 0
    path = [0]  # Store the optimal path
    
    for _ in range(n - 1):
        best_next = None
        best_cost = INF
        # Try finding the best next city to visit
        for city in range(n):
            if (mask & (1 << city)) == 0 and graph[pos][city] > 0:
                new_cost = graph[pos][city] + tsp(mask | (1 << city), city)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_next = city
        
        if best_next is None:
            break  # No valid next city, break loop
        
        path.append(best_next)  # Add city to the path
        mask |= (1 << best_next)  # Mark city as visited
        pos = best_next  # Move to next city
    
    path.append(0)  # Return to the starting city to complete the tour
    return path, optimal_cost

# Get the Final route and its Total route Cost
tour, cost = find_tour()
print("Final route:", [city + 1 for city in tour])  # Convert 0-based index to 1-based
print("Total route Cost:", cost)
