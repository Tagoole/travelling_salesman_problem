import itertools  # Provides functions for creating iterators for efficient looping.
import numpy as np  # For numerical operations and array manipulation.

# Set a fixed random seed for reproducibility.
np.random.seed(42)

# -------------------------------
# Given adjacency matrix for the TSP
# -------------------------------
# This matrix represents the distances (or costs) between cities.
# Each element [i][j] is the distance from city i to city j.
# 'float("inf")' is used to represent missing direct paths between cities.
adj_matrix = [
    [float('inf'), 12, 10, float('inf'), float('inf'), float('inf'), 12],
    [12, float('inf'), 8, 12, float('inf'), float('inf'), float('inf')],
    [10, 8, float('inf'), 11, 3, float('inf'), 9],
    [float('inf'), 12, 11, float('inf'), 11, 10, float('inf')],
    [float('inf'), float('inf'), 3, 11, float('inf'), 6, 7],
    [float('inf'), float('inf'), float('inf'), 10, 6, float('inf'), 9],
    [12, float('inf'), 9, float('inf'), 7, 9, float('inf')]
]
num_cities = len(adj_matrix)  # Total number of cities

# City names for a more descriptive output of the tour.
city_names = ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7"]

# -------------------------------
# Step 1: Preprocess the matrix for MDS
# -------------------------------
# MDS (Multidimensional Scaling) requires a complete distance matrix.
# Replace 'inf' values with a large finite number (1000.0) to simulate missing edges.
adj_matrix_np = np.array(adj_matrix, dtype=float)
large_val = 1000.0
adj_matrix_np[adj_matrix_np == float('inf')] = large_val

# -------------------------------
# Step 2: Use MDS to embed cities into 2D space
# -------------------------------
def mds(D, d=2):
    """
    Classical Multidimensional Scaling (MDS)
    
    Parameters:
        D (numpy.ndarray): The (n x n) distance matrix.
        d (int): The target dimension for embedding (default is 2).
        
    Returns:
        numpy.ndarray: A (n x d) array of coordinates in the new space.
    """
    n = D.shape[0]
    # Square the distance matrix
    D2 = D ** 2
    # Create a centering matrix (H) to adjust the data
    H = np.eye(n) - np.ones((n, n)) / n
    # Double center the squared distance matrix to form the inner-product matrix B
    B = -0.5 * H.dot(D2).dot(H)
    # Perform eigen decomposition on B
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors in descending order based on eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Create a diagonal matrix with square roots of the top 'd' eigenvalues (ensuring non-negative values)
    L = np.diag(np.sqrt(np.maximum(eigvals[:d], 0)))
    V = eigvecs[:, :d]
    # Return the 2D coordinates by multiplying the eigenvectors with the diagonal matrix
    return V.dot(L)

# Generate 2D coordinates for each city using MDS.
coords = mds(adj_matrix_np, d=2)

# -------------------------------
# Step 3: Self-Organizing Map (SOM) for approximating the TSP tour
# -------------------------------
# A SOM is used to generate a heuristic solution for the TSP.
# It organizes a set of neurons into a ring that adapts to the layout of the cities.

# Set the number of neurons. Using more neurons than cities allows for a smoother curve.
num_neurons = 20

# Randomly initialize neuron positions in 2D space.
neurons = np.random.rand(num_neurons, 2)
# Scale the neurons so they cover the range of the city coordinates.
neurons = neurons * (coords.max(axis=0) - coords.min(axis=0)) + coords.min(axis=0)

# Define training parameters:
iterations = 1000            # Total number of iterations for training the SOM.
init_learning_rate = 0.8     # Starting learning rate.
init_radius = num_neurons / 2  # Initial neighborhood radius (half the number of neurons).
time_constant = iterations / np.log(init_radius)  # Time constant for exponential decay of the radius.

# Train the SOM:
for t in range(iterations):
    # Randomly select one city from the 2D embedded coordinates.
    city_idx = np.random.randint(0, coords.shape[0])
    city = coords[city_idx]
    
    # Compute the Euclidean distance from the selected city to all neurons.
    distances = np.linalg.norm(neurons - city, axis=1)
    # Identify the winning neuron (the neuron closest to the selected city).
    winner_idx = np.argmin(distances)
    
    # Decay the learning rate over time.
    learning_rate = init_learning_rate * np.exp(-t / iterations)
    # Decay the neighborhood radius over time.
    radius = init_radius * np.exp(-t / time_constant)
    
    # Update each neuron's position based on its circular distance from the winning neuron.
    for i in range(num_neurons):
        # Compute the circular (ring) distance between the current neuron and the winner.
        dist_to_winner = min(abs(i - winner_idx), num_neurons - abs(i - winner_idx))
        if dist_to_winner < radius:
            # Calculate the influence, which decreases with distance from the winner.
            influence = np.exp(-(dist_to_winner ** 2) / (2 * (radius ** 2)))
            # Adjust the neuron's position toward the selected city.
            neurons[i] += learning_rate * influence * (city - neurons[i])

# -------------------------------
# Step 4: Construct the TSP tour from the trained SOM
# -------------------------------
# For each city, assign the index of its nearest neuron.
city_to_neuron = {}
for i, city in enumerate(coords):
    distances = np.linalg.norm(neurons - city, axis=1)
    city_to_neuron[i] = np.argmin(distances)

# Sort the cities by the order of their nearest neuron along the ring.
sorted_cities = sorted(city_to_neuron, key=lambda k: city_to_neuron[k])

# Rotate the sorted list so that the tour starts with City 1 (index 0).
start_index = sorted_cities.index(0)
rotated_cities = sorted_cities[start_index:] + sorted_cities[:start_index]

# Append the starting city at the end to complete the cycle.
som_route = rotated_cities + [rotated_cities[0]]

# Calculate the total cost of the tour using the original adjacency matrix.
total_cost = 0
for i in range(len(som_route) - 1):
    total_cost += adj_matrix[som_route[i]][som_route[i+1]]

# -------------------------------
# Output the results
# -------------------------------
# Format the tour route using city names for better readability.
formatted_route = " -> ".join(city_names[i] for i in som_route)
print("SOM TSP Route:", formatted_route)
print("Total Route Cost (SOM):", total_cost)
