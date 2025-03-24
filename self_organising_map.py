import matplotlib.pyplot as plt
import numpy as np
# Parameters for the SOM
num_cities = 7
num_neurons = num_cities * 2  # More neurons than cities for flexibility
learning_rate = 0.8
decay_factor = 0.999
num_iterations = 5000

# Randomly initialize neuron positions on a 2D plane
neurons = np.random.rand(num_neurons, 2)

# Randomly generate city positions for visualization
np.random.seed(42)  # For reproducibility
city_positions = np.random.rand(num_cities, 2)

# SOM Training Loop
for iteration in range(num_iterations):
    city_index = np.random.randint(0, num_cities)
    city = city_positions[city_index]

    # Find the winning neuron (closest to the selected city)
    distances = np.linalg.norm(neurons - city, axis=1)
    winner_index = np.argmin(distances)

    # Update neurons using Gaussian neighborhood function
    for i in range(num_neurons):
        distance_to_winner = abs(i - winner_index)
        influence = np.exp(-distance_to_winner**2 / (2 * (num_neurons / 4) ** 2))
        neurons[i] += learning_rate * influence * (city - neurons[i])

    # Decay learning rate
    learning_rate *= decay_factor

# Compute approximate TSP distance by summing Euclidean distances between neurons
approximate_tsp_distance = np.sum(np.linalg.norm(np.diff(neurons, axis=0), axis=1))

# Visualize the SOM result
plt.scatter(city_positions[:, 0], city_positions[:, 1], color='red', label="Cities")
plt.plot(neurons[:, 0], neurons[:, 1], color='blue', label="Neuron Path")
plt.legend()
plt.title("SOM Approximate TSP Solution")
plt.show()

approximate_tsp_distance
