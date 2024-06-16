import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

def simulated_annealing(n, distance_matrix, cooling_scheme, max_iter=10000):
    # Initialize the route with a random permutation of stations
    current_route = np.random.permutation(n)
    current_route = np.append(current_route, current_route[0])  # Make the route circular
    
    # Current distance
    current_distance = total_distance(current_route, distance_matrix)
    
    best_route = np.copy(current_route)
    best_distance = current_distance
    
    for k in range(max_iter):
        # Temperature based on cooling scheme
        if cooling_scheme == 1:
            T = 1 / np.sqrt(1 + k)
        elif cooling_scheme == 2:
            T = -np.log(k + 1)
        else:
            raise ValueError("Invalid cooling scheme")
        
        # Swap two random stations (excluding the last one since it's the same as the first)
        i, j = np.random.choice(n, 2, replace=False)
        new_route = np.copy(current_route)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        
        # Ensure the route remains circular
        new_route[-1] = new_route[0]
        
        new_distance = total_distance(new_route, distance_matrix)
        
        # Accept new route based on probability
        if new_distance < current_distance or np.random.rand() < np.exp((current_distance - new_distance) / T):
            current_route = new_route
            current_distance = new_distance
            
            if current_distance < best_distance:
                best_route = np.copy(current_route)
                best_distance = current_distance
    
    return best_route, best_distance

# Load the cost matrix from CSV
cost_matrix = pd.read_csv('cost.csv', header=None).values

# Number of stations
n = cost_matrix.shape[0]

# Use the first cooling scheme
best_route, best_distance = simulated_annealing(n, cost_matrix, cooling_scheme=1)

print("Best Route:", best_route)
print("Best Distance:", best_distance)
