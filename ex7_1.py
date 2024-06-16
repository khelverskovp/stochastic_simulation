import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route) - 1))

def simulated_annealing(n, coordinates, cooling_scheme, max_iter=10000):
    # Initialize the route with a random permutation of stations
    current_route = np.random.permutation(n)
    current_route = np.append(current_route, current_route[0])  # Make the route circular
    
    # Create the distance matrix
    distance_matrix = np.array([[euclidean_distance(coordinates[i], coordinates[j]) for j in range(n)] for i in range(n)])
    
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

def plot_route(route, coordinates):
    plt.figure(figsize=(10, 6))
    for i in range(len(route) - 1):
        plt.plot([coordinates[route[i]][0], coordinates[route[i+1]][0]], [coordinates[route[i]][1], coordinates[route[i+1]][1]], 'bo-')
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red')
    plt.title("Traveling Salesman Route")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

n = 10  # Number of stations
coordinates = np.random.rand(n, 2)  # Random coordinates for the stations

# Use the first cooling scheme
best_route, best_distance = simulated_annealing(n, coordinates, cooling_scheme=1)
plot_route(best_route, coordinates)

print("Best Route:", best_route)
print("Best Distance:", best_distance)
