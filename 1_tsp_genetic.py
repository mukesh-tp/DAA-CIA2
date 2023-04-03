import numpy as np
import matplotlib.pyplot as plt

# Define fitness function
def fitness(solution, distances):
    total_distance = 0
    for i in range(len(solution)-1):
        total_distance += distances[solution[i], solution[i+1]]
    return 1 / total_distance

# Define crossover function
def crossover(parent1, parent2):
    # Choose random crossover points
    point1 = np.random.randint(0, len(parent1))
    point2 = np.random.randint(point1, len(parent1))

    # Create child
    child = [-1] * len(parent1)
    child[point1:point2+1] = parent1[point1:point2+1]
    remaining = [gene for gene in parent2 if gene not in child]
    child[:point1] = remaining[:point1]
    child[point2+1:] = remaining[point1:]

    return child

# Define mutation function
def mutate(solution):
    # Swap two random cities
    idx1, idx2 = np.random.choice(len(solution), 2, replace=False)
    solution[idx1], solution[idx2] = solution[idx2], solution[idx1]

    return solution

# Define genetic algorithm function
def genetic_algorithm(distances, population_size=100, generations=1000):
    num_cities = distances.shape[0]
    population = [list(np.random.permutation(num_cities)) for _ in range(population_size)]

    best_fitnesses = []
    best_solutions = []

    for generation in range(generations):
        # Evaluate fitness of population
        fitness_values = [fitness(solution, distances) for solution in population]

        # Select parents for mating
        parent_indices = np.random.choice(population_size, size=population_size, replace=True, p=fitness_values/np.sum(fitness_values))
        parents = [population[i] for i in parent_indices]

        # Create next generation through crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            child1 = crossover(parents[i], parents[i+1])
            child2 = crossover(parents[i+1], parents[i])
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

        # Store best solution and fitness
        best_solution = max(population, key=lambda solution: fitness(solution, distances))
        best_fitness = fitness(best_solution, distances)
        best_fitnesses.append(best_fitness)
        best_solutions.append(best_solution)

    # Find best solution
    best_solution = max(population, key=lambda solution: fitness(solution, distances))
    best_fitness = fitness(best_solution, distances)

    return best_solution, best_fitness, best_solutions, best_fitnesses

# Generate random distances between cities
num_cities = 10
cities = np.random.rand(num_cities, 2)
distances = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i, num_cities):
        distance = np.linalg.norm(cities[i] - cities[j])
        distances[i, j] = distance
        distances[j, i] = distance

# Run genetic algorithm
best_solution, best_fitness, best_solutions, best_fitnesses = genetic_algorithm(distances, population_size=100, generations=1000)


# Plot the cities
plt.subplot(1, 2, 1)
plt.plot(cities[:, 0], cities[:, 1], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cities')

# Plot the best path
plt.subplot(1, 2, 2)
best_route = np.array([cities[i] for i in best_solution])
plt.plot(best_route[:, 0], best_route[:, 1], 'o-')
plt.plot([best_route[-1, 0], best_route[0, 0]], [best_route[-1, 1], best_route[0, 1]], 'o-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best Route')

plt.tight_layout()
plt.show()