import numpy as np
import matplotlib.pyplot as plt

# Define the fitness function to evaluate the sine wave
def fitness_function(x):
    y = np.sin(x)
    return np.sum(y)

# Define the cultural algorithm parameters
pop_size = 50  # population size
num_generations = 100  # number of generations
mutation_rate = 0.01  # mutation rate
selection_size = 10  # number of individuals to select for breeding
cultural_rate = 0.1  # cultural learning rate
cultural_pool_size = 5  # size of the cultural pool

# Initialize the population with random values in the range [0, 2pi]
population = np.random.uniform(low=0.0, high=2*np.pi, size=(pop_size,))

# Initialize the cultural pool with the top individuals from the initial population
fitness = np.array([fitness_function(x) for x in population])
cultural_pool_indices = np.argsort(fitness)[-cultural_pool_size:]
cultural_pool = population[cultural_pool_indices]

# Run the cultural algorithm for the specified number of generations
for generation in range(num_generations):

    # Evaluate the fitness of each individual in the population
    fitness = np.array([fitness_function(x) for x in population])

    # Select the best individuals for breeding
    selected_indices = np.argsort(fitness)[-selection_size:]
    selected_population = population[selected_indices]

    # Learn from the cultural pool and update the population
    for i in range(pop_size):
        if np.random.rand() < cultural_rate:
            cultural_individual = np.random.choice(cultural_pool)
            population[i] = cultural_individual

    # Create new offspring by breeding the selected individuals
    offspring = np.empty((pop_size,))
    for i in range(pop_size):
        parent1, parent2 = np.random.choice(selected_population, size=2, replace=False)
        offspring[i] = np.random.uniform(low=0.0, high=2*np.pi) if np.random.rand() < mutation_rate \
            else (parent1 + parent2) / 2

    # Replace the old population with the new offspring
    population = offspring

    # Update the cultural pool with the best individuals from the current population
    fitness = np.array([fitness_function(x) for x in population])
    cultural_pool_indices = np.argsort(fitness)[-cultural_pool_size:]
    cultural_pool = population[cultural_pool_indices]

# Select the best individual from the final population
fitness = np.array([fitness_function(x) for x in population])
best_index = np.argmax(fitness)
best_individual = population[best_index]
best_fitness = fitness[best_index]

# Plot the best individual
x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(x)
plt.plot(x, y, label='True function')
y = np.sin(best_individual)
plt.plot(best_individual, y, 'ro', label='Best individual')
plt.legend()
plt.show()

print(f'Best individual: {best_individual}')
print(f'Best fitness: {best_fitness}')
