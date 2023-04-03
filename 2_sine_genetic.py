import numpy as np
import random
import matplotlib.pyplot as plt

# Define the problem
x_data = np.linspace(-1, 1, 100)
y_data = x_data * np.sin(10 * np.pi * x_data) + 2

# Define the fitness function
def fitness(chromosome):
    y_pred = chromosome[0] * x_data * np.sin(chromosome[1] * np.pi * x_data) + 2
    return np.sum((y_pred - y_data)**2)

# Define the chromosome
chromosome_length = 2
min_gene = 0.01
max_gene = 10

# Initialize the population
population_size = 50
population = [np.random.uniform(min_gene, max_gene, chromosome_length) for i in range(population_size)]

# Define the genetic operators
def tournament_selection(population, fitness):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    fitnesses = [fitness(chromosome) for chromosome in tournament]
    return tournament[np.argmin(fitnesses)]

def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, chromosome_length - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def gaussian_mutation(chromosome, mutation_rate):
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.normal(0, 1)
            chromosome[i] = max(min_gene, min(chromosome[i], max_gene))
    return chromosome

# Define the genetic algorithm parameters
num_generations = 100
mutation_rate = 0.1

# Run the genetic algorithm
for generation in range(num_generations):
    # Evaluate the fitness of the population
    fitnesses = [fitness(chromosome) for chromosome in population]
    best_chromosome = population[np.argmin(fitnesses)]
    print("Generation:", generation, "Best fitness:", fitness(best_chromosome), "Best chromosome:", best_chromosome)

    # Select parents for reproduction
    parents = [tournament_selection(population, fitness) for i in range(population_size)]

    # Perform crossover and mutation to create offspring
    offspring = []
    for i in range(population_size // 2):
        parent1 = parents[random.randint(0, population_size - 1)]
        parent2 = parents[random.randint(0, population_size - 1)]
        child1, child2 = single_point_crossover(parent1, parent2)
        child1 = gaussian_mutation(child1, mutation_rate)
        child2 = gaussian_mutation(child2, mutation_rate)
        offspring.append(child1)
        offspring.append(child2)

    # Replace the old population with the new offspring population
    population = offspring

# Generate the curve
y_curve = best_chromosome[0] * x_data * np.sin(best_chromosome[1] * np.pi * x_data) + 2
print("Best curve:", y_curve)

# Plot the data points and the curve
plt.plot(x_data, y_data, label = 'actual')
plt.plot(x_data, y_curve, label = 'predicted')
plt.legend()
plt.show()
