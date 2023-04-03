import random
import math
import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    return x * math.sin(10 * math.pi * x) + 2

def fitness(position):
    return -abs(target_function(position))

def pso(num_particles, num_iterations):
    
    particles = []
    velocities = []
    for i in range(num_particles):
        position = random.uniform(-1, 1)
        particles.append(position)
        velocity = random.uniform(-1, 1)
        velocities.append(velocity)
    
    personal_best_positions = particles.copy()
    personal_best_fitnesses = [fitness(position) for position in particles]
    
    global_best_index = personal_best_fitnesses.index(max(personal_best_fitnesses))
    global_best_position = personal_best_positions[global_best_index]
    global_best_fitness = personal_best_fitnesses[global_best_index]
    
    for i in range(num_iterations):
        for j in range(num_particles):
            
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            velocity = velocities[j] + r1 * (personal_best_positions[j] - particles[j]) + r2 * (global_best_position - particles[j])
            velocity = max(velocity, -1)
            velocity = min(velocity, 1)
            velocities[j] = velocity
            
            position = particles[j] + velocity
            position = max(position, -1)
            position = min(position, 1)
            particles[j] = position
            
            fitness_value = fitness(position)
            if fitness_value > personal_best_fitnesses[j]:
                personal_best_positions[j] = position
                personal_best_fitnesses[j] = fitness_value
            
            if fitness_value > global_best_fitness:
                global_best_position = position
                global_best_fitness = fitness_value
                
        print("Iteration:", i + 1, "Global Best Position:", global_best_position, "Global Best Fitness:", global_best_fitness)
    
    # Plot the function
    x = np.linspace(-1, 1, 500)
    y = [target_function(xi) for xi in x]
    plt.plot(x, y)
    
    # Plot the best position found by PSO
    plt.plot(global_best_position, target_function(global_best_position), 'ro')
    
    plt.show()
    
    return global_best_position, global_best_fitness

pso(20, 50)
