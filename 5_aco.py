import random

NUM_ANTS = 10
NUM_ITERATIONS = 50
EVAPORATION_RATE = 0.5
ALPHA = 1
BETA = 2
Q = 1
INITIAL_PHEROMONE = 0.1
n = 5

graph = [
    [0, 3, 4, 2, 7],
    [3, 0, 4, 6, 3],
    [4, 4, 0, 5, 8],
    [2, 6, 5, 0, 6],
    [7, 3, 8, 6, 0]
]

heuristic = [[1.0 / graph[i][j] if graph[i][j] > 0 else 0 for j in range(len(graph))] for i in range(len(graph))]

pheromone = [[INITIAL_PHEROMONE for j in range(len(graph))] for i in range(len(graph))]

def choose_next_node(current_node, unvisited_nodes):
    
    probabilities = [0.0] * len(unvisited_nodes)
    total_probability = 0.0
    for i, node in enumerate(unvisited_nodes):
        probabilities[i] = pheromone[current_node][node] * ALPHA * heuristic[current_node][node] * BETA
        total_probability += probabilities[i]
    
    
    rand = random.uniform(0, total_probability)
    cumulative_probability = 0.0
    for i, node in enumerate(unvisited_nodes):
        cumulative_probability += probabilities[i]
        if cumulative_probability >= rand:
            return node

def update_pheromone(trails):
    for i in range(len(pheromone)):
        for j in range(len(pheromone)):
            pheromone[i][j] *= (1 - EVAPORATION_RATE)
            for trail in trails:
                if (i, j) in trail or (j, i) in trail:
                    pheromone[i][j] += Q / graph[i][j]

best_path = None
best_distance = float('inf')
for iteration in range(NUM_ITERATIONS):
    ant_positions = [random.randint(0, len(graph) - 1) for i in range(NUM_ANTS)]
    
    trails = [[] for i in range(NUM_ANTS)]
    
    for step in range(len(graph) - 1):
        for i in range(NUM_ANTS):
            current_node = ant_positions[i]
            unvisited_nodes = [j for j in range(len(graph)) if j != current_node and j not in trails[i]]
            if len(unvisited_nodes) == 0:
                continue
            next_node = choose_next_node(current_node, unvisited_nodes)
            ant_positions[i] = next_node
            trails[i].append((current_node, next_node))
            
            distances = []
            for i in range(NUM_ANTS):
                distance = 0
                for edge in trails[i]:
                    distance += graph[edge[0]][edge[1]]
                    distances.append(distance)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = trails[i]
                        
                        update_pheromone(trails)
                        print("Iteration", iteration, ":", "Best distance =", best_distance, "Best path =", best_path)
print("Final best path:", best_path)
print("Final shortest distance:", best_distance)
