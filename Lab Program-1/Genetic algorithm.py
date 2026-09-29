import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

input_size = 2
hidden_size = 4
output_size = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness(chromosome):
    w1 = chromosome[:input_size * hidden_size].reshape(input_size, hidden_size)
    w2 = chromosome[input_size * hidden_size:
                    input_size * hidden_size + hidden_size].reshape(hidden_size, output_size)

    b1 = chromosome[input_size * hidden_size + hidden_size:
                    input_size * hidden_size + hidden_size + hidden_size].reshape(1, hidden_size)

    b2 = chromosome[-1].reshape(1, output_size)

    h = sigmoid(np.dot(X, w1) + b1)
    output = sigmoid(np.dot(h, w2) + b2)

    error = np.mean((y.reshape(-1, 1) - output) ** 2)
    return 1 / (error + 0.001)

chromosome_length = (
    input_size * hidden_size +
    hidden_size * output_size +
    hidden_size +
    output_size
)

population_size = 50
generations = 100
mutation_rate = 0.1

population = np.random.uniform(-1, 1, (population_size, chromosome_length))

for generation in range(generations):
    fitness_values = np.array([fitness(c) for c in population])

    elite_indices = np.argsort(fitness_values)[-2:]
    new_population = [population[i].copy() for i in elite_indices]

    while len(new_population) < population_size:
        parent1 = population[np.argmax(fitness_values)]
        parent2 = population[np.random.randint(population_size)]

        point = np.random.randint(1, chromosome_length)
        child = np.concatenate((parent1[:point], parent2[point:]))

        mutation = np.random.rand(chromosome_length) < mutation_rate
        child[mutation] += np.random.normal(0, 0.5, np.sum(mutation))

        new_population.append(child)

    population = np.array(new_population)

best = population[np.argmax([fitness(c) for c in population])]

w1 = best[:input_size * hidden_size].reshape(input_size, hidden_size)
w2 = best[input_size * hidden_size:
          input_size * hidden_size + hidden_size].reshape(hidden_size, output_size)

b1 = best[input_size * hidden_size + hidden_size:
          input_size * hidden_size + hidden_size + hidden_size].reshape(1, hidden_size)

b2 = best[-1].reshape(1, output_size)

h = sigmoid(np.dot(X, w1) + b1)
output = sigmoid(np.dot(h, w2) + b2)

print("Predicted Output:")
print(np.round(output))
print("Actual Output:")
print(y.reshape(-1, 1))