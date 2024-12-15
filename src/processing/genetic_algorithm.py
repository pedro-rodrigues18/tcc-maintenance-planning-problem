import time
import numpy as np


class GeneticAlgorithm:
    def __init__(
        self, problem, pop, pop_size, fitness, obj_func, mutation_rate, crossover_rate
    ):
        self.problem = problem
        self.pop = pop
        self.pop_size = pop_size
        self.fitness = fitness
        self.obj_func = obj_func
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_limit: int = 60 * 10  # seconds

    def reproduce(self, a, b):
        """
        Reproduce two individuals.
        """
        n = len(a)
        c = np.random.randint(1, n)
        return np.concatenate((a[:c], b[c:]), axis=0)

    def optimize(self):
        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds.", end="\r")
            if elapsed_time > self.time_limit:
                print(f"Maximum execution time reached: {elapsed_time:.2f} seconds.")
                return self.pop, self.fitness

            new_pop = np.zeros_like(self.pop)

            for i in range(self.pop_size):
                idx = np.random.choice(self.pop_size, 2, replace=False)
                a, b = self.pop[idx]

                child = self.reproduce(a, b)

                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, len(child))
                    child[mutation_idx] = np.random.randint(
                        1, self.problem.time_horizon.time_steps
                    )

                fitness_child = self.obj_func(child)[0]

                print("Fitness child: ", fitness_child)

                if fitness_child < self.fitness[i]:
                    new_pop[i] = child
                    self.fitness[i] = fitness_child
                else:
                    new_pop[i] = self.pop[i]

            if np.all(new_pop == self.pop):
                print("Converged")
                print("Best solution: ", self.pop[np.argmin(self.fitness)])
                print("Best fitness: ", np.min(self.fitness))
                return self.pop, self.fitness

            self.pop = new_pop
