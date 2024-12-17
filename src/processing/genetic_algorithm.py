import time
import numpy as np
import concurrent.futures

from preprocessing.model.problem import Problem
from utils.log import log


class GeneticAlgorithm:
    def __init__(
        self,
        file_name: str,
        problem: Problem,
        pop: np.ndarray,
        pop_size: int,
        fitness: np.ndarray,
        obj_func: callable,
        mutation_rate: float = 0.1,
        time_limit: int = 60 * 10,  # seconds
    ):
        self.file_name = file_name
        self.problem = problem
        self.pop = pop
        self.pop_size = pop_size
        self.fitness = fitness
        self.obj_func = obj_func
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit

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
            # log(f"{self.file_name}", f"GA - Elapsed time: {elapsed_time:.2f} seconds.")
            if elapsed_time > self.time_limit:
                log(
                    f"{self.file_name}",
                    f"GA - Maximum execution time reached: {elapsed_time:.2f} seconds.",
                )
                return self.pop, self.fitness

            new_pop = np.zeros_like(self.pop)

            def evaluate_individual(i):
                """
                Evaluate and generate a new individual.
                """
                idx = np.random.choice(self.pop_size, 2, replace=False)
                a, b = self.pop[idx]

                child = self.reproduce(a, b)

                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, len(child))
                    child[mutation_idx] = np.random.randint(
                        1, self.problem.time_horizon.time_steps
                    )

                fitness_child = self.obj_func(child)[0]

                if fitness_child < self.fitness[i]:
                    return child, fitness_child
                else:
                    return self.pop[i], self.fitness[i]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(evaluate_individual, range(self.pop_size)))

            for i, (new_individual, new_fitness) in enumerate(results):
                new_pop[i] = new_individual
                self.fitness[i] = new_fitness

            if np.all(new_pop == self.pop):
                log(f"{self.file_name}", "GA - Converged")
                log(
                    f"{self.file_name}",
                    f"GA - Solution: {self.pop[np.argmin(self.fitness)]}",
                )
                log(f"{self.file_name}", f"GA - Fitness: {np.min(self.fitness)}")
                return self.pop, self.fitness

            self.pop = new_pop
