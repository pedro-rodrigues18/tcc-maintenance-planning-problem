import time
import numpy as np
import concurrent.futures

from processing.optimization import Optimization
from utils.log import log


class DifferentialEvolution:
    def __init__(
        self,
        file_name: str,
        optimization: Optimization,
        pop: np.ndarray,
        fitness: np.ndarray,
        obj_func: callable,
        bounds: np.ndarray,
        pop_size: int,
        mutation_factor: float = 0.8,
        rho: float = 0.5,
        time_limit: int = 60 * 1,  # seconds
        tol: int = 1e-6,
    ) -> None:
        self.file_name = file_name
        self.optimization = optimization
        self.pop = pop
        self.fitness = fitness
        self.obj_func = obj_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.rho = rho
        self.time_limit = time_limit
        self.tol = tol

    def exponential_crossover(self, x, v):
        n = len(x)
        k1 = np.random.randint(0, n)  # Random initial position
        d = np.random.geometric(
            p=1 - self.rho
        )  # Length based on exponential distribution
        k2 = k1 + d

        # Indicator vector
        r = np.zeros(n, dtype=int)
        for j in range(n):
            if k2 <= n and k1 <= j < k2:
                r[j] = 1
            elif k2 > n and (j < k2 % n or j >= k1):
                r[j] = 1

        # Recombination
        x_recomb = np.where(r == 1, v, x)
        return x_recomb

    def optimize(self):
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            # log(f"{self.file_name}", f"DE - Elapsed time: {elapsed_time:.2f} seconds.")
            if elapsed_time > self.time_limit:
                log(
                    f"{self.file_name}",
                    f"DE - Maximum execution time reached: {elapsed_time:.2f} seconds.",
                )
                break

            new_pop = np.zeros_like(self.pop)
            new_fitness = np.zeros_like(self.fitness)

            def evaluate_individual(j):
                idx = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.pop[idx]

                # Mutation
                mutant = np.clip(
                    a + self.mutation_factor * (b - c),
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                ).astype(int)

                # Crossover
                trial = self.exponential_crossover(self.pop[j], mutant)

                _, pop_penalty = self.optimization._constraints_satisfied(self.pop[j])
                _, trial_penalty = self.optimization._constraints_satisfied(
                    trial.tolist()
                )

                trial_fitness = self.obj_func(trial, trial_penalty)[0]

                diff_penalty = abs(trial_penalty - pop_penalty)

                if diff_penalty < 1e-6:
                    if trial_fitness < self.fitness[j]:
                        return trial, trial_fitness
                    else:
                        return self.pop[j], self.fitness[j]
                elif trial_penalty < pop_penalty:
                    return trial, trial_fitness
                else:
                    return self.pop[j], self.fitness[j]

            results = [evaluate_individual(j) for j in range(self.pop_size)]

            for j, (new_individual, new_ind_fitness) in enumerate(results):
                new_pop[j] = new_individual
                new_fitness[j] = new_ind_fitness

            if np.all(np.abs(self.fitness - self.fitness.mean()) < self.tol):
                break

            self.pop = new_pop
            self.fitness = new_fitness

        best_idx = self.fitness.argmin()
        best_individual = self.pop[best_idx]

        eval = self.obj_func(best_individual)

        log(f"{self.file_name}", f"Objective: {eval[0]}")
        log(f"{self.file_name}", f"Mean risk: {eval[1]}")
        log(f"{self.file_name}", f"Expected excess: {eval[2]}")

        return self._format_solution(best_individual), self.fitness[best_idx]

    def _format_solution(self, best_individual):
        solution = []
        for i, start_time in enumerate(best_individual):
            intervention_name = self.optimization.problem.interventions[i].name
            solution.append(f"{intervention_name} {start_time}")
        return "\n".join(solution)
