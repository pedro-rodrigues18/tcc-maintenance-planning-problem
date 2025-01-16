import time
import numpy as np

from processing.optimization import Optimization
from utils.format_solution import format_solution


class DifferentialEvolution:
    def __init__(
        self,
        start_time_execution: float,
        time_limit: int,
        file_name: str,
        optimization: Optimization,
        pop: np.ndarray,
        fitness: np.ndarray,
        obj_func: callable,
        bounds: np.ndarray,
        pop_size: int,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.5,
        tol: int = 1e-6,
    ) -> None:
        self.start_time_execution = start_time_execution
        self.file_name = file_name
        self.optimization = optimization
        self.pop = pop
        self.fitness = fitness
        self.obj_func = obj_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_limit = time_limit
        self.tol = tol
        self.max_iterations_without_improvement = 100

    def _exponential_crossover(self, x, v):
        n = len(x)
        k1 = np.random.randint(0, n)  # Random initial position
        d = np.random.geometric(
            p=1 - self.crossover_rate
        )  # Length based on exponential distribution
        k2 = k1 + d

        r = np.zeros(n, dtype=int)
        for j in range(n):
            if k2 <= n and k1 <= j < k2:
                r[j] = 1
            elif k2 > n and (j < k2 % n or j >= k1):
                r[j] = 1

        x_recomb = np.where(r == 1, v, x)
        return x_recomb

    def _evaluate_individual(self, j):
        # Mutation DE/best/1
        best_individual = self.pop[self.fitness.argmin()]
        x_r1, x_r2 = np.random.choice(self.pop_size, 2, replace=False)
        mutant = best_individual + self.mutation_rate * (
            self.pop[x_r1] - self.pop[x_r2]
        )
        mutant = np.round(mutant).astype(int)
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

        trial = self._exponential_crossover(self.pop[j], mutant)

        _, pop_penalty = self.optimization._constraints_satisfied(self.pop[j])
        _, trial_penalty = self.optimization._constraints_satisfied(trial.tolist())

        trial_fitness = self.obj_func(trial, trial_penalty)[0]

        diff_penalty = abs(trial_penalty - pop_penalty)

        if trial_fitness < self.fitness[j] and diff_penalty < 1e-6:
            return trial, trial_fitness
        else:
            return self.pop[j], self.fitness[j]

    def _reset_population(self, best_individual):
        self.pop = np.random.randint(
            self.bounds[:, 0],
            self.bounds[:, 1] + 1,
            (self.pop_size - 1, self.bounds.shape[0]),
        )
        self.pop = np.vstack((self.pop, best_individual))
        penalty = np.array(
            [self.optimization._constraints_satisfied(ind)[1] for ind in self.pop]
        )
        self.fitness = np.array(
            [self.obj_func(ind, pen)[0] for ind, pen in zip(self.pop, penalty)]
        )

    def optimize(self):
        remaining_time = self.time_limit - (time.time() - self.start_time_execution)
        iterations_without_improvement = 0
        while remaining_time > 0:
            new_pop = np.zeros_like(self.pop)
            new_fitness = np.zeros_like(self.fitness)

            results = [self._evaluate_individual(j) for j in range(self.pop_size)]

            current_best_fitness = self.fitness.min()
            new_best_fitness = min(new_ind_fitness for _, new_ind_fitness in results)
            if new_best_fitness < current_best_fitness:
                # print(f"New best fitness: {new_best_fitness}", end="\r")
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            for j, (new_individual, new_ind_fitness) in enumerate(results):
                new_pop[j] = new_individual
                new_fitness[j] = new_ind_fitness

            self.pop = new_pop
            self.fitness = new_fitness

            if iterations_without_improvement > self.max_iterations_without_improvement:
                best_individual = self.pop[self.fitness.argmin()]
                self._reset_population(best_individual)
                iterations_without_improvement = 0

            remaining_time = self.time_limit - (time.time() - self.start_time_execution)

        best_idx = self.fitness.argmin()
        best_individual = self.pop[best_idx]

        return (
            format_solution(self.optimization.problem.interventions, best_individual),
            self.fitness[best_idx],
        )
