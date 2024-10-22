import numpy as np
import time
from typing import Tuple, List, Callable
import logging


class DifferentialEvolution:
    def __init__(
        self,
        obj_func: Callable,
        bounds: np.ndarray,
        pop_size: int = 10000,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.7,
        mutation_strategy: str = "rand/1",
        adaptation_rate: float = 0.1,
        time_limit: float = 60 * 15,
        tol: float = 1e-6,
        max_stagnation: int = 20,
        optimization=None,
        problem=None,
    ):
        """
        Differential Evolution algorithm with adaptive parameters and multiple strategies.

        Args:
            obj_func: Objective function to minimize
            bounds: Array of bounds for each parameter (n_params, 2)
            pop_size: Population size
            mutation_factor: Initial mutation factor (F)
            crossover_prob: Initial crossover probability (CR)
            mutation_strategy: One of ['rand/1', 'best/1', 'rand/2', 'best/2']
            adaptation_rate: Rate for parameter adaptation
            time_limit: Maximum execution time in seconds
            tol: Convergence tolerance
            max_stagnation: Maximum generations without improvement
        """
        self.obj_func = obj_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.mutation_strategy = mutation_strategy
        self.adaptation_rate = adaptation_rate
        self.time_limit = time_limit
        self.tol = tol
        self.max_stagnation = max_stagnation
        self.optimization = optimization
        self.problem = problem

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _adapt_parameters(self, success_rate: float) -> None:
        """
        Adapt mutation factor and crossover probability based on success rate
        """
        if success_rate > 0.2:
            self.mutation_factor *= 1 + self.adaptation_rate
            self.crossover_prob *= 1 + self.adaptation_rate
        else:
            self.mutation_factor *= 1 - self.adaptation_rate
            self.crossover_prob *= 1 - self.adaptation_rate

        # Keep parameters in valid ranges
        self.mutation_factor = np.clip(self.mutation_factor, 0.1, 1.0)
        self.crossover_prob = np.clip(self.crossover_prob, 0.1, 0.9)

    def _mutate(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Apply mutation according to selected strategy
        """
        if self.mutation_strategy == "rand/1":
            idx = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = pop[idx]
            mutant = a + self.mutation_factor * (b - c)

        elif self.mutation_strategy == "best/1":
            best_idx = fitness.argmin()
            idx = np.random.choice(self.pop_size, 2, replace=False)
            mutant = pop[best_idx] + self.mutation_factor * (pop[idx[0]] - pop[idx[1]])

        elif self.mutation_strategy == "rand/2":
            idx = np.random.choice(self.pop_size, 5, replace=False)
            a, b, c, d, e = pop[idx]
            mutant = a + self.mutation_factor * (b - c) + self.mutation_factor * (d - e)

        elif self.mutation_strategy == "best/2":
            best_idx = fitness.argmin()
            idx = np.random.choice(self.pop_size, 4, replace=False)
            mutant = (
                pop[best_idx]
                + self.mutation_factor * (pop[idx[0]] - pop[idx[1]])
                + self.mutation_factor * (pop[idx[2]] - pop[idx[3]])
            )

        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1]).astype(int)

    def optimize(self) -> Tuple[str, float]:
        # Initialize population
        pop = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.bounds.shape[0])
        ).astype(int)

        # Evaluate initial population
        fitness = np.array([self.obj_func(ind) for ind in pop])
        best_fitness = fitness.min()
        stagnation_counter = 0
        success_counter = 0

        # Archive for diversity maintenance
        archive = []
        archive_size = self.pop_size

        start_time = time.time()
        generation = 0

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.time_limit:
                self.logger.info(
                    f"Maximum execution time reached: {elapsed_time:.2f} seconds"
                )
                break

            if stagnation_counter >= self.max_stagnation:
                self.logger.info(
                    f"Optimization stagnated for {self.max_stagnation} generations"
                )
                break

            new_pop = np.zeros_like(pop)
            success_in_generation = 0

            for j in range(self.pop_size):
                # Generate mutant
                mutant = self._mutate(pop, fitness, j)

                # Crossover
                cross_points = (
                    np.random.rand(self.bounds.shape[0]) < self.crossover_prob
                )
                cross_points[np.random.randint(0, self.bounds.shape[0])] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Check constraints
                if not self.optimization._constraints_satisfied(trial.tolist()):
                    new_pop[j] = pop[j]
                    continue

                # Evaluate trial solution
                trial_fitness = self.obj_func(trial)

                # Selection with archive
                if trial_fitness < fitness[j]:
                    new_pop[j] = trial
                    fitness[j] = trial_fitness
                    success_in_generation += 1

                    # Update archive
                    if len(archive) < archive_size:
                        archive.append(pop[j])
                    else:
                        random_idx = np.random.randint(0, archive_size)
                        archive[random_idx] = pop[j]
                else:
                    new_pop[j] = pop[j]

            # Update success rate and adapt parameters
            success_rate = success_in_generation / self.pop_size
            self._adapt_parameters(success_rate)

            # Check for improvement
            current_best = fitness.min()
            if current_best < best_fitness:
                best_fitness = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Check convergence
            if np.all(np.abs(fitness - fitness.mean()) < self.tol):
                self.logger.info("Convergence reached")
                break

            pop = new_pop
            generation += 1

            # Log progress periodically
            if generation % 100 == 0:
                self.logger.info(
                    f"Generation {generation}: Best fitness = {best_fitness:.6f}"
                )

        best_idx = fitness.argmin()
        best_individual = pop[best_idx]

        self.logger.info(f"Optimization completed after {generation} generations")
        return self._format_solution(best_individual), fitness[best_idx]

    def _format_solution(self, best_individual: np.ndarray) -> str:
        solution = []
        for i, start_time in enumerate(best_individual):
            intervention_name = self.problem.interventions[i].name
            solution.append(f"{intervention_name} {start_time}")
        return "\n".join(solution)
