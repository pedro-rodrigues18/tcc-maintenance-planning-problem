import time
import numpy as np

from processing.optimization import Optimization


class DifferentialEvolution:
    def __init__(
        self,
        optimization: Optimization,
        obj_func: callable,
        bounds: np.ndarray,
        pop_size: int = 25,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.7,
        time_limit: int = 60 * 15,  # seconds
        tol: int = 1e-6,
    ) -> None:
        self.optimization = optimization
        self.obj_func = obj_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.time_limit = time_limit
        self.tol = tol

    def optimize(self):
        pop = np.random.randint(
            self.bounds[:, 0],
            self.bounds[:, 1] + 1,
            (self.pop_size, self.bounds.shape[0]),
        )

        fitness = np.array([self.obj_func(ind)[0] for ind in pop])

        # print("Initial pop: ", pop)
        # print("Initial fitness: ", fitness)
        # breakpoint()

        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds.", end="\r")
            if elapsed_time > self.time_limit:
                print(f"Maximum execution time reached: {elapsed_time:.2f} seconds.")
                break

            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros_like(fitness)
            for j in range(self.pop_size):
                idx = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = pop[idx]

                # print("Individual a: ", a)
                # print("Individual b: ", b)
                # print("Individual c: ", c)

                # Mutation
                mutant = np.clip(
                    a + self.mutation_factor * (b - c),
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                ).astype(int)

                # Crossover
                cross_points = (
                    np.random.rand(self.bounds.shape[0]) < self.crossover_prob
                )
                cross_points[np.random.randint(0, self.bounds.shape[0])] = True
                trial = np.where(cross_points, mutant, pop[j])

                _, pop_penalty = self.optimization._constraints_satisfied(pop[j])

                # Evaluate restrictions
                _, trial_penalty = self.optimization._constraints_satisfied(
                    trial.tolist()
                )

                # Evaluate solution
                trial_fitness = self.obj_func(trial, trial_penalty)[0]

                diff_penalty = abs(trial_penalty - pop_penalty)

                if diff_penalty < 1e-6:
                    if trial_fitness < fitness[j]:
                        new_pop[j] = trial
                        new_fitness[j] = trial_fitness
                    else:
                        new_pop[j] = pop[j]
                        new_fitness[j] = fitness[j]
                elif trial_penalty < pop_penalty:
                    new_pop[j] = trial
                    new_fitness[j] = trial_fitness
                else:
                    new_pop[j] = pop[j]
                    new_fitness[j] = fitness[j]

            # Convergence
            if np.all(np.abs(fitness - fitness.mean()) < self.tol):
                break

            pop = new_pop
            fitness = new_fitness

        print("\n\nPop: ", pop)
        print("Fitness: ", fitness)
        best_idx = fitness.argmin()
        best_individual = pop[best_idx]

        eval = self.obj_func(best_individual)

        print("Objective: ", eval[0])
        print("Mean risk: ", eval[1])
        print("Expected excess: ", eval[2])

        return self._format_solution(best_individual), fitness[best_idx]

    def _format_solution(self, best_individual):
        solution = []
        for i, start_time in enumerate(best_individual):
            intervention_name = self.optimization.problem.interventions[i].name
            solution.append(f"{intervention_name} {start_time}")
        return "\n".join(solution)
