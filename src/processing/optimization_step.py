import numpy as np
from preprocessing.model.problem import Problem
from processing.optimization import Optimization
from processing.differential_evolution import DifferentialEvolution
from processing.genetic_algorithm import GeneticAlgorithm
from utils.log import log


class OptimizationStep:
    def __init__(
        self,
        problem: Problem,
        file_name: str = None,
        pop_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        mutation_factor: float = 0.8,
        rho: float = 0.5,
    ) -> None:
        self.problem = problem
        self.file_name = file_name
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_factor = mutation_factor
        self.rho = rho

    def __call__(self):
        return self._optimization_step()

    def _optimization_step(self) -> dict[str, np.ndarray]:
        """
        Optimize the problem using the Differential Evolution algorithm.

        Returns:
            - optimization_info: Optimization information, including the best solution and objective value.
        """

        optimization = Optimization(problem=self.problem)

        bounds = self._create_bounds()

        pop = np.random.randint(
            bounds[:, 0],
            bounds[:, 1] + 1,
            (self.pop_size, bounds.shape[0]),
        )

        pop_penalty = [optimization._constraints_satisfied(ind)[1] for ind in pop]

        fitness = np.array(
            [
                optimization._build_objective_function(ind, pop_penalty)[0]
                for (ind, pop_penalty) in zip(pop, pop_penalty)
            ]
        )

        copy_pop = np.copy(pop)
        copy_fitness = np.copy(fitness)

        ga = GeneticAlgorithm(
            file_name=self.file_name,
            problem=self.problem,
            optimization=optimization,
            pop=copy_pop,
            pop_size=self.pop_size,
            fitness=copy_fitness,
            obj_func=optimization._build_objective_function,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
        )

        new_population, new_fitness = ga.optimize()

        # Sort the fitness and population
        sorted_idx = np.argsort(new_fitness)
        new_fitness = new_fitness[sorted_idx]
        new_population = new_population[sorted_idx]

        middle = self.pop_size // 2
        if self.pop_size % 2 == 0:
            pop = np.concatenate((pop[:middle], new_population[:middle]), axis=0)
            fitness = np.concatenate((fitness[:middle], new_fitness[:middle]), axis=0)
        else:
            pop = np.concatenate((pop[:middle], new_population[: middle + 1]), axis=0)
            fitness = np.concatenate(
                (fitness[:middle], new_fitness[: middle + 1]), axis=0
            )

        de = DifferentialEvolution(
            file_name=self.file_name,
            optimization=optimization,
            obj_func=optimization._build_objective_function,
            pop_size=self.pop_size,
            pop=pop,
            fitness=fitness,
            bounds=bounds,
            mutation_factor=self.mutation_factor,
            rho=self.rho,
        )

        solution, fitness = de.optimize()

        optimization_info = {
            "solution": solution,
            "fitness": fitness,
        }

        return optimization_info

    def _create_bounds(self) -> np.ndarray:
        """
        Create the bounds for the optimization problem.
        """
        bounds = np.array(
            [
                [1, self.problem.interventions[i].tmax]
                for i in range(len(self.problem.interventions))
            ]
        )

        return bounds
