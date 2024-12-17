import numpy as np
from preprocessing.model.problem import Problem
from processing.optimization import Optimization
from processing.differential_evolution import DifferentialEvolution
from processing.genetic_algorithm import GeneticAlgorithm


class OptimizationStep:
    def __init__(self, problem: Problem, file_name: str = None):
        self.problem = problem
        self.file_name = file_name

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
        pop_size = 100

        pop = np.random.randint(
            bounds[:, 0],
            bounds[:, 1] + 1,
            (pop_size, bounds.shape[0]),
        )

        fitness = np.array(
            [optimization._build_objective_function(ind)[0] for ind in pop]
        )

        copy_pop = np.copy(pop)
        copy_fitness = np.copy(fitness)

        ga = GeneticAlgorithm(
            file_name=self.file_name,
            problem=self.problem,
            pop=copy_pop,
            pop_size=pop_size,
            fitness=copy_fitness,
            obj_func=optimization._build_objective_function,
            mutation_rate=0.1,
        )

        new_population, new_fitness = ga.optimize()

        # Sort the fitness and population
        sorted_idx = np.argsort(new_fitness)
        new_fitness = new_fitness[sorted_idx]
        new_population = new_population[sorted_idx]

        middle = len(pop) // 2
        pop = np.concatenate((pop[:middle], new_population[:middle]), axis=0)
        fitness = np.concatenate((fitness[:middle], new_fitness[:middle]), axis=0)

        de = DifferentialEvolution(
            file_name=self.file_name,
            optimization=optimization,
            obj_func=optimization._build_objective_function,
            pop_size=pop_size,
            pop=pop,
            fitness=fitness,
            bounds=bounds,
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
