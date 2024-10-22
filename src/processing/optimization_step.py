import numpy as np
from preprocessing.model.problem import Problem
from processing.optimization import Optimization
from processing.differential_evolution import DifferentialEvolution


class OptimizationStep:
    def __init__(self, problem: Problem):
        self.problem = problem

    def __call__(self):
        return self._optimization_step()

    def _optimization_step(self):
        """
        Optimize the problem using the Differential Evolution algorithm.

        Returns:
            - optimization_info: Optimization information, including the best solution and objective value.
        """

        optimization = Optimization(problem=self.problem)

        bounds = self._create_bounds()

        de = DifferentialEvolution(
            optimization=optimization,
            obj_func=optimization._build_objective_function,
            bounds=bounds,
            problem=self.problem,
            mutation_strategy="best/1",
        )

        solution, fitness = de.optimize()

        optimization_info = {
            "solution": solution,
            "fitness": fitness,
        }

        return optimization_info

    def _create_bounds(self):
        """
        Create the bounds for the optimization problem.
        """
        bounds = np.array(
            [[1, self.problem.time_horizon.time_steps]]
            * len(self.problem.interventions)
        )

        return bounds
