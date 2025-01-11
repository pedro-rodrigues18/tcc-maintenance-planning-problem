import time
import numpy as np
from preprocessing.model.problem import Problem
from processing.optimization import Optimization
from processing.gurobi import Gurobi
from processing.differential_evolution import DifferentialEvolution
from utils.log import log
from utils.format_solution import format_solution


class OptimizationStep:
    def __init__(
        self,
        start_time_execution: float,
        time_limit: int,
        problem: Problem,
        file_name: str = None,
        pop_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        rho: float = 0.5,
    ) -> None:
        self.start_time_execution = start_time_execution
        self.time_limit = time_limit
        self.problem = problem
        self.file_name = file_name
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rho = rho

    def __call__(self):
        return self._optimization_step()

    def _optimization_step(self) -> dict[str, np.ndarray]:
        """
        Optimize the problem using the Gurobi and Differential Evolution algorithm.

        Returns:
            - optimization_info: Optimization information, including the best solution and objective value.
        """
        # ------ Gurobi Algorithm ------
        remaining_time = self.time_limit - (time.time() - self.start_time_execution)

        log(self.file_name, "Starting Gurobi algorithm...")

        gb = Gurobi(time_limit=remaining_time, problem=self.problem)

        gurobi_solution = gb.optimize()

        gurobi_solution = gb.get_solution()
        gurobi_objective_value = gb.get_objective_value()
        solution_formated = format_solution(self.problem.interventions, gurobi_solution)

        log(self.file_name, "Gurobi algorithm completed.")
        # log(self.file_name, f"Gurobi solution:\n{solution_formated}")

        # ------ Differential Evolution Algorithm ------
        remaining_time = self.time_limit - (time.time() - self.start_time_execution)

        if remaining_time <= 0:
            log(self.file_name, "Time limit reached.")
            log(self.file_name, "Returning Gurobi solution.")
            log(self.file_name, f"Gurobi objective value: {gurobi_objective_value}")
            return {
                "solution": solution_formated,
                "objective_value": gurobi_objective_value,
            }

        log(self.file_name, "Starting Differential Evolution algorithm...")

        optimization = Optimization(problem=self.problem)

        bounds = self._create_bounds()

        # Create the initial population based on the gurobi solution
        pop = np.zeros((self.pop_size - 1, bounds.shape[0]))
        probabity_change_solution = 0.75
        for i in range(len(gurobi_solution)):
            if np.random.rand() < probabity_change_solution:
                pop[:, i] = np.random.randint(
                    bounds[i][0], bounds[i][1], size=self.pop_size - 1
                )
            else:
                pop[:, i] = np.random.randint(
                    bounds[i][0], bounds[i][1], size=self.pop_size - 1
                )

        # Add the Gurobi solution to the population
        pop = np.vstack(
            (pop, gurobi_solution),
        )

        pop_penalty = [optimization._constraints_satisfied(ind)[1] for ind in pop]

        fitness = np.array(
            [
                optimization._objective_function(ind, pop_penalty)[0]
                for (ind, pop_penalty) in zip(pop, pop_penalty)
            ]
        )

        sorted_idx = np.argsort(fitness)
        fitness = fitness[sorted_idx]
        pop = pop[sorted_idx]

        de = DifferentialEvolution(
            start_time_execution=self.start_time_execution,
            time_limit=remaining_time,
            file_name=self.file_name,
            optimization=optimization,
            obj_func=optimization._objective_function,
            pop_size=self.pop_size,
            pop=pop,
            fitness=fitness,
            bounds=bounds,
            mutation_rate=self.mutation_rate,
        )

        solution, fitness = de.optimize()

        log(self.file_name, "Differential Evolution algorithm completed.")
        # log(self.file_name, f"DE solution:\n{solution}")
        log(self.file_name, f"DE objective value: {fitness}")

        optimization_info = {
            "solution": solution,
            "objective_value": fitness,
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
