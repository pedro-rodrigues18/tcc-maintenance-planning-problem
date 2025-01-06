import numpy as np
from preprocessing.model.problem import Problem
from processing.optimization import Optimization
from processing.differential_evolution import DifferentialEvolution
from processing.genetic_algorithm import GeneticAlgorithm
from utils.log import log
import gurobipy as gp
from gurobipy import GRB


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
        model = gp.Model()

        # Create variables
        x = {
            i: {
                t: model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")
                for t in range(1, self.problem.time_horizon.time_steps + 1)
            }
            for i in range(len(self.problem.interventions))
        }

        # Set objective
        expr = gp.LinExpr()

        for t in range(1, self.problem.time_horizon.time_steps + 1):
            for s in range(self.problem.scenarios[t - 1]):
                for i in range(len(self.problem.interventions)):
                    for st in range(1, self.problem.interventions[i].tmax + 1):
                        try:
                            expr.addTerms(
                                self.problem.interventions[i].risk[str(t)][str(st)][s]
                                / (
                                    self.problem.time_horizon.time_steps
                                    * self.problem.scenarios[t - 1]
                                ),
                                x[i][st],
                            )
                        except KeyError:
                            pass

        model.setObjective(
            expr,
            GRB.MINIMIZE,
        )

        # Add constraints

        # Every intervention must be performed exactly once
        for i in range(len(self.problem.interventions)):
            model.addConstr(
                gp.quicksum(
                    x[i][t] for t in range(1, self.problem.interventions[i].tmax + 1)
                )
                == 1
            )

        # Resource constraints
        for r in range(len(self.problem.resources)):
            for t in range(1, self.problem.time_horizon.time_steps + 1):
                expr = gp.LinExpr()

                for i in range(len(self.problem.interventions)):
                    for st in range(1, self.problem.interventions[i].tmax + 1):
                        try:
                            expr.addTerms(
                                self.problem.interventions[i].resource_workload[
                                    self.problem.resources[r].name
                                ][str(t)][str(st)],
                                x[i][st],
                            )
                        except KeyError:
                            pass

                model.addConstr(self.problem.resources[r].min[t - 1] <= expr)
                model.addConstr(expr <= self.problem.resources[r].max[t - 1])

        # Exclusion constraints

        for e in self.problem.exclusions:
            i1, i2, season = (
                e.interventions[0],
                e.interventions[1],
                e.season,
            )

            i1 = next(
                i
                for i, intervention in enumerate(self.problem.interventions)
                if intervention.name == i1
            )
            i2 = next(
                i
                for i, intervention in enumerate(self.problem.interventions)
                if intervention.name == i2
            )

            for t in season.duration:
                expr = gp.LinExpr()

                for st in range(1, self.problem.interventions[i1].tmax + 1):
                    if st <= t <= st + self.problem.interventions[i1].delta[st - 1] - 1:
                        expr.add(x[i1][st])
                for st in range(1, self.problem.interventions[i2].tmax + 1):
                    if st <= t <= st + self.problem.interventions[i2].delta[st - 1] - 1:
                        expr.add(x[i2][st])

                model.addConstr(expr <= 1)

        model.optimize()

        solution = []

        for i in range(len(self.problem.interventions)):
            for t in range(1, self.problem.time_horizon.time_steps + 1):
                if x[i][t].x > 0.5:
                    solution.append(f"{self.problem.interventions[i].name} {t}")

        optimization_info = {
            "solution": "\n".join(solution),
            "fitness": None,
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
