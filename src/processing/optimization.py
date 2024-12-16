import math
import numpy as np
from preprocessing.model.problem import Problem


class Optimization:
    """
    Class responsible for building the objective function and checking the constraints.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def _build_objective_function(self, start_times, penalty=0) -> float:
        """
        Objective function that combines the average cost and the expected excess.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            objective_value (float): Objective function value
        """
        _, penalty = self._constraints_satisfied(start_times)

        T = self.problem.time_horizon.time_steps
        quantile = self.problem.quantile
        alpha = self.problem.alpha
        mean_risk = 0.0
        expected_excess = 0.0

        for t in range(1, T + 1):
            risk_t = 0.0
            risk_by_scenario = []

            for i, intervention in enumerate(self.problem.interventions):
                start_time = start_times[i]
                if start_time <= t < start_time + intervention.delta[start_time - 1]:
                    for s in range(self.problem.scenarios[t - 1]):
                        try:
                            risk_value = intervention.risk[str(t)][str(start_time)][s]
                        except KeyError:
                            risk_value = 0.0

                        risk_t += risk_value
                        # Acumula o risco para cada cenário no tempo t
                        if len(risk_by_scenario) <= s:
                            risk_by_scenario.append(risk_value)
                        else:
                            risk_by_scenario[s] += risk_value

            # Calcular média de risco para o tempo t
            risk_t /= max(1, self.problem.scenarios[t - 1])  # Evitar divisão por zero
            mean_risk += risk_t

            # Ordena para calcular o excesso usando o quantil
            risk_by_scenario_sorted = sorted(risk_by_scenario)
            quantile_index = int(math.ceil(quantile * len(risk_by_scenario_sorted))) - 1
            excess_t = 0.0
            if risk_by_scenario_sorted:
                excess_t = max(0.0, risk_by_scenario_sorted[quantile_index] - risk_t)

            expected_excess += excess_t

        mean_risk /= T
        expected_excess /= T
        objective = (alpha * mean_risk) + ((1.0 - alpha) * expected_excess)

        return objective + penalty, mean_risk, expected_excess

    def _intervention_constraint(self, start_times) -> tuple[bool, float]:
        """
        Check if all interventions can be scheduled without exceeding the time horizon and respecting tmax.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if all interventions respect the time constraints, False otherwise.
        """
        penalty = 0
        for i, intervention in enumerate(self.problem.interventions):
            start_time = start_times[i]

            if start_time < 0 or start_time > intervention.tmax:
                penalty += start_time - intervention.tmax
                return True, penalty

            if (
                start_time + intervention.delta[start_time - 1] - 1
                > self.problem.time_horizon.time_steps
            ):
                penalty = (
                    start_time + intervention.delta[start_time - 1] - 1
                ) - self.problem.time_horizon.time_steps
                return True, penalty

        return False, penalty

    def _resources_constraint(self, start_times) -> tuple[bool, float]:
        """
        Check if the resource usage by all interventions respects the limits for each period.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if the resources are within the limits, False otherwise.
        """
        eps = 1e-6
        penalty = 0
        for t in range(1, self.problem.time_horizon.time_steps + 1):
            for resource in self.problem.resources:
                total_resource_usage = 0
                for i, intervention in enumerate(self.problem.interventions):
                    start_time = start_times[i]
                    if (
                        start_time
                        <= t
                        <= start_time + intervention.delta[start_time - 1] - 1
                    ):
                        try:
                            if intervention.resource_workload[resource.name][str(t)]:
                                total_resource_usage += intervention.resource_workload[
                                    resource.name
                                ][str(t)][str(start_time)]

                        except KeyError:
                            pass

                # Verifica se o uso de recursos está dentro dos limites de mínimo e máximo
                if total_resource_usage < resource.min[t - 1] - eps:
                    penalty += resource.min[t - 1] - total_resource_usage

                elif total_resource_usage > resource.max[t - 1] + eps:
                    penalty += total_resource_usage - resource.max[t - 1]

        if penalty > 0:
            return True, penalty

        return False, penalty

    def _exclusion_constraint(self, start_times) -> tuple[bool, float]:
        """
        Check if mutually exclusive interventions are not occurring at the same time.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if the exclusions are respected, False otherwise.
        """
        penalty = 0
        for exclusion in self.problem.exclusions:

            i1, i2, season = (
                exclusion.interventions[0],
                exclusion.interventions[1],
                exclusion.season,
            )

            for i, intervention in enumerate(self.problem.interventions):
                # indice 0 representa intervenção 1
                if intervention.name == i1:
                    i1 = i
                if intervention.name == i2:
                    i2 = i

            start1 = start_times[i1]
            end1 = start1 + self.problem.interventions[i1].delta[start1 - 1] - 1

            start2 = start_times[i2]
            end2 = start2 + self.problem.interventions[i2].delta[start2 - 1] - 1

            t_start = max(start1, start2)
            t_end = min(end1, end2)

            if t_start <= t_end:
                for t in range(t_start, t_end + 1):
                    if t in season.duration:
                        penalty += 1

        return penalty > 0, penalty

    def _constraints_satisfied(self, start_times) -> bool:

        penalty = 0.0

        intervention_constraint = self._intervention_constraint(start_times)
        resources_constraint = self._resources_constraint(start_times)
        exclusion_constraint = self._exclusion_constraint(start_times)

        if intervention_constraint[0]:
            penalty += intervention_constraint[1] * 1e6
        if resources_constraint[0]:
            penalty += resources_constraint[1] * 1e6
        if exclusion_constraint[0]:
            penalty += exclusion_constraint[1] * 1e6

        return (
            not (
                intervention_constraint[0]
                or resources_constraint[0]
                or exclusion_constraint[0]
            ),
            penalty,
        )
