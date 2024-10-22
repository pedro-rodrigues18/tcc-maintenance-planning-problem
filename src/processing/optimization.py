from preprocessing.model.problem import Problem


class Optimization:
    """
    Class responsible for optimizing the maintenance scheduling problem using the
    Differential Evolution algorithm.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem

    def _build_objective_function(self, start_times):
        """
        Objective function that combines the average cost and the expected excess.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            objective_value (float): Objective function value
        """
        if not self._constraints_satisfied(start_times):
            return float("inf")

        T = self.problem.time_horizon.time_steps
        alpha = self.problem.alpha

        mean_risk = 0.0
        expected_excess = 0.0

        for t in range(T):
            risks_at_t = []

            for i, intervention in enumerate(self.problem.interventions):
                start_time = start_times[i]

                if start_time <= t < start_time + intervention.delta[start_time - 1]:
                    for risk in intervention.risk:
                        if risk.time_step == t + 1:
                            risks_at_t.extend(risk.scenarios)

            if risks_at_t:
                mean_risk_t = sum(risks_at_t) / len(risks_at_t)
                mean_risk += mean_risk_t

                quantile_tau = self._calculate_quantile(
                    risks_at_t, self.problem.quantile
                )

                excess_tau = max(0, quantile_tau - mean_risk_t)
                expected_excess += excess_tau

        mean_risk /= T
        expected_excess /= T

        objective_value = alpha * mean_risk + (1 - alpha) * expected_excess

        return objective_value

    def _calculate_quantile(self, data, tau):
        """
        Calculate the tau quantile of a list of data.

        Args:
            data (list of float): List of values.
            tau (float): The quantile value (ex: 0.9 for the 90th percentile).

        Returns:
            quantile_value (float): Value of the tau quantile.
        """
        sorted_data = sorted(data)
        k = int(tau * len(sorted_data)) - 1
        return sorted_data[k]

    def _intervention_constraint(self, start_times) -> bool:
        """
        Check if all interventions can be scheduled without exceeding the time horizon and respecting tmax.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if all interventions respect the time constraints, False otherwise.
        """
        for i, intervention in enumerate(self.problem.interventions):
            start_time = start_times[i]

            if start_time < 0 or start_time > intervention.tmax:
                return False

            if (
                start_time + intervention.delta[start_time - 1]
                > self.problem.time_horizon.time_steps
            ):
                return False

        return True

    def _resources_constraint(self, start_times) -> bool:
        """
        Check if the resource usage by all interventions respects the limits for each period.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if the resources are within the limits, False otherwise.
        """
        eps = 1e-6

        for t in range(self.problem.time_horizon.time_steps):
            for resource in self.problem.resources:
                total_resource_usage = 0

                for i, intervention in enumerate(self.problem.interventions):
                    start_time = start_times[i]
                    if (
                        start_time
                        <= t
                        < start_time + intervention.delta[start_time - 1]
                    ):
                        if t in intervention.resource_workload[resource.name]:
                            total_resource_usage += intervention.resource_workload[
                                resource.name
                            ][t]

                # Verifica se o uso de recursos está dentro dos limites de mínimo e máximo
                if (
                    total_resource_usage < resource.min[t] - eps
                    or total_resource_usage > resource.max[t] + eps
                ):
                    return False

        return True

    def _exclusion_constraint(self, start_times) -> bool:
        """
        Check if mutually exclusive interventions are not occurring at the same time.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if the exclusions are respected, False otherwise.
        """
        for exclusion in self.problem.exclusions:
            i1, i2, season = (
                exclusion.interventions[0],
                exclusion.interventions[1],
                exclusion.season,
            )

            start1 = start_times[i1]
            end1 = start1 + self.problem.interventions[i1].delta[start1 - 1] - 1

            start2 = start_times[i2]
            end2 = start2 + self.problem.interventions[i2].delta[start2 - 1] - 1

            if start1 <= end2 and start2 <= end1:
                t_start = max(start1, start2)
                t_end = min(end1, end2)
                for t in range(t_start, t_end + 1):
                    if self.problem.which_season(t) == season:
                        return False

        return True

    def _constraints_satisfied(self, start_times) -> bool:
        return (
            self._intervention_constraint(start_times)
            and self._resources_constraint(start_times)
            and self._exclusion_constraint(start_times)
        )
