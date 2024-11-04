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
        _, penalty = self._constraints_satisfied(start_times)

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

        return objective_value + penalty

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
                # print("Start time: ", start_time)
                # print("Tmax: ", intervention.tmax)
                penalty += start_time - intervention.tmax
                return False, penalty

            if (
                start_time + intervention.delta[start_time - 1] - 1
                > self.problem.time_horizon.time_steps
            ):
                # print("Start time: ", start_time)
                # print("Delta: ", intervention.delta[i])
                # print("Time horizon: ", self.problem.time_horizon.time_steps)
                penalty = (
                    start_time + intervention.delta[start_time - 1] - 1
                ) - self.problem.time_horizon.time_steps
                return False, penalty

        return True, penalty

    def _resources_constraint(self, start_times) -> bool:
        """
        Check if the resource usage by all interventions respects the limits for each period.

        Args:
            start_times (list): List with the start times for each intervention.

        Returns:
            bool: True if the resources are within the limits, False otherwise.
        """
        eps = 1e-6
        penalty = 0
        for t in range(1, self.problem.time_horizon.time_steps):
            # print("Time step: ", t)
            # breakpoint()
            for resource in self.problem.resources:
                total_resource_usage = 0
                for i, intervention in enumerate(self.problem.interventions):
                    start_time = start_times[i]
                    # print("start_times: ", len(start_times))
                    # print("i: ", i)
                    # print("delta: ", len(intervention.delta))
                    if (
                        start_time
                        <= t
                        <= start_time + intervention.delta[start_time - 1] - 1
                    ):
                        # print("t: ", t)
                        # print("Resource workload: ", intervention.resource_workload)
                        # print("Resource name: ", resource.name)
                        try:
                            if t in intervention.resource_workload[resource.name]:
                                total_resource_usage += intervention.resource_workload[
                                    resource.name
                                ][t]
                        except KeyError:
                            pass

                # Verifica se o uso de recursos está dentro dos limites de mínimo e máximo
                if (
                    total_resource_usage < resource.min[t] - eps
                    or total_resource_usage > resource.max[t] + eps
                ):
                    # print("Resource usage: ", total_resource_usage)
                    penalty += abs(total_resource_usage - resource.max[t])
                    return False, penalty

        return True, penalty

    def _exclusion_constraint(self, start_times) -> bool:
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
            # print("i1: ", i1)
            # print("i2: ", i2)
            # print("season: ", season)
            for i, intervention in enumerate(self.problem.interventions):
                # print("i: ", i)
                # indice 0 representa intervenção 1
                if intervention.name == i1:
                    i1 = i
                if intervention.name == i2:
                    i2 = i

            # print("i1: ", i1)
            # print("i2: ", i2)
            # breakpoint()

            start1 = start_times[i1]
            end1 = start1 + self.problem.interventions[i1].delta[i1] - 1

            start2 = start_times[i2]
            end2 = start2 + self.problem.interventions[i2].delta[i2] - 1

            if start1 < end2 and start2 < end1:
                t_start = max(start1, start2)
                t_end = min(end1, end2)
                for t in range(t_start, t_end + 1):
                    if t in season.duration:
                        penalty += 1
                        return False, penalty

        return True, penalty

    def _constraints_satisfied(self, start_times) -> bool:

        penalty = 0

        intervention_constraint = self._intervention_constraint(start_times)
        resources_constraint = self._resources_constraint(start_times)
        exclusion_constraint = self._exclusion_constraint(start_times)

        if not intervention_constraint[0]:
            # print("Intervention constraint violated.")
            # print("Penalty: ", intervention_constraint[1])
            penalty += intervention_constraint[1]
        if not resources_constraint[0]:
            # print("Resources constraint violated.")
            # print("Penalty: ", resources_constraint[1])
            penalty += resources_constraint[1]
        if not exclusion_constraint[0]:
            # print("Exclusion constraint violated.")
            # print("Penalty: ", exclusion_constraint[1])
            penalty += exclusion_constraint[1]

        return (
            intervention_constraint[0]
            and resources_constraint[0]
            and exclusion_constraint[0],
            penalty,
        )
