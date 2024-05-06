from preprocessing.model.problem import Problem


class OptimizationStep:
    def __init__(self, problem: Problem):
        self.problem = problem

    def __call__(self):
        raise NotImplementedError
