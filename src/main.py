import os

from preprocessing.input_problem_loader import InputProblemLoader
from processing.optimization_step import OptimizationStep


def main():
    """
    Student: Pedro Henrique Rodrigues Pereira
    Professor: Dr. André Luiz Maravilha da Silva
    """
    # ------------- Load the Problem Here ----------------

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "../instances/example/example1.json")
    problem_loader = InputProblemLoader(input_path)

    problem = problem_loader()

    # ------------- Make the Optimization Here ----------------

    optimization_step = OptimizationStep(problem)
    pass


if __name__ == "__main__":
    main()
