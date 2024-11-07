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
    file_name = "E_01"
    input_path = os.path.join(current_dir, f"../input/{file_name}.json")
    problem_loader = InputProblemLoader(input_path)

    problem = problem_loader()

    # ------------- Make the Optimization Here ----------------

    print("\nOptimizing the problem...")

    optimization_step = OptimizationStep(problem)

    optimization_info = optimization_step()

    print("\nOptimization completed.")
    print("\nBest solution:")
    print(optimization_info["solution"])
    print("\nObjective value:")
    print(optimization_info["fitness"])
    print("\n")

    print("Saving the solution to the output file...")

    with open(f"output/{file_name}.txt", "w") as f:
        f.write(optimization_info["solution"])

    print("Done!")


if __name__ == "__main__":
    main()
