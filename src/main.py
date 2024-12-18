import os
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.input_problem_loader import InputProblemLoader

from processing.optimization_step import OptimizationStep


from utils.log import log


def main() -> None:
    """
    Student: Pedro Henrique Rodrigues Pereira
    Professor: Dr. André Luiz Maravilha da Silva
    """
    # ------------- Load the Problem ----------------
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # file_name = "E_02"
    # input_path = os.path.join(current_dir, f"../input/{file_name}.json")

    file_name = sys.argv[1]  # Instance name passed as argument
    input_path = os.path.join(current_dir, f"../input/{file_name}.json")

    log(file_name, f"Starting at {current_time}")

    problem_loader = InputProblemLoader(input_path)

    problem = problem_loader()

    # ------------- Make the Optimization ----------------
    log(f"{file_name}", "Optimizing the problem...")

    pop_size = int(sys.argv[2]) if int(sys.argv[2]) % 2 == 0 else int(sys.argv[2]) + 1
    crossover_rate = float(sys.argv[3])
    mutation_rate = float(sys.argv[4])
    mutation_factor = float(sys.argv[5])
    rho = float(sys.argv[6])

    optimization_step = OptimizationStep(
        problem=problem,
        file_name=file_name,
        pop_size=pop_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_factor=mutation_factor,
        rho=rho,
    )

    optimization_info = optimization_step()

    log(f"{file_name}", "\nOptimization completed.")
    # log(f"{file_name}", "\nSolution found:")
    # log(f"{file_name}", optimization_info["solution"])
    # log(f"{file_name}", "\nObjective value:")
    # log(f"{file_name}", str(optimization_info["fitness"]))
    log(f"{file_name}", "Saving the solution to the output file...")

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_name}.txt"
    with open(output_file, "w") as f:
        f.write(f"{optimization_info['solution']}")

    log(f"{file_name}", "Done!\n")

    # Print the fitness value to use in irace
    print(optimization_info["fitness"])


if __name__ == "__main__":
    main()
