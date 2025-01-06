import json
import os
import sys
from pathlib import Path
import time
import concurrent.futures

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.model.problem import Problem

from preprocessing.input_problem_loader import InputProblemLoader

from processing.optimization_step import OptimizationStep

from utils.log import log


def load_problem(current_dir, instance) -> Problem:
    """
    Load the problem from the input file

    Args:
        current_dir (str): The current directory.
        instance (str): The instance name.
    Returns:
        dict: The problem object.
    """

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    input_path = os.path.join(current_dir, f"../input/{instance}.json")

    log(instance, f"Starting at {current_time}")

    problem_loader = InputProblemLoader(input_path)

    problem = problem_loader()

    return problem


def make_optimization(
    problem, instance, pop_size, crossover_rate, mutation_rate, mutation_factor, rho
) -> tuple:
    """
    Perform optimization on the given problem instance using specified parameters.

    Args:
        problem (Problem): The problem to be optimized.
        instance (str): The name of the problem instance.
        pop_size (int): The population size.
        crossover_rate (float): The crossover rate.
        mutation_rate (float): The mutation rate.
        mutation_factor (float): The mutation factor.
        rho (float): The parameter rho.
    Returns:
        A tuple containing the solution and its fitness value.
    """

    log(f"{instance}", "Optimizing the problem...")

    optimization_step = OptimizationStep(
        problem=problem,
        file_name=instance,
        pop_size=pop_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_factor=mutation_factor,
        rho=rho,
    )

    optimization_info = optimization_step()

    log(f"{instance}", "\nOptimization completed.")
    log(f"{instance}", "Saving the solution to the output file...")

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{instance}.txt"
    with open(output_file, "w") as f:
        f.write(f"{optimization_info['solution']}")

    log(f"{instance}", "Done!\n")

    return optimization_info["solution"], optimization_info["fitness"]


def run_all_instances(parameters) -> None:
    """
    Run instances in parallel with concurrent.futures

    Args:
        parameters (dict): The parameters of the instances
    Returns:
        None
    """
    instances = parameters["set"]["A"]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = {
            executor.submit(
                make_optimization,
                load_problem(os.path.dirname(os.path.abspath(__file__)), instance_name),
                instance_name,
                instance_params["pop_size"],
                instance_params["crossover_rate"],
                instance_params["mutation_rate"],
                instance_params["mutation_factor"],
                instance_params["rho"],
            ): instance_name
            for instance_name, instance_params in instances.items()
        }

        for future in concurrent.futures.as_completed(results):
            instance = results[future]
            try:
                _, fitness = future.result()
                print(f"{instance}: {fitness}")
            except Exception as e:
                print(f"{instance} generated an exception: {e}")


def main() -> None:
    """
    Student: Pedro Henrique Rodrigues Pereira
    Professor: Dr. André Luiz Maravilha da Silva
    """
    # ------------- Load the Problem ----------------

    current_dir = os.path.dirname(os.path.abspath(__file__))

    parameters_path = os.path.join(current_dir, f"../input/parameters.json")

    try:
        with open(parameters_path, "r") as file:
            parameters = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {parameters_path} not found")

    irace = parameters["irace"]  # If True, the parameters will be passed by irace
    run_all = parameters[
        "run_all"
    ]  # This parameter is used to run all instances in parallel

    if run_all:
        run_all_instances(parameters)
    else:
        if irace:
            instance = sys.argv[1]
            pop_size = int(sys.argv[2])
            crossover_rate = float(sys.argv[3])
            mutation_rate = float(sys.argv[4])
            mutation_factor = float(sys.argv[5])
            rho = float(sys.argv[6])
        else:
            dataset = "B"
            instance = "B_01"  # The default instance because it is the smallest and runs faster
            pop_size = parameters["set"][dataset][instance]["pop_size"]
            crossover_rate = parameters["set"][dataset][instance]["crossover_rate"]
            mutation_rate = parameters["set"][dataset][instance]["mutation_rate"]
            mutation_factor = parameters["set"][dataset][instance]["mutation_factor"]
            rho = parameters["set"][dataset][instance]["rho"]

        problem = load_problem(current_dir, instance)

        # ------------- Make the Optimization ----------------

        _, solution_fitness = make_optimization(
            problem,
            instance,
            pop_size,
            crossover_rate,
            mutation_rate,
            mutation_factor,
            rho,
        )

        print(f"{instance}: {solution_fitness}")


if __name__ == "__main__":
    main()
