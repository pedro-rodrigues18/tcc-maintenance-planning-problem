#!/bin/bash

# Get the parameters
instance=$4
pop_size=$5
crossover_rate=$6
mutation_rate=$7
mutation_factor=$8
rho=$9

# Path
PYTHON="/home/pedro/Repositories/tcc-maintenance-planning-problem/.venv/bin/python"
SCRIPT="/home/pedro/Repositories/tcc-maintenance-planning-problem/src/main.py"

# Execute the algorithm
$PYTHON $SCRIPT $instance $pop_size $crossover_rate $mutation_rate $mutation_factor $rho
