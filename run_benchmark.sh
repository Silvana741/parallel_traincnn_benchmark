#!/bin/bash

# Script to run MPI program with different number of processes

PYTHON_SCRIPT="train_benchmark.py"

LOG_FILE="benchmark_results_2.txt"

# Clear the log file if it exists
> "$LOG_FILE"

# Number of processes to test
PROCESS_COUNTS=(10 12 16)

# Loop through each process count
for PROCS in "${PROCESS_COUNTS[@]}"
do
    echo "Running with $PROCS processes..."
    
    # Execute the MPI program
    mpiexec --host 127.0.0.1 -n $PROCS python $PYTHON_SCRIPT >> "$LOG_FILE" 2>&1

    # Add a separator in the log file for clarity
    echo "--------------------------------------------" >> "$LOG_FILE"
done

echo "Benchmark completed. Results saved in $LOG_FILE."
