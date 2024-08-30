#!/bin/bash

## Example MCMC script to be run on cluster. This particular script
## is executed via a command in the terminal of the following form
## sbatch example_mcmc_script_for_cluster.sh
## We note that these instructions can differ from one cluster to another.

#SBATCH -A user_name
#SBATCH -p name_of_cluster
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

echo $SLURM_JOB_ID > output_file.txt

omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads

## Here 8 refers to the number of chains
mpirun --map-by node --bind-to none -np 8 ./run_cloe.py configs/config_default.yaml >> output_file.txt
