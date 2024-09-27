#!/bin/bash

## Example MCMC script to be run on cluster. 
## This particular script is meant to be executed on a cluster 
## that uses the SLURM job scheduler, typically via a terminal 
## command of the following form:
## sbatch example_mcmc_script_for_cluster.sh
## We note that these instructions can differ from one cluster to another.

#SBATCH -A user_name
#SBATCH -p name_of_cluster
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=job_name
#SBATCH --output=path_to_output_file.txt
#SBATCH --error=path_to_error_file.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## Here 8 refers to the number of chains
mpirun --map-by node --bind-to none -np 8 ./run_cloe.py configs/config_default.yaml
