#!/bin/bash

## Example MCMC script to be run on cluster. This particular script
## is executed via a command in the terminal of the following form
## addqueue -q "NAME_OF_CLUSTER" -n 1x32 -s -m 2.5 example_mcmc_script_for_cluster.sh
## where 1x32 implies a single node with 32 cores and -m refers to the memory in GB.

cd $SLURM_SUBMIT_DIR

## Change to your own directory
source /users/sjoudaki/anaconda3/etc/profile.d/conda.sh

conda activate likelihood

## Here 8 refers to the number of chains
mpirun -n 8 python runmcmc_spectro.py
