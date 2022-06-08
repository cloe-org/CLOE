#!/bin/bash

## Example MCMC script to be run on cluster. This particular script
## is executed via a command in the terminal of the following form
## addqueue -q "NAME_OF_QUEUE" -n 1x32 -s -m 2.5 example_mcmc_script_for_cluster.sh
## where 1x32 implies a single node with 32 cores and 2.5 refers to the memory in GB.
## We note that these instructions can differ from one cluster to another.

cd $SLURM_SUBMIT_DIR

## Change to your own directory
source /users/sjoudaki/anaconda3/etc/profile.d/conda.sh

conda activate cloe

## Here 8 refers to the number of chains
mpirun -n 8 python mcmc_scripts/runmcmc_spectro.py
