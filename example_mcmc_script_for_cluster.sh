#!/bin/bash

cd $SLURM_SUBMIT_DIR

## Change to your own directory
source /users/sjoudaki/anaconda3/etc/profile.d/conda.sh

conda activate likelihood

mpirun -n 8 python runmcmc_spectro.py
