## Run the script to compute the GCph and WL C_ells
$ python -m cProfile -o profiling_script_C_ells_ph.pstats profiling_script_C_ells_ph.py 100
## the command line argument "100" can be any integer, specifying the number of ells to run and compare with the benchmark files

## Use snakeviz to visualize the profiling output as a local HTML site
$ snakeviz profiling_script_C_ells_ph.pstats


