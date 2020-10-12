## Run the script to compute the GCph and WL C_ells
$ python -m cProfile -o profiling_script_C_ells_ph.pstats profiling_script_C_ells_ph.py
## Computes the GCph and WL C_ells for 100 values of ell, between 10 and 1000.
 
## Use snakeviz to visualize the profiling output as a local HTML site
$ snakeviz profiling_script_C_ells_ph.pstats


