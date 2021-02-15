## Run the script to evaluate the Euclid likelihood
$ python -m cProfile -o profiling_script.pstats profiling_script.py

## Given the command line argument of 1, only a subset of the 
## lensing and clustering observables are computed and output instead.
$ python -m cProfile -o profiling_script.pstats profiling_script.py 1
## If the argument above is 0 instead, the full likelihood is evaluated as
## in the scenario where no command line argument is provided by the user.
 
## Use snakeviz to visualize the profiling output as a local HTML site
$ snakeviz profiling_script.pstats
