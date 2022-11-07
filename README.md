[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# CLOE: Cosmology Likelihood for Observables in Euclid

This repository contains the theoretical computation of Euclid observables as well as the computation of the likelihood given some fiducial data. The likelihood is designed to work as an external likelihood for the Bayesian Analysis Code `Cobaya`.
The package Cosmology Likelihood for Observables in `Euclid` (CLOE) is developed by the Inter Science Working Group Taskforce for Likelihood development (IST:L) to compute the `Euclid`  likelihood. 

Check [documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/index.html)

## What's Cobaya?

`Cobaya` (code for Bayesian analysis, and Spanish for Guinea Pig) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers.

Check [documentation](https://cobaya.readthedocs.io/en/latest/index.html)


###  Installation with a conda environment

To build the package in a dedicated conda environment with development tools run:

```bash
conda env create -f environment.yml
conda activate cloe
python setup.py install
```

### Unit tests

To run the unit tests locally:

```bash
python setup.py test
```

Note that this requires the development tools.

### Running CLOE

To run CLOE, execute:

```bash
python run_cloe.py configs/config_default.yaml
```

### How do I import CLOE as an external likelihood for `Cobaya`?
Open and play with ```DEMO.ipynb```. You can find it and launch it with jupyter with

```
cd likelihood-implementation/notebooks/
jupyter notebook
```

This DEMO allows to compute the Galaxy Clustering and Weak Lensing observational probes as defined in the current recipe and computes the likelihood value given some benchmark data. It uses Cobaya as the main Bayesian Analysis tool.

## Structure of the repository
*  **configs**: folder containing configurations files to specify the cosmological and nuisance parameters with user specifications for scales and redshifts
*  **data**:  folder containing at the moment, the fiducial data labeled as `ExternalBenchmark` for photometric and spectrocopic probes
*  **docs**:  folder containing automatically generated documentation
*  **example**: folder containing  example yaml configuration files for the user
*  **mcmc scripts**: folder containing example python scripts to run mcmc chains for different combinations of probes with free or fixed nuisance parameters
*  **notebooks**: folder containing example jupyter notebooks
*  **scripts**: folder containing  example python scripts to simulate data
*  ```run_cloe.py```: top level script for running the CLOE user interface
*  ```setup.py```: top level script for installing or testing the CLOE package
*  ```LICENCE.txt```: file containing the MIT license CLOE is under
*  **cloe**: folder containing the CLOE python package
     *  ```cobaya_interface.py```: interface with Cobaya, passes theory needed to other classes, returns loglike
                ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.cobaya_interface.html))
     
     * cosmo: folder containing cosmology module which takes cosmological parameters and theory requirements from Cobaya
        *   ```cosmology.py```: file containing cosmology class
                    ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.cosmo.cosmology.html))
        
    * data_reader: folder containing routines to read level 3 information and data files
        *  ```reader.py```: class with functions to read fiducial data 
                    ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.data_reader.reader.html))
        
    * fftlog: folder containing routines to perform integrals and apply transforms 
        *  ```fftlog.py```: class to perform integrals with the FFTLog algorithm
                    ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.fftlog.fftlog.html))    
        *  ```hankel.py```: class to perform hankel transforms with the FFTLog algorithm
                   ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.fftlog.hankel.html))  
        *  ```utils.py```: utility functions for the FFTLog module
                   ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.fftlog.utils.html))  
                   
    * like_calc: folder containing code that calculates the likelihood given the data and the theory prediction
        * ```euclike.py```: class with the calculation of the likelihood and construction of the covariance matrices in the way it is needed by the calculation ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.like_calc.euclike.html))
                      
   * masking: folder containing code to implement the masking of the data vector and of the covariance matrix
        *   ```data_handler.py```: class to rearrange data vectors, covariance matrices and to obtain the masking vector
                   ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.masking.data_handler.html))    
        *    ```masking.py```: class to mask the data vector and the covariance matrix
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.masking.masking.html))  
   
   * non_linear: folder containing the different non-linear recipes from the Insterscience Taskforce Non-Linear
        *   ```miscellanous.py```: class with functions from `cosmology.py` that are required in the nonlinear module
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.miscellanous.html))    
        *   ```nonlinear.py```: class to compute non-linear recipes
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.nonlinear.html))
        *   ```pLL_phot.py```: class containing the recipes for the lensing x lensing power spectrum
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.pgg_phot.html))
        *   ```pgL_phot.py```: class containing the recipes for the photometric galaxy x lensing power spectrum
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.pgL_phot.html))
        *   ```pgg_phot.py```: class containing the recipes for the photometric galaxy x galaxy power spectrum
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.pgg_phot.html)) 
        *   ```pgg_spectro.py```: class containing the recipes for the spectroscopic galaxy x galaxy power spectrum
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.pgg_spectro.html))
        *    ```power_spectrum.py```: class containing the recipes for a generic power spectrum
                  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.non_linear.power_spectrum.html) 
   
    * photometric_survey: folder containing code for photometric observables 
        *  ```photo.py```: class with the theoretical prediction of the photometric observables ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.photometric_survey.photo.html))
        *  ```redshift_distribution.py```: class to construct the photometric redshift distributions ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.photometric_survey.redshift_distribution.html))
        
    * spectroscopic_survey:   folder containing code for the spectroscopy galaxy clustering observable
        *   ```spectro.py```: class with the theoretical prediction of the spectroscopy galaxy clustering observables  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.spectroscopic_survey.html))
        
    * auxiliary:  folder containing code for auxiliary functions
        *   ```getdist_routines.py```: function to produce a triangle plot for the specified chain  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.getdist_routines.html))
        *   ```likelihood_yaml_handler.py```:  module to handle the yaml files and related dictionaries ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.likelihood_yaml_handler.html))
        *   ```logger.py```:  module to handle the production of log files 
                        ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.logger.html))
        *   ```matrix_manipulator.py```:  function that merges two matrices
                       ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.matrix_manipulator.html)
        *   ```observables_dealer.py```:  module to read the observable dictionary and visualise the matrix that selects observables
                       ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.observables_dealer.html))
        *   ```plotter.py```: class with plotting functions  
                       ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.plotter.html))
        *   ```plotter_default.py```:  dictionary with default settings for plotting routines
                       ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.plotter_default.html))
        *   ```redshift_bins.py```:  function to operate on redshift bin edges
                       ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.run_method.html)) 
        *   ```run_method.py```:  function to check if the code runs with an interactive interface or not
                       ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.run_method.html))
        *   ```yaml_handler.py```:  function to read and write yaml files
                        ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.auxiliary.run_method.html))     
    
    * user_interface:  folder containing code that implements the top level user interface of CLOE
        *   ```likelihood_ui.py```: class with top level user interface for running CLOE
                      ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/cloe.user_interface.likelihood_ui.html))


    * tests: folder containing unit tests to check CLOE code
        *   ```test_DEMO.py```: class with unit tests for the DEMO jupyter notebook
        *   ```test_aux.py```: class with unit tests for the auxiliary functions module
        *   ```test_cosmo.py```: class with unit tests for cosmology class
        *   ```test_data_handler.py```: class with unit tests for data handler class
        *   ```test_data_reader.py```: class with unit tests for data reader class
        *   ```test_fftlog.py```: class with unit tests for fftlog class
        *   ```test_like_calc.py```: class with unit tests for likelihood  class
        *   ```test_likelihood_ui.py```: class with unit tests for user interface class
        *   ```test_likelihood_yaml_handler.py```: class with unit tests for likelihood yaml handler class
        *   ```test_masking.py```: class with unit tests for masking class
        *   ```test_matrix_manipulator.py```: class with unit tests for matrix manipulator class
        *   ```test_nonlinear.py```: class with unit tests for non linear class
        *   ```test_phot.py```: class with unit tests for photometric class
        *   ```test_redshift_bins.py```: class with unit tests for redshift bin function 
        *   ```test_redshift_distribution.py```: class with unit tests for photometric redshift distribution class
        *   ```test_spectro.py```: class with unit tests for spectroscopy class
        *   ```test_wrapper.py```: class with unit tests for Cobaya interface class
        *   ```test_yaml_handler.py```: class with unit tests for auxiliary yaml handler class
