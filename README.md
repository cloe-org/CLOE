[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# CLOE: Cosmology Likelihood for Observables in Euclid

This repository contains the theoretical computation of Euclid observables as well as the computation of the likelihood given some fiducial data. CLOE is designed to work as an external likelihood for the Bayesian Analysis Code `Cobaya`.

Check [documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/index.html)

## What's COBAYA?

`Cobaya` (code for Bayesian analysis, and Spanish for Guinea Pig) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers.

Check [documentation](https://cobaya.readthedocs.io/en/latest/index.html)


### Conda environment

To build the package in a dedicated Conda environment with development tools run:

```bash
conda env create -f environment.yml
conda activate cloe
python setup.py install
```

### Stand-alone build

The CLOE package can be obtained by cloning this repository using git and get  installed by simply running:

```bash
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
Open and play with ```DEMO.ipynb```

## Structure of the repository

*  **data**: folder that contains, at the moment, the fiducial data
*  **docs**: automatically generated documentation
*  **notebooks**: folder containing example notebooks
*  **scripts**: folder containing example scripts
*  **cloe**: CLOE code
     *  ```cobaya_interface.py```: interface with COBAYA, pass theory needed to other classes, returns loglike
     * like_calc: code that calculates the likelihood given the data and the theory prediction
        * ```euclike.py```: class with the calculation of the likelihood and construction of the covariance matrices in the way it is needed by the calculation ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/likelihood.like_calc.euclike.html))
     
     * cosmo: cosmology module which takes cosmological parameters and theory requirements from Cobaya
        *   ```cosmology.py```: class with cosmological function ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/likelihood.cosmo.cosmology.html))
        
    * data_reader: code containing routines to read OU-level3 information and data files
        *  ```reader.py```: class with functions to read fiducial data ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/likelihood.data_reader.reader.html))
    * photometric_survey: code for photo-z observables 
        *  ```photo.py```: class with the theoretical prediction of the photo-z observables ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/likelihood.photometric_survey.photo.html))
    * spectroscopic_survey: code for the spectroscopy GC observable
        * ```spec.py```: class with the theoretical prediction of the spectroscopy GC observables  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/likelihood.spectroscopic_survey.spec.html))       
    * auxiliary: code for auxiliary functions
        * ```plotter.py```: class with plotting functions  ([documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/likelihood.auxiliary.plotter.html))
    * test: unit tests to check CLOE code
        *   ```test_cosmo.py```: class with unit tests for cosmology class cosmology.py
        *   ```test_shear.py```: class with unit tests for shear class shear.py
        *   ```test_spec.py```: class with unit tests for spec class spec.py
        *   ```test_wrapper.py```: class with a general call to COBAYA `get_model` wrapper
