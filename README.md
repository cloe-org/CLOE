[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# Mock structure for COBAYA as external likelihood

### What's COBAYA?

Cobaya (code for bayesian analysis, and Spanish for Guinea Pig) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers (including the advanced MCMC sampler from CosmoMC, and the advanced nested sampler PolyChord).

### COBAYA's documentation

Check [documentation](https://cobaya.readthedocs.io/en/latest/index.html)


### Stand alone build

The package can be installed by simply running:

```bash
python setup.py install
```

### Conda environment

To build the package in a dedicated Conda environment with development tools run:

```bash
conda env create -f environment.yml
conda activate cosmobox
python setup.py install
```

### Unit tests

To run the unit tests locally run:

```bash
python setup.py test
```

Note that this requires the development tools.


### Calling external likelihood
Open and play with ```likelihood_test.ipynb```

## Structure of the repository

*  **data**: future data folder
*  **docs**: documentation
*  **likelihood**: likelihood code
    *  ```cobaya_interface.py```: interface with COBAYA, pass theory needs to other classes, returns loglike  
    * test: unit tests to check likelihood code
        *   ```test_estimates.py```: class with unit tests for common class estimate.py
        *   ```test_cosmo.py```: class with unit tests for cosmology class cosmology.py
        *   ```test_shear.py```: class with unit tests for shear class shear.py
        *   ```test_spec.py```: class with unit tests for spec class spec.py
        *   ```test_wrapper.py```: class with a general call to COBAYA `get_model` wrapper
     * cosmo: cosmology module which takes cosmological parameters and theory needs from Cobaya
        *   ```cosmology.py```: class with cosmological functions
                                      
    * general_specs: general module with common aux functions useful for both WL and GC
        *   ```estimates.py```
    * photometric_survey: code for photo z observable
        *  ```shear.py```
    * spectroscopic_survey: code for the spectroscopy GC observable
        * ```spec.py```
