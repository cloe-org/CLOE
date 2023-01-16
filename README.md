[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# CLOE: Cosmology Likelihood for Observables in Euclid

This repository contains the theoretical computation of Euclid observables as well as the computation of the likelihood given some fiducial data. The likelihood is designed to work as an external likelihood for the Bayesian Analysis Code `Cobaya`.
The package Cosmology Likelihood for Observables in `Euclid` (CLOE) is developed by the Inter-Science Working Group Task Force for Likelihood development (IST:L) to compute the `Euclid`  likelihood.

Check [documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/index.html)

## What's Cobaya?

`Cobaya` (code for Bayesian analysis) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers.

Check [documentation](https://cobaya.readthedocs.io/en/latest/index.html)


###  Installation with a conda environment

To build the package in a dedicated conda environment with development tools run:

```bash
conda env create -f environment.yml
conda activate cloe
pip install .
```

### Unit and verification tests

To run the unit tests locally:

```bash
python -m pytest
```

To run the verification tests locally:

```bash
python -m pytest cloe/tests/verification
```

Note that these tests require the development tools.

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
*  **cloe**: folder containing the CLOE python package (see the [API documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/index.html) for details)
