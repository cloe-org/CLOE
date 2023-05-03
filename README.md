[![pipeline status](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/pipeline.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master) [![coverage report](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/badges/master/coverage.svg)](https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/commits/master)

# CLOE: Cosmology Likelihood for Observables in Euclid

This repository contains the theoretical computation of Euclid observables as well as the computation of the likelihood given some fiducial data. The likelihood is designed to work as an external likelihood for the Bayesian Analysis Code `Cobaya`.
The package Cosmology Likelihood for Observables in `Euclid` (CLOE) is developed by the Inter-Science Working Group Task Force for Likelihood development (IST:L) to compute the `Euclid`  likelihood.

Check [documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/index.html)

## What's Cobaya?

`Cobaya` (code for Bayesian analysis) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers.

Check [documentation](https://cobaya.readthedocs.io/en/latest/index.html)

## Installation

**git clone CLOE**

```shell
git clone https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation.git
```
We provide below different options for installation: [with conda](#conda), or [with pip](#pip), or [with docker](#docker). In the [troubleshooting](#troubles) section you can find a list of  suggestions to solve issues that we encountered so far, which may help you with the installation.

### 1. Installation with a conda environment <a name="conda"></a>

**Option a)**

Use the `environment.yml` that we provide: 

```shell
conda env create -f environment.yml
conda activate cloe
pip install .
```

**Option b)** 

If our environment does not work on your cluster (ex. it gets stuck), but you have anaconda, we recommend to create your own environment as follows: 

```shell
conda create -n cloe python=3.9 
source activate cloe
```

Then, install all the following packages, in this order either with pip or with conda (depending on what works on your cluster):

```shell
pip install astropy
pip install jupyter
pip install matplotlib
conda install mpi4py
conda install numpy
conda install scipy
conda install seaborn
conda install tensorflow
pip install fast-pt
pip install camb
pip install gsl
pip install cobaya
pip install classy
pip install baccoemu
pip install euclidemu2
pip install pytest
pip install pytest-cov
pip install pytest-pycodestyle
pip install pytest-pydocstyle
pip install sphinx
pip install sphinx-rtd-theme
pip install numpydoc
```

### 2. Installation with pip <a name="pip"></a>

If you don't have anaconda, you would have to install manually on your cluster all the packages listed in Option b) written above, with pip.

### 3. CLOE Docker image <a name="docker"></a>

The CLOE Docker image comes with CLOE and all dependencies pre-installed. In order to use this image you will need to have [Docker](https://www.docker.com/) installed on the relevant machine.

#### Pull the Docker Image

Log into the Euclid GitLab container registry,

```bash
docker login gitlab.euclid-sgs.uk:4567
```

pull the latest CLOE Docker image

```bash
docker pull gitlab.euclid-sgs.uk:4567/pf-ist-likelihood/likelihood-implementation/cloe
```

and tag an alias called `cloe` to avoid writing the full image name for every command.

```bash
git tag gitlab.euclid-sgs.uk:4567/pf-ist-likelihood/likelihood-implementation/cloe cloe
```

No further installation or set up is required.

#### Run a Docker container

##### Interactive container 

An interactive CLOE Docker container can be launched as follows.

```bash
docker run -it --rm cloe
```

Inside the container you will need to activate the `cloe` environment.

```bash
conda activate cloe
```

All the CLOE package materials can be found in `/home`. 

##### Detached container

CLOE can be run in a non-interactive (i.e. detached) container as follows:

```bash
docker run --rm cloe bash -cl "<COMMAND>"
```

where `<COMMAND>` is the command line you wish to run, e.g. to run the `run_cloe.py` script.

```bash
docker run --rm cloe bash -cl "python run_cloe.py configs/config_profiling_evaluate_likelihood.yaml"
```

##### Jupyter notebook

It is also possible to launch a Jupyter Notebook using a CLOE Docker container as the backend. To do so run the following:

```bash
docker run -p 8888:8888 --rm cloe bash -cl "notebook"
```


## Troubleshooting <a name="troubles"></a>

### Problems installing openmpi4/4.1.1

Try installing (loading, depending on cluster) the following packages in this order: 

```shell
module load gnu9
module load anaconda
module load openmpi4/4.1.1
```

### Problems with classy
Users of Mac computers with Apple chips often encouter issues when trying to install `classy` using `pip`. First of all, `classy` is not a requirement for CLOE to work, if you are happy using CAMB as a Boltzmann solver you can simply skip the installation of `classy`. If you need to install it, you can download CLASS and edit the Makefile with one of the two following options:

1. deactivate OpenMP or
2. change the C compiler to `gcc`.

Then, compiling with `make` (or `make -j`, check the CLASS documentation) will also generate the python wrapper `classy`.

### Problems with other specific packages
Users have sometimes encountered issues installing some of the packages (euclidemu2). Before debugging, try to use "conda install" instead of "pip install" (or viceversa, depending on what gives error).

### Problems with pip
If pip install doesn't work, either try 'conda install' or try the following:
```shell
git clone https://github.com/astropy/extension-helpers.git
cd extension-helpers
pip install .
```

## Running CLOE

To run CLOE, execute:

```bash
python run_cloe.py configs/config_default.yaml
```

The output is written in a folder called `chains` that is generated at runtime.

## Structure of the repository
*  **cloe**: folder containing the CLOE python package (see the [API documentation](http://pf-ist-likelihood.pages.euclid-sgs.uk/likelihood-implementation/index.html) for details)
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


## Unit and verification tests

To run the unit tests locally:

```bash
python -m pytest
```

To run the verification tests locally:

```bash
python -m pytest cloe/tests/verification
```

Note that these tests require the development tools.

## How do I import CLOE as an external likelihood for `Cobaya`?
Open and play with ```DEMO.ipynb```. You can find it and launch it with jupyter with

```
cd likelihood-implementation/notebooks/
jupyter notebook
```

This DEMO allows to compute the Galaxy Clustering and Weak Lensing observational probes as defined in the current recipe and computes the likelihood value given some benchmark data. It uses Cobaya as the main Bayesian Analysis tool.

