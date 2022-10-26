Getting started
==================

This repository contains the theoretical computation of `Euclid` observables as well as the computation of the likelihood given some fiducial data. The likelihood is designed to work as an external likelihood for the Bayesian Analysis Code Cobaya.
The package **Cosmology Likelihood for Observables in Euclid** **(CLOE)** is developed by the Inter Science Working Group Taskforce for Likelihood development.

What's Cobaya?
-------------------------

**Cobaya** (code for Bayesian Analysis, and Spanish for Guinea Pig) is a framework for sampling and statistical modelling: it allows you to explore an arbitrary prior or posterior using a range of Monte Carlo samplers (check its `docs  <https://cobaya.readthedocs.io/en/latest/index.html>`_).


Installation with a conda environment
-----------------------------------------------

To build the package in a dedicated `conda environment  <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ with development tools run:

.. code-block:: bash

	conda env create -f environment.yml
	conda activate cloe
	python setup.py install

Unit tests
-------------

To run the unit tests locally:

.. code-block:: bash

	python setup.py test

Note that this requires the development tools.

Running CLOE
--------------------

To run CLOE, execute:

.. code-block:: bash

	python run_cloe.py configs/config_default.yaml
	
How do I import the likelihood as an external likelihood for Cobaya?
------------------------------------------------------------------------------------------

Open and play with DEMO.ipynb. You can find it and launch it with jupyter with 


.. code-block:: bash
	
	cd likelihood-implementation/notebooks/
	jupyter notebook
 
 
This DEMO allows to compute the Galaxy Clustering and Weak Lensing observational probes as defined in the current recipe and computes the likelihood value given some benchmark data. It uses Cobaya as the main Bayesian Analysis tool.

More information about the structure of the repository can be found in the `README  <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/README.md>`_


