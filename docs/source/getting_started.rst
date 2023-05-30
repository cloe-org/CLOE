Getting started
==================

Dependencies
--------------

You can find them listed `here <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/develop/environment.yml>`_.

Installation with a Conda environment
-----------------------------------------------

To build the package in a dedicated `Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ with development tools, execute the following commands:

.. code-block:: bash

	conda env create -f environment.yml
	conda activate cloe
	python setup.py install

Stand-alone build
-------------------------

The CLOE package can be obtained by cloning this repository using git and get installed by running:

.. code-block:: bash
	
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
	
Is there a demonstration of how to use CLOE?
---------------------------------------------

Yes! Learn how to use CLOE with our demo. You can launch it with Jupyter Notebook: 


.. code-block:: bash
	
	jupyter-notebook notebooks/DEMO.ipyng


More information about the structure of the repository can be found in the `README <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/README.md>`_. 