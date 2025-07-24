Installation
============

Downloading CLOE
----------------

The CLOE repository can be cloned through the following command:

.. code-block:: bash

	git clone https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation.git


Installing CLOE
---------------

We provide below different options for installation, with conda, pip, or docker. 
In the troubleshooting section you can find a list of suggestions to solve issues encountered so far, which may help you with the installation.

**Installation with a Conda environment (recommended)**

To build the package in a dedicated `Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ with development tools, execute the following commands:

.. code-block:: bash

	conda env create -f environment.yml
	conda activate cloe
	pip install .

In order to use CLOE with CosmoSIS, one subsequently needs to add the line

.. code-block:: python

    cloe_parameters = "cloe_parameters"

in ``cosmosis/datablock/cosmosis_py/section_names.py`` where CosmoSIS is installed.
No additional steps are required to use CLOE with Cobaya.

**Stand-alone build**

Alternatively, the CLOE package can be built by cloning this repository using git and installing the versions of the dependencies listed in `environment.yml <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/environment.yml>`_ 
using your favourite tool for installing python packages. Then, install CLOE by running

.. code-block:: bash
	
	python setup.py install


**Docker image**

The CLOE Docker image comes with CLOE and all dependencies pre-installed. In order to use this image you will need to have Docker installed on the relevant machine.

First, log into the Euclid GitLab container registry,

.. code-block:: bash

	docker login gitlab.euclid-sgs.uk:4567

Next, pull the latest CLOE Docker image:

.. code-block:: bash

	docker pull gitlab.euclid-sgs.uk:4567/pf-ist-likelihood/likelihood-implementation/cloe

and tag an alias called cloe to avoid writing the full image name for every command.

.. code-block:: bash

	docker tag gitlab.euclid-sgs.uk:4567/pf-ist-likelihood/likelihood-implementation/cloe cloe

No further installation or set up is required. 

An interactive CLOE Docker container can now be launched as follows.

.. code-block:: bash

	docker run -it --rm cloe

Inside the container, you will need to activate the cloe environment.

.. code-block:: bash

	conda activate cloe

All the CLOE package materials can be found in /home. 

CLOE can moreover be run in a non-interactive (i.e. detached) container as follows:

.. code-block:: bash

	docker run --rm cloe bash -cl "<COMMAND>"

where <COMMAND> is the command line you wish to run, e.g. the ``run_cloe.py`` script.

.. code-block:: bash

	docker run --rm cloe bash -cl "python run_cloe.py configs/config_profiling_evaluate_likelihood.yaml"

It is also possible to launch a Jupyter Notebook using a CLOE Docker container as the backend. To do so, run the following:

.. code-block:: bash

	docker run -p 8888:8888 --rm cloe bash -cl "notebook"

Troubleshooting
---------------
        
- Users of Mac computers with Apple chips often encouter issues when trying to install CLASS (``classy``) using conda/pip.

We first note that ``classy`` is not a requirement for CLOE to work. As such, including it in the CLOE conda environment 
is optional and if you are happy using CAMB as a Boltzmann solver, you can simply skip the installation of ``classy``.  
If you need to install it, but unable to do so via conda/pip, you can download CLASS and edit the Makefile with one of the two following options:

1. deactivate OpenMP or
2. change the C compiler to gcc.
        
Then, compiling with ``make`` (or ``make -j``, check the CLASS documentation) will also generate the python wrapper ``classy``.

- In the event that pip install does not work, either try ``conda install`` or try the following:

.. code-block:: bash

        git clone https://github.com/astropy/extension-helpers.git
        cd extension-helpers
        pip install .

- In the event of problems installing openmpi4, try installing (loading, depending on cluster) the following packages in this order:

.. code-block:: bash
        
        module load gnu9
        module load anaconda
        module load openmpi4/4.1.1
