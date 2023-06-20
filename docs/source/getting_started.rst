Getting started
==================

Dependencies
--------------

You can find them listed `here <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/environment.yml>`_.

Installation with a Conda environment
-----------------------------------------------

To build the package in a dedicated `Conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ with development tools, execute the following commands:

.. code-block:: bash

	conda env create -f environment.yml
	conda activate cloe
	python setup.py install

Stand-alone build
-------------------------

The CLOE package can be obtained by cloning this repository using git and installing the versions of the dependencies listed in `environment.yml <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/environment.yml>`_ 
using your favourite tool for installing python packages. Then, install CLOE by running:

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

**Using CLOE with external datasets such as Planck**

The CLOE conda environment does not include the Planck likelihood (``clik``) and the user will therefore need to separately install it to consider a combined analysis of Euclid and Planck.

Once the Planck likelihood is installed, the following lines can be included in the ``likelihood`` field of ``config_default.yaml`` (in the same way as Euclid):

.. code-block:: python

	likelihood:
  		planck_2018_lowl.TT_clik:
    		clik_file: /your/path/to/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik
  		planck_2018_lowl.EE_clik:
    		clik_file: /your/path/to/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik
  		planck_2018_highl_plik.TTTEEE:
    		clik_file: /your/path/to/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik

If the Planck likelihood is installed via ``cobaya-install`` (which is a feature of Cobaya and not CosmoSIS), then the ``clik_file`` lines above need to be removed.

Instead of directly editing ``config_default.py``, it is also possible to add the corresponding call in the ``likelihood`` block within the ``info`` dictionary of the example run scripts provided in the ``mcmc_scripts`` directory of CLOE. 
These example scripts accomplish exactly the same commands as the ``run_cloe.py`` instructions. The IST:L team has constructed the scripts from an internal notebook that is based on the contents of ``config_default.yaml``.
	
Is there a demonstration of how to use CLOE?
---------------------------------------------

Yes! Learn how to use CLOE with our demo. You can launch it with Jupyter Notebook: 


.. code-block:: bash
	
	jupyter-notebook notebooks/DEMO.ipyng


More information about the structure of the repository can be found in the `README <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/README.md>`_. 