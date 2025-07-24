Quick Guide
===========

Running CLOE
------------

**With YAML File**

CLOE can be run through the following command that uses the configuration YAML file:

.. code-block:: bash

	python run_cloe.py configs/config_default.yaml

A shorthand version of this is ``./run_cloe.py``. 

**With Python Script**

CLOE can alternatively be run through a Python script, examples of which are provided in the ``mcmc_scripts`` directory.
In a combined analysis of the 3×2pt observables and redshift-space galaxy clustering in ΛCDM, the user would thereby execute:

.. code-block:: bash

	python mcmc_scripts/runmcmc_3x2pt+GCsp_LCDM.py

In either of these cases, the user can perform the run on a computing cluster through the example script provided in ``example_mcmc_script_for_cluster.sh``. 
In both cases, the output is written to a folder called ``chains`` that is generated at runtime.
These chains can then be used to obtain the marginalized parameter constraints and contours using public tools such as `GetDist <https://getdist.readthedocs.io/en/latest/>`_.

User Choices
------------

The user can choose the configurations for the Monte Carlo run through ``config_default.yaml``.

This includes whether the Monte Carlo run is performed with ``Cobaya`` or ``CosmoSIS`` by modifying the ``backend`` key
and whether ``CAMB`` or ``CLASS`` is used to compute the linear matter power spectrum and background expansion by modifying the ``solver`` key.
The user will also be able to choose a variety of other settings in this configuration file, such as the cosmological probes and datasets to consider in the analysis, 
the cosmological model, the treatment of systematic uncertainties, nonlinear modeling, scale cuts, and parameter priors.

The user is able to perform a single likelihood evaluation by, for instance, changing the ``solver`` key to ``evaluate`` in the case of Cobaya.
The user can also write the theory predictions to file at this fixed point in parameter space by setting ``print_theory`` to ``True``.


Graphical User Interface
------------------------

We have created a Graphical User Interface (GUI) to assist users
in running CLOE. This GUI targets beginner users in particular, 
by allowing them to rapidly produce CLOE-compatible configurations. 
To this end, the GUI is not designed to perform a Monte Carlo run, 
but rather allows the user to generate the configuration files needed for such a run. 
The GUI is also not meant to replace the configuration files in the ``configs`` directory, 
but rather provides an interactive way to guide users in the creation of these files.

The GUI is activated by executing ``gui/script_gui.py``.

The primary limitation of the GUI is that it provides less fine-grained 
control over CLOE options compared to the direct usage of the configuration files. 
As a result, more advanced users will likely prefer to either directly modify the example 
configuration files that we have provided or create entirely new files
themselves. Users also have the possibility to modify the GUI-generated configuration files at the command line.


Using CLOE with External Datasets
---------------------------------

CLOE can be used together with external datasets as part of the Cobaya and CosmoSIS platforms. Note however that some of these external datasets require a separate installation.
In particular, the CLOE conda environment does not include the Planck likelihood and the user will therefore need to separately install it to consider a combined analysis of Euclid and Planck.

Once the Planck likelihood is installed, the ``likelihood`` block for CLOE with Cobaya and the ``pipeline`` block for CLOE with
CosmoSIS will need to be updated in the standard way for the use of this dataset on the Cobaya/CosmoSIS platforms. 

As a concrete example, in the case of Cobaya, the following lines can be included in ``config_default.yaml``:

.. code-block:: python

	likelihood:
  		planck_2018_lowl.TT_clik:
    		clik_file: /your/path/to/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik
  		planck_2018_lowl.EE_clik:
    		clik_file: /your/path/to/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik
  		planck_2018_highl_plik.TTTEEE:
    		clik_file: /your/path/to/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik

If the Planck likelihood is installed via ``cobaya-install`` (which as the name suggests is solely a feature of Cobaya), then the ``clik_file`` lines above need to be removed.

Instead of directly editing ``config_default.py``, it is also possible to add the corresponding call in the ``likelihood`` block within the ``info`` dictionary of the example run scripts provided in the ``mcmc_scripts`` directory of CLOE. 
	

Unit and Verification Tests
---------------------------

To run the unit tests locally:

.. code-block:: bash

	python -m pytest

To run the verification tests locally:

.. code-block:: bash

	python -m pytest cloe/tests/verification

Note that these tests require the development tools.


Dependencies
------------

You can find the code dependencies listed `here <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/environment.yml>`_.


Is There a Demonstration of How to Use CLOE?
---------------------------------------------

Yes! Learn how to use CLOE with our Jupyter notebooks.
These can be launched in the following way, here in the case
of the demo notebook:

.. code-block:: bash
	
	jupyter-notebook notebooks/DEMO.ipyng

This notebook illustrates how to compute the theory predictions and likelihood for
the primary probes given synthetic Euclid data. 
We have also provided an interactive demonstration in the :doc:`Cookbook section <./example>`.

More broadly, CLOE contains a variety of Jupyter Notebooks in the ``notebooks`` directory 
that demonstrate its use and perform distinct validations. Two examples constitute:

- The notebook ``DEMO.ipynb`` is executed as part of the continuous integration of CLOE, where it allows for an end-to-end verification 
  test that covers a larger scope than individual code units and helps determine if CLOE as a whole is working as expected.

- The notebook ``cosmosis_validation.ipynb`` illustrates the agreement of the log-likelihood
  and intermediate quantities between CLOE’s CosmoSIS and Cobaya backends. 

Analogous demonstration notebooks are given by

- CMB cross-correlations: ``CMBX_Probes.ipynb``
- Clusters of galaxies probes: ``DEMO_Clusters_of_Galaxies.ipynb`` 
- Spectroscopic magnification bias: ``DEMO_with_magnification.ipynb`` 
- The Weyl potential: ``DEMO_Weyl.ipynb`` 
- Photometric galaxy clustering: ``DEMO_ISTNL_photo.ipynb``, ``DEMO_ISTNL_photo_NLbias.ipynb`` 
- Spectroscopic galaxy clustering: ``DEMO_ISTNL_spectro.ipynb``, ``eft_multipoles.ipynb`` 
- NLA and TATT IA models: ``DEMO_ISTNL_photo_IA_NLA.ipynb``, ``DEMO_ISTNL_photo_IA_TATT.ipynb``. 

More information about the structure of the repository can be found in the `README <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/README.md>`_.
