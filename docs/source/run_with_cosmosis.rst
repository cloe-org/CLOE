Running CLOE with ``CosmoSIS``
=================================

The default sampler for CLOE is ``Cobaya``; however the option of using ``CosmoSIS`` is also available should the user choose to do so. This section details the two ways of using CLOE with ``CosmoSIS``: the main difference between them being whether the user wants to call the Boltzmann solver (a) directly from ``CosmoSIS``, or (b) through ``Cobaya`` running under-the-hood. In both cases, the user does not have to interact directly with ``Cobaya``.

All the relevant scripts pertaining to running CLOE with ``CosmoSIS`` are contained within the ``cosmosis`` directory of the CLOE repository. For a detailed description of each file, please refer to the `README <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/cosmosis/README.md>`_. 

For each of the two cases, an example ``values.ini`` and ``priors.ini`` file has been provided. Additionally, for testing and verification purposes, the user can run the `verification Jupyter notebook <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/notebooks/cosmosis_validation.ipynb>`_.

Note: These instructions assume that the user already has a working installation of ``CosmoSIS`` and the CosmoSIS standard library. Learn more about downloading and installing `CosmoSIS <https://cosmosis.readthedocs.io/en/latest/>`_. This holds if the user has performed the conda installation for CLOE.

``CosmoSIS`` installation
-------------------------
The primary approach to install CosmoSIS is through the CLOE conda environment. As a secondary approach, it is possible to install the ``CosmoSIS`` library (without external cosmological codes and likelihoods) via

.. code-block: bash

    pip install cosmosis


``CosmoSIS`` on its own
---------------------------------------

Should the user choose to run the pipeline as in scenario (a), i.e., separating the Boltzmann solver and likelihood calculations, simply choose CosmoSIS as the backend and refer to the following ini file in configs/config_default.yaml:

.. code-block:: bash

    cosmosis/run_cosmosis.ini

``CosmoSIS`` with ``Cobaya``
-----------------------------

Should the user choose to run the pipeline as in scenario (b), i.e., calling ``Cobaya`` through ``CosmoSIS`` to carry out both the cosmological and likelihood calculations, choose the following ini file instead:

.. code-block:: bash

    cosmosis/run_cosmosis_with_cobaya.ini
