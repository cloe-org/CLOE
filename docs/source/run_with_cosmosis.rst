Running CLOE with ``CosmoSIS``
=================================

The default sampler for CLOE is ``Cobaya``; however the option of using ``CosmoSIS`` is also available should the user choose to do so. This section details the two ways of using CLOE with ``CosmoSIS``: the main difference between them being whether the user wants to call the Boltzmann solver (a) directly from ``CosmoSIS``, or (b) through ``Cobaya`` running under-the-hood. In both cases, the user does not have to interact directly with ``Cobaya``.

All the relevant scripts pertaining to running CLOE with ``CosmoSIS`` are contained within the ``cosmosis`` directory of the CLOE repository. For a detailed description of each file, please refer to the `README <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/cosmosis/README.md>`_. 

For each of the two cases, an example ``values.ini`` and ``priors.ini`` file has been provided. Additionally, for testing and verification purposes, the user can run the `verification Jupyter notebook <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/blob/master/cosmosis/cosmosis_validation.ipynb>`_.

Note: These instructions assume that the user already has a working installation of ``CosmoSIS`` and the CosmoSIS standard library. Learn more about downloading and installing `CosmoSIS <https://cosmosis.readthedocs.io/en/latest/>`_. 

``CosmoSIS`` installation
-------------------------
To install the ``CosmoSIS`` library (without external cosmological codes and likelihoods), it is sufficient to do

.. code-block: bash

    pip install cosmosis


``CosmoSIS`` independent of ``Cobaya``
---------------------------------------

Should the user choose to run the pipeline as in scenario (a), i.e., separating the Boltzmann solver and likelihood calculations, simply run (assuming MPI is not being called):

.. code-block:: bash

    cosmosis cosmosis/run_cosmosis.ini

``CosmoSIS`` with ``Cobaya``
-----------------------------

Should the user choose to run the pipeline as in scenario (b), i.e., calling ``Cobaya`` through ``CosmoSIS`` to carry out both the cosmological and likelihood calculations, run instead (assuming MPI is not being called):

.. code-block:: bash

    cosmosis cosmosis/run_cosmosis_with_cobaya.ini
