About this code
==================

**Cosmology Likelihood for Observables in Euclid (CLOE)** is a software package that allows the user to obtain model predictions and cosmological parameter constraints for synthetic and real Euclid data. 
The software package has been developed by the Euclid Consortium. This includes the `core team <https://gitlab.euclid-sgs.uk/pf-ist-likelihood/likelihood-implementation/-/wikis/ISTL-core-members>`_ of the Inter-Science Taskforce for Likelihood (IST:L) within Euclid, 
in close collaboration with all of the Euclid Science Working Groups, Organisational Units, and the Inter-Science Taskforce for Nonlinear effects (IST:NL).

In the latest version of CLOE, the Euclid observables are defined by the following set:

- Weak Gravitational Lensing
- Photometric Galaxy Clustering
- Photometric Galaxy-Galaxy Lensing
- Spectroscopic Galaxy Clustering

CLOE allows the user to consider these probes either separately or in a self-consistent combined analysis. 
Additional Euclid observables in CLOE include galaxy cluster probes and cross-correlations with the cosmic microwave background.
It is also possible to analyze the Euclid data alongside other external datasets. 

Integration of CLOE with other codes
-------------------------------------

CLOE allows the user to obtain the linear matter power spectrum from either of the `CAMB <https://camb.readthedocs.io/en/latest/>`_ and `CLASS <https://lesgourg.github.io/class_public/class.html>`_ Boltzmann codes.

In order to obtain cosmological parameter constraints, CLOE reads in the redshift distributions and computes the theoretical predictions of the Euclid observables, which are used together with the data and covariance 
to obtain the likelihood. The likelihood is then evaluated across the parameter space using one of the samplers of `Cobaya <https://cobaya.readthedocs.io/en/latest/>`_ or `CosmoSIS <https://cosmosis.readthedocs.io/en/latest/>`_ 
to obtain the posterior probability.

Lensing effects beyond LCDM models
------------------------------------

CLOE now enables the computation of the conversion factor between Weyl and matter power spectra,
specifically for analyzing the lensing effects within modified gravity theories.
