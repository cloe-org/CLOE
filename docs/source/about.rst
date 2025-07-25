Overview
========

General Description
-------------------

**The Cosmology Likelihood for Observables in Euclid (CLOE)** is a software package that allows the user to obtain model predictions and cosmological parameter constraints for synthetic and real Euclid data. 
This software package has been developed by the Euclid Consortium. 

In the latest version of CLOE, the cosmological observables are defined by the following set:

- Cosmic shear tomography
- Photometric galaxy clustering tomography
- Photometric galaxy-galaxy lensing tomography 
- Spectroscopic / Redshift-space galaxy clustering
- Cross-correlations with the cosmic microwave background
- Galaxy cluster probes

CLOE allows the user to consider these probes either separately or in a self-consistent combined analysis. 
It is also possible to analyze the Euclid data alongside other external datasets within the
`Cobaya <https://cobaya.readthedocs.io/en/latest/>`_ and `CosmoSIS <https://cosmosis.readthedocs.io/en/latest/>`_ platforms.

Integration of CLOE with Other Codes
-------------------------------------

CLOE allows the user to obtain the linear matter power spectrum from either of the 
`CAMB <https://camb.readthedocs.io/en/latest/>`_ and `CLASS <https://lesgourg.github.io/class_public/class.html>`_ Boltzmann codes.
This matter power spectrum can be extended to nonlinear scale via one of four prescriptions:
`Halofit <https://github.com/cmbant/CAMB/blob/master/fortran/halofit.f90>`_, `HMCODE <https://github.com/alexander-mead/HMcode>`_, 
`EuclidEmulator2 <https://github.com/miknab/EuclidEmulator2>`_, and `BACCO <https://baccoemu.readthedocs.io>`_.
The impact of baryonic feedback is then captured by either  HMCODE, BACCO, or `BCemu <https://github.com/sambit-giri/BCemu>`_.

The nonlinear anisotropic galaxy power spectrum modeling through the effective field theory of large scale structure relies on 
`FAST-PT <https://github.com/JoeMcEwen/FAST-PT>`_. 

.. note::
    In order to obtain the cosmological parameter constraints, CLOE reads in the measurements, covariance, and derived data products such as the redshift distributions 
    and mixing matrices. It computes the theoretical predictions of the observables, which are used together with the measurements and covariance to obtain 
    the likelihood. The likelihood is then evaluated across the parameter space using one of the Monte Carlo samplers of Cobaya or CosmoSIS to obtain the posterior probability.


Structure of CLOE
-----------------

The CLOE repository has the following structure:

- ``cloe``: CLOE source code and unit tests in Python
- ``configs``: CLOE configuration files in YAML
- ``cosmosis``: CLOE source code in Python and configuration files in INI for its CosmoSIS interface.
- ``data``: data products (be it real or synthetic)
- ``docs``: CLOE Sphinx documentation
- ``example``: example YAML files
- ``gui``: CLOE graphical user interface
- ``mcmc scripts``: example Python scripts to run MCMC chains
- ``notebooks``: CLOE demonstration and validation Jupyter Notebooks
- ``scripts``: example Python scripts to simulate data
- ``environment.yml``: CLOE conda environment
- ``example_mcmc_script_for_cluster.sh``: example shell script to submit jobs on computing cluster
- ``run_cloe.py``: top level script for running the CLOE user interface
- ``setup.py``: top level script for installing or testing CLOE
- ``LICENCE.txt``: file containing the LGPL license of CLOE


CLOE Features, Benchmarking, and Use Illustration
-------------------------------------------------

For a description of CLOE's features, benchmarking, and illustration of its use, we refer to the :doc:`set of publications <./cite_us>` released alongside this repository.

The features can be summarized as follows:

- Cosmological probe (1): Photometric 3×2pt - angular power spectra
- Cosmological probe (2): Photometric 3×2pt - pseudo-Cℓ
- Cosmological probe (3): Photometric 3×2pt - correlation functions
- Cosmological probe (4): Spectroscopic galaxy clustering - multipole power spectra
- Cosmological probe (5): Spectroscopic galaxy clustering - window-convolved multipole power spectra
- Cosmological probe (6): Spectroscopic galaxy clustering - multipole correlation functions
- Cosmological probe (7): CMB lensing and temperature correlations (Gϕ, Lϕ, ϕϕ, GT)
- Cosmological probe (8): Clusters of galaxies
- User selection of probes, scales, redshifts: Masking vector formalism
- Einstein-Boltzmann solver (1): CAMB
- Einstein-Boltzmann solver (2): CLASS
- Nonlinear matter power spectrum (1): Halofit
- Nonlinear matter power spectrum (2): HMCODE (2016 and 2020 versions)
- Nonlinear matter power spectrum (3): BACCO
- Nonlinear matter power spectrum (4): EuclidEmu2
- Baryonic feedback (1): HMCODE (2016 and 2020 versions)
- Baryonic feedback (2): BACCO
- Baryonic feedback (3): BCemu
- Reweighting of lensing kernels: BNT transformation
- Large-angle corrections: Extended Limber and curved-sky
- Redshift space distortions (1): Photometric galaxy clustering and galaxy-galaxy lensing
- Redshift space distortions (2): Spectroscopic galaxy clustering
- Intrinsic alignments (1): Extended nonlinear linear alignment model
- Intrinsic alignments (2): Tidal alignment and tidal torquing model
- Photometric source redshift uncertainties: Shift (δzs) in mean redshift of source galaxy distribution
- Photometric lens redshift uncertainties: Shift (δzl) in mean redshift of lens galaxy distribution
- Spectroscopic redshift uncertainties (1): Constant across redshift bins
- Spectroscopic redshift uncertainties (2): Linear in redshift
- Photometric galaxy bias (1): Linear interpolation of input values
- Photometric galaxy bias (2): Constant in each tomographic bin
- Photometric galaxy bias (3): Cubic polynomial in redshift
- Photometric galaxy bias (4): 1-loop perturbation theory
- Galaxy power spectrum (1): Linear theory
- Galaxy power spectrum (2): Nonlinear prescription: EFTOFLSS
- Shear calibration uncertainties: Multiplicative bias parameter for each tomographic bin
- Weak lensing generalization: Weyl power spectrum
- Photometric magnification bias (1): Linear interpolation of input values
- Photometric magnification bias (2): Constant nuisance parameter for each bin
- Photometric magnification bias (3): Cubic polynomial in redshift
- Spectroscopic magnification bias: Standard formalism
- Spectroscopic sample impurities (1): Redshift-independent outlier fraction
- Spectroscopic sample impurities (2): Outlier fraction for each redshift bin
- Data reader: Both generic and Euclid-specific data formats
- Code robustness: Unit tests, continuous integration, Docker images
- Code benchmarking: Primary Euclid observables
- Efficient integration: FFTLog
- Plotting routines: Cosmological observables and chains
- Likelihood shape (1): Gaussian (analytic covariance)
- Likelihood shape (2): Non-Gaussian (simulated covariance)
- User interface (1): Executable
- User interface (2): Jupyter demo notebook
- User interface (3): Graphical user interface for creating configuration files
- Code documentation: Docstrings (Sphinx Numpydoc)
- Sampling platform (1): Cobaya
- Sampling platform (2): CosmoSIS
- Extended cosmology (1): Evolving dark energy (w0–wa)
- Extended cosmology (2): Modified gravity (via modified growth index γMG)
- Extended cosmology (3): Nonzero curvature
- Extended cosmology (4): Sum of neutrino masses
