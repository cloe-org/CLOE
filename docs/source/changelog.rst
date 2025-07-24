Changelog
=========

There have been multiple internal releases of CLOE within the
Euclid Consortium, and the most recent stable version is ``v2.1.1``.
Here, we provide a brief overview of the development history,
which has thus far consisted in over 300 code mergers.

CLOE ``Version 1.0``
--------------------

This version of the code was internally released to
the Euclid Consortium in April 2021. 

It includes the basic 
aspects of CLOE, such as an interface to COBAYA (and
through it to CAMB), along with modules for reading in the
measurements and covariances, computing the theory predictions 
for the photometric and spectroscopic probes, and
calculating the likelihood. This version of the code is thereby
capable of performing an end-to-end inference that produces
parameter constraints. 

However, it also has major limitations,
such as no masking of the measurements and covariance, 
limited treatment of the systematic uncertainties, absence of 
nonlinear corrections to the theory predictions (aside from 
Halofit and HMCODE for the matter power spectrum and 
the NLA model for the intrinsic alignments), and a single 
summary statistics for the core observables (i.e. 3×2pt angular
power spectra and redshift-space multipole power spectra).

CLOE ``Version 2.0`` 
--------------------

This version of the code was internally released to the
Euclid Consortium in June 2023. There were two subsequent minor 
patches, ``v2.0.1`` and ``v2.0.2``, internally released in 
August and November 2023, respectively. 

This version of the code has major upgrades in a variety of areas. This
includes the creation of the CLOE overlayer, the interface
to the linear matter power spectrum and background expansion/growth 
histories of CLASS, the masking prescription, an expanded treatment 
of systematic uncertainties (such as magnification bias, galaxy bias, 
photometric redshift uncertainties, and sample impurities), modified 
gravity corrections to the growth history, and the capability to consider 
summary statistics in configuration space (i.e. 3×2pt correlation 
functions and multipole correlation functions). 

This version of CLOE also includes new features such as the BNT transformation, 
GUI, and redshift-space distortions for the photometric galaxy clustering 
and galaxy-galaxy lensing. This version of CLOE moreover includes emulators 
for the nonlinear matter power spectrum and the EFTOFLSS model for the nonlinear galaxy 
power spectrum. The patches ``v2.0.1`` and ``v2.0.2`` primarily incorporate a third-order polynomial in 
redshift for the magnification bias, spectroscopic redshift errors, and the CosmoSIS backend. 
We note that ``v2.0.2`` is the version of the code used to perform the parameter inference in Euclid 
Collaboration: Cañas Herrera et al. (2025). 

CLOE ``Version 2.1`` 
--------------------

This most recent stable version of the code was internally 
released to the Euclid Consortium in October 2024. 
There was a subsequent minor patch ``v2.1.1`` in July 2025.

It includes the window convolution for both the photometric and spectroscopic 
probes, i.e. summary statistics in the form of 3×2pt pseudo-Cℓ and
window-convolved multipole power spectra. It moreover includes the 
FFTLog implementation, a unified treatment of Cobaya and CosmoSIS via
the CLOE overlayer, and the ability to read in data files in
the Euclid OU-LE3 format via EuclidLib. This version of
CLOE moreover includes an optimization of the integration
strategy for the weak lensing and magnification kernels 
(resulting in a factor of six speed improvement), along with bug
fixes pertaining to the magnification bias, data handler 
module, Conda environment, measurement units, and 
checkpointing of Monte Carlo runs. This version also includes the
masking of the configuration space probes, removal of 
deprecated code, and updates to notebooks and scripts for 
compatibility with the latest version of CLOE. 

In addition, this CLOE version includes
the baryonic feedback emulators of BACCO and BCemu,
the 1-loop perturbation theory prescription for the nonlinear
photometric galaxy clustering, the TATT model for intrinsic 
alignments, and an extension of the spectroscopic galaxy
clustering modeling for compatibility with massive neutrinos. 
This version of CLOE includes the spectroscopic
magnification bias and interface to the Weyl potential power
spectrum. This version moreover includes additional probes,
in the form of CMB cross-correlations and cluster counts.
These additional probes have the same functionalities as the
primary probes in that they can be run using either Cobaya
or CosmoSIS, and using either CAMB or CLASS. 
The patch ``v2.1.1`` primarily incorporates the feature to 
write the theory predictions to file, minor bug fixes, and 
enhancements to the Sphinx documentation.
