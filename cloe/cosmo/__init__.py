# -*- coding: utf-8 -*-
r"""COSMOLOGY MODULE

This module stores various cosmological parameters and includes
calculation of cosmological quantities not provided by Boltzmann codes.

**List of cosmological parameters implemented**


        ``H0``: float
            Present-day Hubble constant :math:`{\rm (km·s^{-1}·Mpc^{-1})}`
        ``H0_Mpc``: float
            Present-day Hubble constant :math:`{\rm (Mpc^{-1})}`
        ``omch2``: float
            Present-day CDM energy density
            :math:`\Omega_{\rm CDM}(H_0/100)^2`
        ``ombh2``: float
            Present-day baryon energy density
            :math:`\Omega_{\rm baryon}(H_0/100)^2`
        ``omkh2``: float
            Present-day curvature energy density
            :math:`\Omega_{\rm k}(H_0/100)^2`
        ``Omc``: float
            Present-day CDM energy density
            :math:`\Omega_{\rm CDM}`
        ``Omb``: float
            Present-day baryon energy density
            :math:`\Omega_{\rm baryon}`
        ``Omk``: float
            Present-day curvature energy density
            :math:`\Omega_{\rm k}`
        ``As``: float
            Amplitude of the primordial power spectrum
        ``ns``: float
            Spectral tilt of the primordial
            power spectrum
        ``sigma8_0``: float
            :math:`\sigma_8` evaluated at z = 0
        ``w``: float
           Dark energy equation of state
        ``wa``: float
           Dark energy equation of state
        ``gamma_MG``: float
           Modified Gravity :math:`\gamma` parameter
        ``omnuh2``: float
            Present-day massive neutrinos energy density
            :math:`\Omega_{\rm neutrinos}(H_0/100)^2`
        ``Omnu``: float
            Present-day massive neutrinos energy density
            :math:`\Omega_{\rm neutrinos}`
        ``Omm``: float
            Present-day total matter energy density
            :math:`\Omega_{\rm m}`
            Assumes sum of baryons, CDM and neutrinos
        ``mnu``: float
            Sum of massive neutrino species masses (eV)
        ``comov_dist``: list
            Value of comoving distances at redshifts `z_win`
        ``angular_dist``: list
            Value of angular diameter distances at redshifts `z_win`
        ``H``: list
            Hubble function evaluated at redshifts `z_win`
        ``H_Mpc``: list
            Hubble function evaluated at redshifts `z_win` in units
            of :math:`{\rm Mpc^{-1}}`
        ``Pk_delta``: function
            Interpolator function for linear matter :math:`P(k)` from
            Boltzmann code
        ``Pk_cb``: function
            Interpolator function for cdm+b :math:`P(k)` from
            Boltzmann code
        ``Pk_halomodel_recipe``: function
            Interpolator function for nonlinear matter :math:`P(k)` from
            Boltzmann code
        ``Pk_weyl``: function
            Interpolator function for linear Weyl :math:`P(k)` from
            Boltzmann code
        ``Pk_weyl_NL``: function
            Interpolator function for nonlinear Weyl :math:`P(k)` from
            Boltzmann code
        ``fsigma8``: list
            :math:`f \sigma_8` function evaluated at redshift `z`
        ``sigma8``: list
            :math:`\sigma_8` function evaluated at redshift `z`
        ``c``: float
            Speed-of-light in units of :math:`{\rm km·s^{-1}}`
        ``r_z_func``: function
            Interpolated function for comoving distance
        ``d_z_func``: function
            Interpolated function for angular diameter distance
        ``sigma8_z_func``: function
            Interpolated function for :math:`\sigma_8`
        ``fsigma8_z_func``: function
            Interpolated function for :math:`f \sigma_8`
        ``f_z``: function
            Interpolated growth rate function
        ``H_z_func``: function
            Interpolated function for Hubble parameter
        ``H_z_func_Mpc``: function
            Interpolated function for Hubble parameter :math:`{\rm Mpc^{-1}}`
        ``D_z_k_func``: function
            Interpolated function for growth factor
        ``z_win``: list
            Array of redshifts at which :math:`H` and :obj:`comov_dist`
            are evaluated at
        ``k_win``: list
            Array of wavenumbers which will be used to evaluate galaxy power
            spectra
        ``Pmm_phot``: function
            Matter-matter power spectrum for photometric probes
        ``Pgg_phot``: function
            Galaxy-galaxy power spectrum for GCphot
        ``Pgdelta_phot``: function
            Galaxy-matter power spectrum for GCphot
        ``Pgg_spectro``: function
            Galaxy-galaxy power spectrum for GCspectro
        ``Pgdelta_spectro``: function
            Galaxy-matter power spectrum for GCspectro
        ``Pii``: function
            Intrinsic alignment (intrinsic-intrinsic) power spectrum
        ``Pdeltai``: function
            Density-intrinsic cross-spectrum
        ``Pgi_phot``: function
            Photometric galaxy-intrinsic cross-spectrum
        ``Pgi_spectro``: function
            Spectroscopic galaxy-intrinsic cross-spectrum
        ``MG_mu``: function
            mu function from Modified Gravity parametrization
        ``MG_sigma``: function
            sigma function from Modified Gravity parametrization
        ``NL_boost``: float
            Nonlinear boost factor
        ``NL_flag_phot_matter``: int
            Nonlinear matter flag for 3x2pt photometric probes
        ``NL_flag_spectro``: int
            Nonlinear flag for GCspectro
        ``bias_model``: int
            bias model
        ``magbias_model``: int
            Magnification bias model
        ``luminosity_ratio_z_func``: function
            Luminosity ratio interpolator for IA model
        ``nuisance_parameters``: dict
            Contains all nuisance bias parameters
            and IA parameters which are sampled over.
            At the moment, we have implemented
            10 constant bias for the photometric
            recipe. There are 3 bias options (linear, constant,
            polynomial) for the photometric bias.
            4 for spectroscopic recipes,
            and 3 intrinsic alignment parameters.
            This dictionary also stores the choice of likelihood
            to be evaluated, i.e. photometric, spectroscopic, or 3x2pt.
            By default, if a choice isn't explicitly specified, the 3x2pt
            likelihood is calculated.
"""

__all__ = ['cosmology']  # list submodules

from . import *
