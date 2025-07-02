"""COSMOLOGY

Class to store cosmological parameters and functions.
"""

import numpy as np
from scipy import interpolate
from astropy import constants as const
from cloe.non_linear.nonlinear import Nonlinear
from cloe.auxiliary import redshift_bins as rb
from cloe.auxiliary.logger import log_debug, log_error
from scipy.integrate import quad
from copy import deepcopy


class CosmologyError(Exception):
    r"""
    Class to define Exception Error.
    """

    pass


class Cosmology:
    r"""
    Class for cosmological observables.
    """

    def __init__(self):
        r"""
        List of cosmological parameters implemented.

        Parameters
        ----------
        H0: float
            Present-day Hubble constant :math:`{\rm (km·s^{-1}·Mpc^{-1})}`
        H0_Mpc: float
            Present-day Hubble constant :math:`{\rm (Mpc^{-1})}`
        omch2: float
            Present-day CDM energy density
            :math:`\Omega_{\rm CDM}(H_0/100)^2`
        ombh2: float
            Present-day baryon energy density
            :math:`\Omega_{\rm baryon}(H_0/100)^2`
        omkh2: float
            Present-day curvature energy density
            :math:`\Omega_{\rm k}(H_0/100)^2`
        Omc: float
            Present-day CDM energy density
            :math:`\Omega_{\rm CDM}`
        Omb: float
            Present-day baryon energy density
            :math:`\Omega_{\rm baryon}`
        Omk: float
            Present-day curvature energy density
            :math:`\Omega_{\rm k}`
        As: float
            Amplitude of the primordial power spectrum
        ns: float
            Spectral tilt of the primordial
            power spectrum
        sigma8_0: float
            :math:`\sigma_8` evaluated at z = 0
        w: float
           Dark energy equation of state
        wa: float
           Dark energy equation of state
        gamma_MG: float
           Modified Gravity :math:`\gamma` parameter
        omnuh2: float
            Present-day massive neutrinos energy density
            :math:`\Omega_{\rm neutrinos}(H_0/100)^2`
        Omnu: float
            Present-day massive neutrinos energy density
            :math:`\Omega_{\rm neutrinos}`
        Omm: float
            Present-day total matter energy density
            :math:`\Omega_{\rm m}`
            Assumes sum of baryons, CDM and neutrinos
        mnu: float
            Sum of massive neutrino species masses (eV)
        comov_dist: list
            Value of comoving distances at redshifts `z_win`
        angular_dist: list
            Value of angular diameter distances at redshifts `z_win`
        H: list
            Hubble function evaluated at redshifts `z_win`
        H_Mpc: list
            Hubble function evaluated at redshifts `z_win` in units
            of :math:`{\rm Mpc^{-1}}`
        Pk_delta_Boltzmann: function
            Interpolator function for linear matter :math:`P(k)` from
            Boltzmann code
        Pk_cb_Boltzmann: function
            Interpolator function for cdm+b :math:`P(k)` from
            Boltzmann code
        Pk_halomodel_recipe_Boltzmann: function
            Interpolator function for nonlinear matter :math:`P(k)` from
            Boltzmann code
        Pk_delta: function
            Interpolator function for linear matter :math:`P(k)`. Coincides
            with `Pk_delta_Boltzmann` if `use_gamma_MG` is False, otherwise
            returns a rescaled version according to the value of `gamma_MG`.
        Pk_cb: function
            Interpolator function for cdm+b :math:`P(k)`. Coincides with
            `Pk_cb_Boltzmann` if `use_gamma_MG` is False, otherwise returns
            a rescaled version according to the value of `gamma_MG`.
        Pk_halomodel_recipe: function
            Interpolator function for nonlinear matter :math:`P(k)`. Coincides
            with `Pk_halomodel_rcipe_Boltzmann` if `use_gamma_MG` is False,
            otherwise returns a rescaled version according to the value of
            `gamma_MG`.
        Pk_weyl: function
            Interpolator function for linear Weyl :math:`P(k)` from
            Boltzmann code
        Pk_weyl_NL: function
            Interpolator function for nonlinear Weyl :math:`P(k)` from
            Boltzmann code
        fsigma8: list
            :math:`f \sigma_8` function evaluated at redshift `z`
        sigma8: list
            :math:`\sigma_8` function evaluated at redshift `z`
        c: float
            Speed-of-light in units of :math:`{\rm km·s^{-1}}`
        r_z_func: function
            Interpolated function for comoving distance
        d_z_func: function
            Interpolated function for angular diameter distance
        sigma8_z_func: function
            Interpolated function for :math:`\sigma_8`
        fsigma8_z_func: function
            Interpolated function for :math:`f \sigma_8`
        r_win: list
            List of radii which will be used to evaluate sigmaR
        sigmaR_z_func: function
            Interpolated function for sigmaR, depending on z and R
        sigmaR_z_func_cb: function
            Interpolated function for sigmaR_cb, depending on z and R
        f_z: function
            Interpolated growth rate function
        H_z_func: function
            Interpolated function for Hubble parameter
        H_z_func_Mpc: function
            Interpolated function for Hubble parameter :math:`{\rm Mpc^{-1}}`
        _D_z_k_func: function
            Interpolated function for growth factor
        D_z_k_func: function
            Wrapper for interpolated function for growth factor, that makes
            the output type consistent with the input type.
        D_z_k_func_MG: function
            Interpolated function for growth factor (modified gravity)
        z_win: list
            Array of redshifts at which :math:`H` and :obj:`comov_dist`
            are evaluated at
        k_win: list
            Array of wavenumbers which will be used to evaluate galaxy power
            spectra
        Pmm_phot: function
            Matter-matter power spectrum for photometric probes
        Pgg_phot: function
            Galaxy-galaxy power spectrum for GCphot
        Pgdelta_phot: function
            Galaxy-matter power spectrum for GCphot
        Pgg_spectro: function
            Galaxy-galaxy power spectrum for GCspectro
        Pgdelta_spectro: function
            Galaxy-matter power spectrum for GCspectro
        Pii: function
            Intrinsic alignment (intrinsic-intrinsic) power spectrum
        Pdeltai: function
            Matter-intrinsic cross-spectrum
        Pgi_phot: function
            Photometric galaxy-intrinsic cross-spectrum
        Pgi_spectro: function
            Spectroscopic galaxy-intrinsic cross-spectrum
        MG_mu: function
            mu function from Modified Gravity parametrization
        MG_sigma: function
            sigma function from Modified Gravity parametrization
        NL_boost: float
            Nonlinear boost factor
        NL_flag_phot_matter: int
            Nonlinear matter flag for 3x2pt photometric probes
        NL_flag_spectro: int
            Nonlinear flag for GCspectro
        IA_flag: int
            intrinsic alignment model flag
        IR_resum: str
            IR-resummation model flag
        print_theory: bool
            Print theory predictions to file
        a00e: function
            a00e one-loop term for IA TATT model (arxiv:1708.09247)
        c00e: function
            c00e one-loop term for IA TATT model (arxiv:1708.09247)
        a0e0e: function
             a0e0 eone-loop term for IA TATT model (arxiv:1708.09247)
        a0b0b: function
            a0b0b one-loop term for IA TATT model (arxiv:1708.09247)
        ae2e2: function
            ae2e2 one-loop term for IA TATT model (arxiv:1708.09247)
        ab2b2: function
            ab2b2 one-loop term for IA TATT model (arxiv:1708.09247)
        a0e2: function
            a0e2 one-loop term for IA TATT model (arxiv:1708.09247)
        b0e2: function
            b0e2 one-loop term for IA TATT model (arxiv:1708.09247)
        d0ee2: function
            d0ee2 one-loop term for IA TATT model (arxiv:1708.09247)
        d0bb2: function
            d0bb2 one-loop term for IA TATT model (arxiv:1708.09247)
        bias_model: int
            bias model
        magbias_model: int
            Magnification bias model
        luminosity_ratio_z_func: function
            Luminosity ratio interpolator for IA model
        nuisance_parameters: dict
            Contains all nuisance bias parameters
            and IA parameters which are sampled over.
            At the moment, we have implemented
            10 constant bias for photo-z
            recipe and 4 for spectro recipe,
            and 3 IA parameters for the NLA model
            while 5 for the TATT model. The
            initialized values of the fiducial
            cosmology dictionary corresponds to:

                * Photo-z values corresponding to
                :math:`b_{(x,i)}=\sqrt{1+\bar{b}_{(x,i)}}`

                There are 3 bias options (linear, constant, polynomial)

                * Spectroscopic bias values in arXiv:1910.09273

                * IA values in arXiv:1910.09273 (NLA)
                and arxiv:1708.09247 (TATT)

                * Additional parameters for GCspectro, as provided by IST:NL

            This dictionary also stores the choice of likelihood
            to be evaluated, i.e. photometric, spectroscopic, or 3x2pt.
            By default the 3x2pt likelihood is calculated.
        """
        # Initialize cosmo dictionary
        # Note: added speed of light to dictionary. It is in
        # units of km/s to be dimensionally consistent with H0
        self.cosmo_dic = {  # Constants
                          'H0': 67.0,
                          'omch2': 0.122,
                          'ombh2': 0.022,
                          'omnuh2': 0.000644,
                          'Omnu': 0.00143715,
                          'Omk': 0.0,
                          'Omm': 0.32,
                          'Omc': 0.27,
                          'Omb': 0.05,
                          'w': -1.0,
                          'wa': 0.0,
                          'gamma_MG': 0.55,
                          'mnu': 0.06,
                          'tau': 0.07,
                          'nnu': 3.046,
                          'ns': 0.96,
                          'As': 2.1e-9,
                          'sigma8_0': 0.816,
                          'c': const.c.to('km/s').value,
                          'MG_mu': None,
                          'MG_sigma': None,
                          # Lists
                          'z_win': None,
                          'z_win_max': None,
                          'k_win': None,
                          'comov_dist': None,
                          'angular_dist': None,
                          'H': None,
                          'H_Mpc': None,
                          'fsigma8': None,
                          'sigma8': None,
                          'D_z_k': None,
                          # Interpolators
                          'Pk_delta_Boltzmann': None,
                          'Pk_cb_Boltzmann': None,
                          'Pk_halomodel_recipe_Boltzmann': None,
                          'Pk_delta': None,
                          'Pk_cb': None,
                          'Pk_halomodel_recipe': None,
                          'Pk_weyl': None,
                          'Pk_weyl_NL': None,
                          'Pmm_phot': None,
                          'Pgg_phot': None,
                          'Pgdelta_phot': None,
                          'Pgg_spectro': None,
                          'Pgdelta_spectro': None,
                          'Pii': None,
                          'Pdeltai': None,
                          'Pgi_phot': None,
                          'Pgi_spectro': None,
                          'r_z_func': None,
                          'z_r_func': None,
                          'f_K_z_func': None,
                          '_f_K_z12_func': None,
                          'f_K_z12_func': None,
                          'd_z_func': None,
                          'H_z_func': None,
                          'H_z_func_Mpc': None,
                          '_D_z_k_func': None,
                          'D_z_k_func': None,
                          'D_z_k_func_MG': None,
                          'sigma8_z_func': None,
                          'fsigma8_z_func': None,
                          'f_z': None,
                          # For galaxy clusters
                          'r_win': None,
                          'sigmaR_z_func': None,
                          'sigmaR_z_func_cb': None,
                          'luminosity_ratio_z_func': None,
                          'Weyl_matter_ratio': None,
                          # NL_boost
                          'NL_boost': None,
                          # NL flags
                          'NL_flag_phot_matter': 0,
                          'NL_flag_phot_bias': 0,
                          'NL_flag_spectro': 0,
                          # IA flag
                          'IA_flag': 0,
                          # IR-resummation flag
                          'IR_resum': 'DST',
                          # Baryonic feedback flag
                          'NL_flag_phot_baryon': 0,
                          # Baryonic feedback z-dependence model flag
                          'Baryon_redshift_model': True,
                          # bias model
                          # 1 => linear interpolation,
                          # 2 => constant in bins,
                          # 3 => 3rd order polynomial,
                          'bias_model': 1,
                          # magnification bias model
                          # 1 => linear interpolation,
                          # 2 => constant in bins,
                          # 3 => cubic dependence of redshift,
                          'magbias_model': 2,
                          # Use Modified Gravity gamma
                          'use_gamma_MG': 0,
                          # use magnification bias for GC spectro
                          'use_magnification_bias_spectro': 0,
                          # Use Weyl power spectrum (workaround approach)
                          'use_Weyl': False,
                          # Print theory predictions
                          'print_theory': False,
                          # Redshift dependent purity correction
                          'f_out_z_dep': False,
                          # One-loop order terms for the TATT model
                          'a00e': None,
                          'c00e': None,
                          'a0e0e': None,
                          'a0b0b': None,
                          'ae2e2': None,
                          'ab2b2': None,
                          'a0e2': None,
                          'b0e2': None,
                          'd0ee2': None,
                          'd0bb2': None,
                          # Spectroscopic galaxy clustering redshift error
                          'GCsp_z_err': False,
                          'nuisance_parameters': {
                             # Intrinsic alignments (NLA and TATT)
                             # Set a2_ia, b1_ia and eta2_ia to 0 for NLA
                             'a1_ia': 1.72,
                             'a2_ia': 2,
                             'b1_ia': 1,
                             'eta1_ia': -0.41,
                             'eta2_ia': 1,
                             'beta1_ia': 0.0,
                             'pivot_redshift': 0.,
                             # Linear photometric galaxy bias (IST:F case)
                             'b1_photo_bin1': 1.03047,
                             'b1_photo_bin2': 1.06699,
                             'b1_photo_bin3': 1.17363,
                             'b1_photo_bin4': 1.23340,
                             'b1_photo_bin5': 1.27510,
                             'b1_photo_bin6': 1.28574,
                             'b1_photo_bin7': 1.39434,
                             'b1_photo_bin8': 1.49170,
                             'b1_photo_bin9': 1.52334,
                             'b1_photo_bin10': 1.54639,
                             'b1_photo_bin11': 1.68682,
                             'b1_photo_bin12': 2.03066,
                             'b1_photo_bin13': 2.57812,
                             # Linear photometric galaxy bias (polynomial case)
                             'b1_0_poly_photo': 0.830703,
                             'b1_1_poly_photo': 1.190547,
                             'b1_2_poly_photo': -0.928357,
                             'b1_3_poly_photo': 0.423292,
                             # Quadratic photometric galaxy bias
                             'b2_photo_bin1': 0.0,
                             'b2_photo_bin2': 0.0,
                             'b2_photo_bin3': 0.0,
                             'b2_photo_bin4': 0.0,
                             'b2_photo_bin5': 0.0,
                             'b2_photo_bin6': 0.0,
                             'b2_photo_bin7': 0.0,
                             'b2_photo_bin8': 0.0,
                             'b2_photo_bin9': 0.0,
                             'b2_photo_bin10': 0.0,
                             'b2_photo_bin11': 0.0,
                             'b2_photo_bin12': 0.0,
                             'b2_photo_bin13': 0.0,
                             # Quadratic photometric galaxy bias (poly case)
                             'b2_0_poly_photo': 0.0,
                             'b2_1_poly_photo': 0.0,
                             'b2_2_poly_photo': 0.0,
                             'b2_3_poly_photo': 0.0,
                             # Non-local photometric galaxy biases
                             'bG2_photo_bin1': 0.0, 'bG3_photo_bin1': 0.0,
                             'bG2_photo_bin2': 0.0, 'bG3_photo_bin2': 0.0,
                             'bG2_photo_bin3': 0.0, 'bG3_photo_bin3': 0.0,
                             'bG2_photo_bin4': 0.0, 'bG3_photo_bin4': 0.0,
                             'bG2_photo_bin5': 0.0, 'bG3_photo_bin5': 0.0,
                             'bG2_photo_bin6': 0.0, 'bG3_photo_bin6': 0.0,
                             'bG2_photo_bin7': 0.0, 'bG3_photo_bin7': 0.0,
                             'bG2_photo_bin8': 0.0, 'bG3_photo_bin8': 0.0,
                             'bG2_photo_bin9': 0.0, 'bG3_photo_bin9': 0.0,
                             'bG2_photo_bin10': 0.0, 'bG3_photo_bin10': 0.0,
                             'bG2_photo_bin11': 0.0, 'bG3_photo_bin11': 0.0,
                             'bG2_photo_bin12': 0.0, 'bG3_photo_bin12': 0.0,
                             'bG2_photo_bin13': 0.0, 'bG3_photo_bin13': 0.0,
                            # Non-local photometric galaxy bias (poly case)
                             'bG2_0_poly_photo': 0.0,
                             'bG2_1_poly_photo': 0.0,
                             'bG2_2_poly_photo': 0.0,
                             'bG2_3_poly_photo': 0.0,
                             'bG3_0_poly_photo': 0.0,
                             'bG3_1_poly_photo': 0.0,
                             'bG3_2_poly_photo': 0.0,
                             'bG3_3_poly_photo': 0.0,
                             # Magnification bias
                             'magnification_bias_1': -0.8499,
                             'magnification_bias_2': -1.023,
                             'magnification_bias_3': -0.9474,
                             'magnification_bias_4': -0.9837,
                             'magnification_bias_5': -0.7699,
                             'magnification_bias_6': -0.7137,
                             'magnification_bias_7': -0.52345,
                             'magnification_bias_8': -0.40845,
                             'magnification_bias_9': -0.41045,
                             'magnification_bias_10': -0.0824,
                             'magnification_bias_11': 0.4917,
                             'magnification_bias_12': 0.98555,
                             'magnification_bias_13': 1.6056,
                             # Magnification bias (polynomial case)
                             'b0_mag_poly': -1.50685,
                             'b1_mag_poly': 1.35034,
                             'b2_mag_poly': 0.08321,
                             'b3_mag_poly': 0.04279,
                             # Multiplicative bias
                             'multiplicative_bias_1': 0.0,
                             'multiplicative_bias_2': 0.0,
                             'multiplicative_bias_3': 0.0,
                             'multiplicative_bias_4': 0.0,
                             'multiplicative_bias_5': 0.0,
                             'multiplicative_bias_6': 0.0,
                             'multiplicative_bias_7': 0.0,
                             'multiplicative_bias_8': 0.0,
                             'multiplicative_bias_9': 0.0,
                             'multiplicative_bias_10': 0.0,
                             'multiplicative_bias_11': 0.0,
                             'multiplicative_bias_12': 0.0,
                             'multiplicative_bias_13': 0.0,
                             # Spectroscopic galaxy bias
                             'b1_spectro_bin1': 1.4614804,
                             'b1_spectro_bin2': 1.6060949,
                             'b1_spectro_bin3': 1.7464790,
                             'b1_spectro_bin4': 1.8988660,
                             'b2_spectro_bin1': 0.0,
                             'b2_spectro_bin2': 0.0,
                             'b2_spectro_bin3': 0.0,
                             'b2_spectro_bin4': 0.0,
                             'bG2_spectro_bin1': 0.0,
                             'bG2_spectro_bin2': 0.0,
                             'bG2_spectro_bin3': 0.0,
                             'bG2_spectro_bin4': 0.0,
                             'bG3_spectro_bin1': 0.0,
                             'bG3_spectro_bin2': 0.0,
                             'bG3_spectro_bin3': 0.0,
                             'bG3_spectro_bin4': 0.0,
                             # Finger of God counterterms
                             'c0_spectro_bin1': 0.0,
                             'c0_spectro_bin2': 0.0,
                             'c0_spectro_bin3': 0.0,
                             'c0_spectro_bin4': 0.0,
                             'c2_spectro_bin1': 0.0,
                             'c2_spectro_bin2': 0.0,
                             'c2_spectro_bin3': 0.0,
                             'c2_spectro_bin4': 0.0,
                             'c4_spectro_bin1': 0.0,
                             'c4_spectro_bin2': 0.0,
                             'c4_spectro_bin3': 0.0,
                             'c4_spectro_bin4': 0.0,
                             'ck4_spectro_bin1': 0.0,
                             'ck4_spectro_bin2': 0.0,
                             'ck4_spectro_bin3': 0.0,
                             'ck4_spectro_bin4': 0.0,
                             # Shot noise parameters
                             'aP_spectro_bin1': 0.0,
                             'aP_spectro_bin2': 0.0,
                             'aP_spectro_bin3': 0.0,
                             'aP_spectro_bin4': 0.0,
                             'Psn_spectro_bin1': 0.0,
                             'Psn_spectro_bin2': 0.0,
                             'Psn_spectro_bin3': 0.0,
                             'Psn_spectro_bin4': 0.0,
                             'e0k2_spectro_bin1': 0.0,
                             'e0k2_spectro_bin2': 0.0,
                             'e0k2_spectro_bin3': 0.0,
                             'e0k2_spectro_bin4': 0.0,
                             'e2k2_spectro_bin1': 0.0,
                             'e2k2_spectro_bin2': 0.0,
                             'e2k2_spectro_bin3': 0.0,
                             'e2k2_spectro_bin4': 0.0,
                             # Purity of spectroscopic samples
                             'f_out': 0.0,
                             'f_out_1': 0.0,
                             'f_out_2': 0.0,
                             'f_out_3': 0.0,
                             'f_out_4': 0.0,
                             # Spectroscopic galaxy magnification bias
                             'magnification_bias_spectro_bin1': 0.79,
                             'magnification_bias_spectro_bin2': 0.87,
                             'magnification_bias_spectro_bin3': 0.96,
                             'magnification_bias_spectro_bin4': 0.98,
                             # Example BCemu baryon params (fitted to BAHAMAS)
                             'log10Mc_bcemu_bin1': 13.402342835406102,
                             'log10Mc_bcemu_bin2': 13.255603572838046,
                             'log10Mc_bcemu_bin3': 13.39534479993441,
                             'log10Mc_bcemu_bin4': 13.491189798802624,
                             'log10Mc_bcemu_bin5': 13.517099731729203,
                             'log10Mc_bcemu_bin6': 13.431948448938623,
                             'log10Mc_bcemu_bin7': 13.159142065726643,
                             'log10Mc_bcemu_bin8': 12.630530198678242,
                             'log10Mc_bcemu_bin9': 11.768928839012311,
                             'log10Mc_bcemu_bin10': 11.123969034839764,
                             'log10Mc_bcemu_bin11': 11.123969034839764,
                             'log10Mc_bcemu_bin12': 11.123969034839764,
                             'log10Mc_bcemu_bin13': 11.123969034839764,
                             'mu_bcemu_bin1': 1.081169575215787,
                             'mu_bcemu_bin2': 1.0748796706532235,
                             'mu_bcemu_bin3': 1.257505674386812,
                             'mu_bcemu_bin4': 1.3953104854720346,
                             'mu_bcemu_bin5': 1.4707848100185459,
                             'mu_bcemu_bin6': 1.451326233210004,
                             'mu_bcemu_bin7': 1.2738670043792995,
                             'mu_bcemu_bin8': 0.8851923901330802,
                             'mu_bcemu_bin9': 0.23572246186875243,
                             'mu_bcemu_bin10': -0.006953332782079408,
                             'mu_bcemu_bin11': -0.006953332782079408,
                             'mu_bcemu_bin12': -0.006953332782079408,
                             'mu_bcemu_bin13': -0.006953332782079408,
                             'thej_bcemu_bin1': 3.6711008267835776,
                             'thej_bcemu_bin2': 4.0853297524846175,
                             'thej_bcemu_bin3': 4.1001870114090835,
                             'thej_bcemu_bin4': 4.085897535238355,
                             'thej_bcemu_bin5': 4.062128106370927,
                             'thej_bcemu_bin6': 4.0416849190138695,
                             'thej_bcemu_bin7': 4.0405617133891685,
                             'thej_bcemu_bin8': 4.073797196972992,
                             'thej_bcemu_bin9': 4.168188042786498,
                             'thej_bcemu_bin10': 4.567495847826163,
                             'thej_bcemu_bin11': 4.567495847826163,
                             'thej_bcemu_bin12': 4.567495847826163,
                             'thej_bcemu_bin13': 4.567495847826163,
                             'gamma_bcemu_bin1': 2.732738146480909,
                             'gamma_bcemu_bin2': 2.794658925091356,
                             'gamma_bcemu_bin3': 3.008141783989797,
                             'gamma_bcemu_bin4': 3.2186226058208787,
                             'gamma_bcemu_bin5': 3.425034377127418,
                             'gamma_bcemu_bin6': 3.6237277892257937,
                             'gamma_bcemu_bin7': 3.805233587893605,
                             'gamma_bcemu_bin8': 3.952453083293017,
                             'gamma_bcemu_bin9': 4.012632091622541,
                             'gamma_bcemu_bin10': 3.1454520255604006,
                             'gamma_bcemu_bin11': 3.1454520255604006,
                             'gamma_bcemu_bin12': 3.1454520255604006,
                             'gamma_bcemu_bin13': 3.1454520255604006,
                             'delta_bcemu_bin1': 6.407442578598766,
                             'delta_bcemu_bin2': 7.0846518716943025,
                             'delta_bcemu_bin3': 7.226700228338714,
                             'delta_bcemu_bin4': 7.343194626402527,
                             'delta_bcemu_bin5': 7.470370166898234,
                             'delta_bcemu_bin6': 7.638743359373677,
                             'delta_bcemu_bin7': 7.891612514615537,
                             'delta_bcemu_bin8': 8.252389998618296,
                             'delta_bcemu_bin9': 8.702184867164187,
                             'delta_bcemu_bin10': 7.994110032082859,
                             'delta_bcemu_bin11': 7.994110032082859,
                             'delta_bcemu_bin12': 7.994110032082859,
                             'delta_bcemu_bin13': 7.994110032082859,
                             'eta_bcemu_bin1': 0.20680675958346154,
                             'eta_bcemu_bin2': 0.22813237994091348,
                             'eta_bcemu_bin3': 0.21989900653826788,
                             'eta_bcemu_bin4': 0.2090499814223423,
                             'eta_bcemu_bin5': 0.19645554090162448,
                             'eta_bcemu_bin6': 0.18238365514915927,
                             'eta_bcemu_bin7': 0.16686344766458094,
                             'eta_bcemu_bin8': 0.14883503959461816,
                             'eta_bcemu_bin9': 0.1240089652119986,
                             'eta_bcemu_bin10': 0.044931761090432315,
                             'eta_bcemu_bin11': 0.044931761090432315,
                             'eta_bcemu_bin12': 0.044931761090432315,
                             'eta_bcemu_bin13': 0.044931761090432315,
                             'deta_bcemu_bin1': 0.08189803795251066,
                             'deta_bcemu_bin2': 0.05028436917683151,
                             'deta_bcemu_bin3': 0.04842000359519622,
                             'deta_bcemu_bin4': 0.04869423933122641,
                             'deta_bcemu_bin5': 0.04959595282843388,
                             'deta_bcemu_bin6': 0.05011878848954406,
                             'deta_bcemu_bin7': 0.04907477885822071,
                             'deta_bcemu_bin8': 0.04681178740273659,
                             'deta_bcemu_bin9': 0.048062862071344824,
                             'deta_bcemu_bin10': 0.14550946518064195,
                             'deta_bcemu_bin11': 0.14550946518064195,
                             'deta_bcemu_bin12': 0.14550946518064195,
                             'deta_bcemu_bin13': 0.14550946518064195,
                             # BCemu params for the redshift parametrization
                             # (based on arxiv:2108.08863 and test chains)
                             'log10Mc_bcemu_0': 13.32,
                             'nu_log10Mc_bcemu': -0.15,
                             'thej_bcemu_0': 3.5,
                             'nu_thej_bcemu': 0.0,
                             'deta_bcemu_0': 0.2,
                             'nu_deta_bcemu': 0.6,
                             'mu_bcemu_0': 1.0,
                             'nu_mu_bcemu': 0.0,
                             'gamma_bcemu_0': 2.5,
                             'nu_gamma_bcemu': 0.0,
                             'delta_bcemu_0': 7.0,
                             'nu_delta_bcemu': 0.0,
                             'eta_bcemu_0': 0.2,
                             'nu_eta_bcemu': 0.0,
                             # Bacco baryon parameters (fitted to BAHAMAS)
                             'M_c_bacco_bin1': 14.158079090897681,
                             'M_c_bacco_bin2': 14.413536974582364,
                             'M_c_bacco_bin3': 14.513655381066439,
                             'M_c_bacco_bin4': 14.59501140993621,
                             'M_c_bacco_bin5': 14.667602932620955,
                             'M_c_bacco_bin6': 14.736117785720888,
                             'M_c_bacco_bin7': 14.803902874910724,
                             'M_c_bacco_bin8': 14.873354165369339,
                             'M_c_bacco_bin9': 14.947678648592403,
                             'M_c_bacco_bin10': 15.179943239223197,
                             'M_c_bacco_bin11': 15.179943239223197,
                             'M_c_bacco_bin12': 15.179943239223197,
                             'M_c_bacco_bin13': 15.179943239223197,
                             'eta_bacco_bin1': -0.34397323435383403,
                             'eta_bacco_bin2': -0.3823001659654948,
                             'eta_bacco_bin3': -0.3666759403458338,
                             'eta_bacco_bin4': -0.34766555599217897,
                             'eta_bacco_bin5': -0.32723937222645033,
                             'eta_bacco_bin6': -0.3066851485202732,
                             'eta_bacco_bin7': -0.28793191687619096,
                             'eta_bacco_bin8': -0.2761849439738153,
                             'eta_bacco_bin9': -0.29044476508897765,
                             'eta_bacco_bin10': -0.3012527587604943,
                             'eta_bacco_bin11': -0.3012527587604943,
                             'eta_bacco_bin12': -0.3012527587604943,
                             'eta_bacco_bin13': -0.3012527587604943,
                             'beta_bacco_bin1': -0.1401597529600361,
                             'beta_bacco_bin2': -0.14602731261184787,
                             'beta_bacco_bin3': -0.15945179021822595,
                             'beta_bacco_bin4': -0.1717165402260191,
                             'beta_bacco_bin5': -0.18260340867450087,
                             'beta_bacco_bin6': -0.1914516575772147,
                             'beta_bacco_bin7': -0.19665359534532956,
                             'beta_bacco_bin8': -0.1938633002915493,
                             'beta_bacco_bin9': -0.16799956045034847,
                             'beta_bacco_bin10': -0.1180328507054961,
                             'beta_bacco_bin11': -0.1180328507054961,
                             'beta_bacco_bin12': -0.1180328507054961,
                             'beta_bacco_bin13': -0.1180328507054961,
                             'M1_z0_cen_bacco_bin1': 11.340025775544985,
                             'M1_z0_cen_bacco_bin2': 11.30961906881156,
                             'M1_z0_cen_bacco_bin3': 11.366002025818862,
                             'M1_z0_cen_bacco_bin4': 11.423530231429538,
                             'M1_z0_cen_bacco_bin5': 11.479266623369345,
                             'M1_z0_cen_bacco_bin6': 11.529947744472405,
                             'M1_z0_cen_bacco_bin7': 11.569052829475833,
                             'M1_z0_cen_bacco_bin8': 11.578833725581436,
                             'M1_z0_cen_bacco_bin9': 11.496367669453079,
                             'M1_z0_cen_bacco_bin10': 11.360811343412026,
                             'M1_z0_cen_bacco_bin11': 11.360811343412026,
                             'M1_z0_cen_bacco_bin12': 11.360811343412026,
                             'M1_z0_cen_bacco_bin13': 11.360811343412026,
                             'theta_inn_bacco_bin1': -0.9790217925155928,
                             'theta_inn_bacco_bin2': -1.3854647301685985,
                             'theta_inn_bacco_bin3': -1.3604452515853098,
                             'theta_inn_bacco_bin4': -1.2949667258085438,
                             'theta_inn_bacco_bin5': -1.2056425079533284,
                             'theta_inn_bacco_bin6': -1.0991845011332575,
                             'theta_inn_bacco_bin7': -0.9805620838281162,
                             'theta_inn_bacco_bin8': -0.8624918408856748,
                             'theta_inn_bacco_bin9': -0.7968588249755786,
                             'theta_inn_bacco_bin10': -0.5312787510657424,
                             'theta_inn_bacco_bin11': -0.5312787510657424,
                             'theta_inn_bacco_bin12': -0.5312787510657424,
                             'theta_inn_bacco_bin13': -0.5312787510657424,
                             'M_inn_bacco_bin1': 12.472,
                             'M_inn_bacco_bin2': 11.098133175636903,
                             'M_inn_bacco_bin3': 10.685545014634902,
                             'M_inn_bacco_bin4': 10.379401082211185,
                             'M_inn_bacco_bin5': 10.124549632511826,
                             'M_inn_bacco_bin6': 9.89485323082124,
                             'M_inn_bacco_bin7': 9.669674533440968,
                             'M_inn_bacco_bin8': 9.423522683043274,
                             'M_inn_bacco_bin9': 9.086255671995564,
                             'M_inn_bacco_bin10': 8.119132619254136,
                             'M_inn_bacco_bin11': 8.119132619254136,
                             'M_inn_bacco_bin12': 8.119132619254136,
                             'M_inn_bacco_bin13': 8.119132619254136,
                             'theta_out_bacco_bin1': 0.2672123790126238,
                             'theta_out_bacco_bin2': 0.21845633941784046,
                             'theta_out_bacco_bin3': 0.22229391886311567,
                             'theta_out_bacco_bin4': 0.23124624733764712,
                             'theta_out_bacco_bin5': 0.24338032280596764,
                             'theta_out_bacco_bin6': 0.25802109222979985,
                             'theta_out_bacco_bin7': 0.27483978056155306,
                             'theta_out_bacco_bin8': 0.29287229590732955,
                             'theta_out_bacco_bin9': 0.3076030069491944,
                             'theta_out_bacco_bin10': 0.3569278707998519,
                             'theta_out_bacco_bin11': 0.3569278707998519,
                             'theta_out_bacco_bin12': 0.3569278707998519,
                             'theta_out_bacco_bin13': 0.3569278707998519,
                             # Bacco params for the redshift parametrization
                             # (based on arxiv:2011.15018 and test chains )
                             'M_c_bacco_0': 14,
                             'nu_M_c_bacco': -0.15,
                             'eta_bacco_0': -0.3,
                             'nu_eta_bacco': 0.0,
                             'beta_bacco_0': -0.22,
                             'nu_beta_bacco': 0.0,
                             'M1_z0_cen_bacco_0': 10.5,
                             'nu_M1_z0_cen_bacco': 0.0,
                             'theta_inn_bacco_0': -0.86,
                             'nu_theta_inn_bacco': 0.0,
                             'M_inn_bacco_0': 13.4,
                             'nu_M_inn_bacco': 0.0,
                             'theta_out_bacco_0': 0.25,
                             'nu_theta_out_bacco': 0.0,
                             # HMcode2020_feedback parameter
                             # (from arxiv:2009.01858)
                             'HMCode_logT_AGN': 7.8,
                             # HMcode2016 parameters (values for matter-only
                             # from arxiv:1505.07833)
                             'HMCode_eta_baryon': 0.603,
                             'HMCode_A_baryon': 3.13,
                             # 1-point redshift error dispersion for GCspectro
                             'sigma_z': 0.002,
                             # Redshift distribution shifts
                             'dz_1_GCphot': -0.02222, 'dz_1_WL': -0.02222,
                             'dz_2_GCphot': -0.02155, 'dz_2_WL': -0.02155,
                             'dz_3_GCphot': -0.01588, 'dz_3_WL': -0.01588,
                             'dz_4_GCphot': 0.00339, 'dz_4_WL': 0.00339,
                             'dz_5_GCphot': 0.00196, 'dz_5_WL': 0.00196,
                             'dz_6_GCphot': -0.00653, 'dz_6_WL': -0.00653,
                             'dz_7_GCphot': -0.00996, 'dz_7_WL': -0.00996,
                             'dz_8_GCphot': -0.01318, 'dz_8_WL': -0.01318,
                             'dz_9_GCphot': -0.00929, 'dz_9_WL': -0.00929,
                             'dz_10_GCphot': -0.02095, 'dz_10_WL': -0.02095,
                             'dz_11_GCphot': -0.00782, 'dz_11_WL': -0.00782,
                             'dz_12_GCphot': -0.00108, 'dz_12_WL': -0.00108,
                             'dz_13_GCphot': -0.04686, 'dz_13_WL': -0.04686}
                             }

        self.cosmo_dic['H0_Mpc'] = (self.cosmo_dic['H0'] /
                                    const.c.to('km/s').value)
        self.nonlinear = Nonlinear(self.cosmo_dic)

    @property
    def pk_source_phot(self):
        r"""Identifier for linear vs nonlinear class for photometric probes.

        Selects either the same Cosmology class from which it is called
        or the attribute corresponding to the instance of a :obj:`non_linear`
        class, based on the value of the nonlinear flag for the
        photometric probes.
        """

        if self.cosmo_dic['NL_flag_phot_matter'] == 0:
            return self
        else:
            return self.nonlinear

    @property
    def pk_source_spectro(self):
        r"""Identifier for linear vs nonlinear class.

        Selects either the same :obj:`cosmology` class from which it is called
        or the attribute corresponding to the instance of a :obj:`non_linear`
        class, based on the value of the nonlinear flag for the spectroscopic
        probes.
        """
        if self.cosmo_dic['NL_flag_spectro'] == 0:
            return self
        else:
            return self.nonlinear

    def matter_density(self, zs):
        r"""
        Computes the matter density as

        .. math::
            \Omega_{\rm m}(z) = \Omega_{{\rm m},0}(1+z)^3H_0^2/H^2(z)

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts for the matter density

        Returns
        -------
        Matter density parameter: numpy.ndarray
            Matter density as a function of redshift

        """
        H_frac = (self.cosmo_dic['H0'] / self.cosmo_dic['H'])**2
        return self.cosmo_dic['Omm'] * (1 + zs)**3 * H_frac

    def growth_factor(self, zs, ks):
        r"""
        Computes growth factor according to

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts for the power spectrum
        ks: numpy.ndarray
            List of modes for the power spectrum

        Returns
        -------
        Growth factor: numpy.ndarray
            Growth factor as function of redshift and wavenumber

        """
        # This function will be updated.
        # We want to obtain delta directly from Cobaya.
        # Here depends on z and k.
        try:
            power_interp = self.cosmo_dic['Pk_delta_Boltzmann']
            P_z_k = power_interp.P(zs, ks)
            D_z_k = np.sqrt(P_z_k / power_interp.P(0.0, ks))
            return D_z_k
        except CosmologyError:
            ('Computation error in D(z, k)')

    def growth_rate(self, zs, ks):
        r"""Growth rate.

        Adds an interpolator for the growth rate (this function is actually
        deprecated since we use the growth rate directly from Cobaya).

        .. math::
            f(z, k) &=-\frac{(1+z)}{D(z,k)}\frac{dD(z, k)}{dz}\\

        Parameters
        ----------
        zs: list
            List of redshift for the power spectrum
        ks: float
            Mode for the power spectrum

        Returns
        -------
        Growth rate: object
            Interpolator growth rate as function of redshift and wavenumber

        """
        # To be updated.
        # We want to obtain delta directly from Cobaya.
        # This function depends on both z and k.
        # Here 1 + z = 1 / a where a is the scale factor
        D_z_k = self.growth_factor(zs, ks)
        # This will work when k is fixed, not an array
        try:
            f_z_k = -(1 + zs) * np.gradient(D_z_k, zs[1] - zs[0]) / D_z_k
            return interpolate.InterpolatedUnivariateSpline(
                x=zs, y=f_z_k, ext=2)
        except CosmologyError:
            log_error('Computation error in f(z, k)')
            log_debug('Check k is a scalar, not an array')

    def interp_growth_rate(self):
        r"""Interpolates the growth rate.

        Adds an interpolator for the growth rate (function of redshift)
        to the cosmo dictionary. The growth rate is defined depending on
        the value of the `use_gamma_MG` flag, as either

        .. math::
                       f(z) &=f\sigma_8(z) / \sigma_8(z)\\

        or

        .. math::
                f(z;\gamma_{\rm MG})=[\Omega_{\rm m}(z)]^{\gamma_{\rm MG}}

        """
        z_win = self.cosmo_dic['z_win']
        if self.cosmo_dic['use_gamma_MG']:
            growth = self.matter_density(z_win)**self.cosmo_dic['gamma_MG'] + \
                     (self.cosmo_dic['gamma_MG'] + 4.0 / 7.0) * \
                     self.cosmo_dic['Omk']
        else:
            fs8 = self.cosmo_dic['fsigma8_z_func'](z_win)
            s8 = self.cosmo_dic['sigma8_z_func'](z_win)
            growth = fs8 / s8
        self.cosmo_dic['f_z'] = \
            interpolate.InterpolatedUnivariateSpline(x=z_win, y=growth, ext=2)

    def _growth_integrand_MG(self, z_prime):
        r"""Integrand function for the :obj:`growth_factor_MG`.

        .. math::
              \frac{f(z';\gamma_{\rm MG})}{1+z'}

        Parameters
        ----------
        z_prime: float
           Integrand variable (redshift)
        """
        if self.cosmo_dic['use_gamma_MG']:
            growth = self.cosmo_dic['f_z'](z_prime)
        else:
            growth = self.matter_density(z_prime)**0.55
        return growth / (1.0 + z_prime)

    def growth_factor_MG(self):
        r"""
        Computes the growth factor using the :math:`\gamma_{\rm MG}` as

        .. math::
           D(z;\gamma_{\rm MG}) = {\rm exp}\left[\int_z^\infty{\rm d}z' \
               \frac{f(z';\gamma_{\rm MG})}{1+z'}\right]

        and normalizes as for :math:`D(z)/D(0)`.

        Returns
        -------
        Growth factor for MG: numpy.ndarray
            Values of the growth factor using the modified
            gravity parameter :math:`\gamma_{\rm MG}`
        """
        integral = [quad(self._growth_integrand_MG, z,
                         self.cosmo_dic['z_win'][-1])[0] for z in
                    self.cosmo_dic['z_win']]
        return np.exp(integral) / np.exp(integral[0])

    def assign_growth_factor(self):
        """Interpolates the growth factor.

        Adds an interpolator for the growth factor (function of redshift and
        scale) to the cosmo dictionary.
        """

        if self.cosmo_dic['use_gamma_MG']:
            z_win = self.cosmo_dic['z_win']
            self.cosmo_dic['D_z_k_func_MG'] = \
                interpolate.interp1d(z_win, self.growth_factor_MG(),
                                     kind="cubic")

        self.cosmo_dic['D_z_k_func'] = self.growth_factor

    def interp_comoving_dist(self):
        """Interpolates the comoving distance.

        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explicitly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates comoving distance as a function of redshift.
        """

        self.cosmo_dic['r_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win_max'],
            y=self.cosmo_dic['comov_dist'],
            ext=2
        )

    def interp_z_of_r(self):
        """Interpolates the redshift.

        Adds an interpolator for the redshift as a function of the
        comoving distance to the dictionary.

        Updates 'key' in the :obj:`cosmo_dic` attribute of the class
        by adding an interpolator object
        which interpolates redshift as a function of comoving distance.

        Note: The interpolator is used in in photo.py member functions
        ``z_plus1`` and ``z_minus1``. There the values are extrapolated
        for values larger than :math:`r_{max}` since the corresponding
        multipole factor is larger than unity. Therefore, we set here the
        extrapolation mode to return zeros and not raise a ValueError as
        everywhere else in the code, so that the code runs smoothly with RSD.
        """
        self.cosmo_dic['z_r_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['comov_dist'],
            y=self.cosmo_dic['z_win_max'], ext='zeros')

    def interp_transverse_comoving_dist(self):
        """Interpolates the transverse comoving distance.

        Adds an interpolator for the transverse comoving distance to the
        dictionary so that it can be evaluated at redshifts not explicitly
        supplied to Cobaya.

        Updates 'key' in the :obj:`cosmo_dic` attribute of the class
        by adding an interpolator object which interpolates
        transverse comoving distance as a function of redshift.
        """
        transverse_comoving_dist = (self.cosmo_dic['angular_dist'] *
                                    (1.0 + self.cosmo_dic['z_win']))
        self.cosmo_dic['f_K_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'],
                y=transverse_comoving_dist, ext=2)

    def interp_transverse_comoving_dist_z12(self):
        """Interpolates the transverse comoving distance.

        Adds an interpolator for the transverse comoving distance from
        :math:`z_1` to :math:`z_2` to the dictionary so that it can be
        evaluated at redshifts not explicitly supplied from the Boltzmann
        solver.

        Updates 'key' in the cosmo_dic attribute of the current class
        by adding an interpolator object which interpolates the
        transverse comoving distance as a function of the specified
        redshifts.
        """
        x_int = self.cosmo_dic['z_win_max']

        if isinstance(self.cosmo_dic['comov_dist'], tuple):
            comov_dist = np.array(self.cosmo_dic['comov_dist'][0])
        elif isinstance(self.cosmo_dic['comov_dist'], np.ndarray):
            comov_dist = np.array(self.cosmo_dic['comov_dist'])

        int_z1z2 = ((comov_dist[None, :] - comov_dist[:, None]) *
                    self.cosmo_dic['H0'] / self.cosmo_dic['c'])
        if self.cosmo_dic['Omk'] == 0.0:
            y_int = int_z1z2
        elif self.cosmo_dic['Omk'] > 0.0:
            y_int = (np.sinh(np.sqrt(self.cosmo_dic['Omk']) * int_z1z2) /
                     np.sqrt(self.cosmo_dic['Omk']))
        else:
            y_int = (np.sin(np.sqrt(-self.cosmo_dic['Omk']) * int_z1z2) /
                     np.sqrt(-self.cosmo_dic['Omk']))
        y_int *= (self.cosmo_dic['c'] / self.cosmo_dic['H0'])

        self.cosmo_dic['_f_K_z12_func'] = \
            interpolate.RectBivariateSpline(x_int, x_int, y_int, kx=3, ky=3)

    def f_K_z12_wrapper(self, z1, z2):
        """Wrapper for the transverse comoving distance from z1 to z2.

        Does type checking, calls the method stored in
        :obj:`self.cosmo_dic['_f_K_z12_func']`, and returns the output variable
        according to the type of the input variables. The output distance
        is positive-defined, and the function is symmetric in :math:`z_1`
        and :math:`z_2`, except for the shape of the return value.

        Parameters
        ----------
        z1: float or int or numpy.ndarray
            Lower redshift :math:`z_1`
        z2: float or int or numpy.ndarray
            Upper redshift :math:`z_2`

        Returns
        -------
        Transverse comoving distance: float or numpy.ndarray
            Transverse comoving distance between :math:`z_1` and :math:`z_2`
        """
        if (isinstance(z1, (int, float)) and isinstance(z2,
                                                        (int, float))):
            f_K_z12 = self.cosmo_dic['_f_K_z12_func'](z1, z2)[0][0]
        elif (isinstance(z1, (int, float)) and isinstance(z2, np.ndarray)):
            f_K_z12 = self.cosmo_dic['_f_K_z12_func'](z1, z2)[0]
        elif (isinstance(z1, np.ndarray) and isinstance(z2, (int, float))):
            f_K_z12 = self.cosmo_dic['_f_K_z12_func'](z1, z2)[:, 0]
        else:
            f_K_z12 = self.cosmo_dic['_f_K_z12_func'](z1, z2)

        return abs(f_K_z12)

    def interp_angular_dist(self):
        """Interpolates the angular diameter distance.

        Adds an interpolator for angular distance to the dictionary so that
        it can be evaluated at redshifts not explicitly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class by adding an
        interpolator object of angular diameter distance
        as a function of redshift.
        """
        self.cosmo_dic['d_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['angular_dist'], ext=2)

    def interp_H(self):
        """Interpolates the Hubble parameter.

        Adds an interpolator for the Hubble parameter to the dictionary so that
        it can be evaluated at redshifts not explicitly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates the Hubble parameter
        :math:`H(z)` as a function of redshift.
        """
        self.cosmo_dic['H_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['H'], ext=2)

    def interp_H_Mpc(self):
        """Interpolates the Hubble parameter (Mpc).

        Adds an interpolator for the Hubble parameter in Mpc to the
        dictionary so that it can be evaluated at redshifts not
        explictly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates the Hubble parameter
        :math:`H(z)` in Mpc as a function of redshift.
        """
        self.cosmo_dic['H_z_func_Mpc'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['H_Mpc'], ext=2)

    def interp_sigma8(self):
        r"""Interpolates :math:`\sigma_8`.

        Adds an interpolator for the matter fluctuation
        parameter :math:`\sigma_8` to the dictionary so that it
        can be evaluated at redshifts not explicitly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates :math:`\sigma_8` as a function of redshift.
        """
        self.cosmo_dic['sigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['sigma8'], ext=2)

    def interp_fsigma8(self):
        r"""Interpolates :math:`f\sigma_8`.

        Adds an interpolator for :math:`f\sigma_8` to the dictionary
        so that it can be evaluated at redshifts
        not explicitly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates :math:`f\sigma_8` as a
        function of redshift.
        """
        self.cosmo_dic['fsigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['fsigma8'], ext=2)

    def interp_sigmaR(self):
        r"""Interp fsigma8

        Adds an interpolator for :math:`f\sigma_R` to the dictionary
        so that it can be evaluated at redshifts
        not explictly supplied to Cobaya

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates :math:`f\sigma_8` as a
        function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        if self.cosmo_dic['r_win'] is None:
            raise Exception('Boltzmann code radius binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['sigmaR_z_func'] = \
            interpolate.RectBivariateSpline(self.cosmo_dic['z_win'],
                                            self.cosmo_dic['r_win'],
                                            self.cosmo_dic['sigmaR'],
                                            kx=1, ky=1)

    def interp_sigmaR_cb(self):
        r"""Interp sigmaR_cb

        Adds an interpolator for :math:`sigma_R_cb` to the dictionary
        so that it can be evaluated at redshifts
        not explictly supplied to Cobaya

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates :math:`sigma_R_cb` as a
        function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        if self.cosmo_dic['r_win'] is None:
            raise Exception('Boltzmann code radius binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['sigmaR_z_func_cb'] = \
            interpolate.RectBivariateSpline(self.cosmo_dic['z_win'],
                                            self.cosmo_dic['r_win'],
                                            self.cosmo_dic['sigmaR_cb'],
                                            kx=1, ky=1)

    def create_phot_galbias(self, model=None, x_values=[0.0, 4.0],
                            y_values=[1.0, 1.0]):
        r"""Creates the photometric galaxy bias.

        Creates the photometric galaxy bias as
        function/interpolator of the redshift.
        The function is stored in the cosmo dictionary 'b1_inter'.

        The bias model is selected from the key 'bias_model'
        in :obj:`cosmo_dic`.
        The implemented models are:

            #. Linear interpolation
            #. Bias is constant in bin (returns one here)
            #. Polynomial bias function

        Parameters
        ----------
        model: integer
            selection of the bias model.
            If None, uses the one stored in cosmo_dic['bias_model']
        x_values: numpy.ndarray of float
            x-values for the interpolator.
        y_values: numpy.ndarray of float
            y-values for the interpolator.

        Raises
        ------
        ValueError
            If the bias model parameter in the cosmo dictionary
            is not 1, 2, or 3
        """

        if model is None:
            bias_model = self.cosmo_dic['bias_model']
        else:
            bias_model = model

        if bias_model == 1:
            self.cosmo_dic['b1_inter'] \
                = self.istf_phot_galbias_interpolator(
                    self.cosmo_dic['redshift_bins_means_phot'])
        elif bias_model == 2:
            self.cosmo_dic['b1_inter'] \
                = rb.linear_interpolator(x_values, y_values)
        elif bias_model == 3:
            self.cosmo_dic['b1_inter'] = self.poly_phot_galbias
        else:
            raise ValueError('Parameter bias_model not valid:'
                             f'{bias_model}')

    def istf_phot_galbias_interpolator(self, redshift_means):
        r"""IST:F Photometric galaxy bias interpolator.

        Returns a linear interpolator for the galaxy bias for the
        photometric GC probes at a given redshift.

        Parameters
        ----------
        redshift_means: numpy.ndarray of float
            Array of tomographic redshift bin means for GCphot

        Returns
        -------
        Interpolator: rb.linear_interpolator
            Linear interpolator of photometric galaxy bias
        """

        nuisance_par = self.cosmo_dic['nuisance_parameters']

        istf_bias_list = [nuisance_par[f'b1_photo_bin{idx}']
                          for idx, vl in
                          enumerate(redshift_means, start=1)]

        return rb.linear_interpolator(redshift_means, istf_bias_list)

    def poly_phot_galbias(self, redshift):
        r"""Polynomial photometric galaxy bias.

        Computes bias using a 3rd order polynomial function of redshift.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias

        Returns
        -------
        Photometric polynomial galaxy bias: float or numpy.ndarray
            Value(s) of photometric galaxy bias at input redshift(s)
        """
        nuisance = self.cosmo_dic['nuisance_parameters']
        return nuisance['b1_0_poly_photo'] + \
            nuisance['b1_1_poly_photo'] * redshift + \
            nuisance['b1_2_poly_photo'] * np.power(redshift, 2) + \
            nuisance['b1_3_poly_photo'] * np.power(redshift, 3)

    def compute_phot_galbias(self, redshift):
        r"""Computes the photometric galaxy bias.

        Computes galaxy bias(es) for GCphot
        at a given redshift.

        The bias model is implemented in the method
        `create_phot_galbias`, which must be called before,
        in order to use this function.

        Parameters
        ----------
        redshift: numpy.ndarray of float
            Redshift(s) at which to calculate bias

        Returns
        -------
        Photometric interpolated galaxy bias: numpy.ndarray of float
            Value(s) of photometric galaxy bias at input redshift(s)
        """

        return self.cosmo_dic['b1_inter'](redshift)

    def istf_spectro_galbias(self, redshift):
        """IST:F Spectroscopic galaxy bias interpolator.

        Gets galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift(s), according to the linear recipe
        used for version 1.0 of CLOE (default recipe).

        Attention: this will change in the future.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias

            Default is Euclid IST: Forecasting choices

        Returns
        -------
        Spectroscopic galaxy bias: float or numpy.ndarray
            Value(s) of spectroscopic galaxy bias at input redshift(s)

        Raises
        ------
        ValueError
            If redshift is outside of the bounds defined by the first
            and last element of bin_edges
        """
        bin_edges = self.cosmo_dic['redshift_bins_means_spectro']

        nuisance_src = self.cosmo_dic['nuisance_parameters']

        try:
            z_bin = rb.find_bin(redshift, bin_edges, False)
            bi_val = np.array([nuisance_src[f'b1_spectro_bin{i}']
                               for i in np.nditer(z_bin)])
            return bi_val[0] if np.isscalar(redshift) else bi_val
        except (ValueError, KeyError):
            raise ValueError('Spectroscopic galaxy bias cannot be obtained. '
                             'Check that redshift is inside the bin edges'
                             'and valid bi_spec\'s are provided.')

    def Pmm_phot_def(self, redshift, k_scale):
        r"""Matter power spectrum.

        Computes the matter-matter power spectrum for the photometric probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum

        Returns
        -------
        Photometric matter-matter power spectrum:  float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and k-mode for photometric probes
        """
        pval = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgg_phot_def(self, redshift, k_scale):
        r"""Galaxy power spectrum.

        Computes the galaxy-galaxy power spectrum for the photometric probe.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum

        Returns
        -------
        Photometric galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for GCphot
        """
        pval = ((self.compute_phot_galbias(redshift) ** 2) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgg_spectro_def(self, redshift, k_scale, mu_rsd):
        r"""Redshift-space galaxy power spectrum.

        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between the pair separation and
            the line of sight

        Returns
        -------
        Spectroscopic galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift, wavenumber and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.istf_spectro_galbias(redshift)
        growth = self.cosmo_dic['f_z'](redshift)
        power = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def Pgdelta_phot_def(self, redshift, k_scale):
        r"""Galaxy-matter cross-power spectrum.

        Computes the galaxy-matter power spectrum for the photometric probe.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_g^{\rm photo}(z)] P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Photometric galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """
        pval = (self.compute_phot_galbias(redshift) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgdelta_spectro_def(self, redshift, k_scale, mu_rsd):
        r"""Galaxy-matter cross-power spectrum in redshift space.

        Computes the redshift-space galaxy-matter power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm g \delta}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z)+f(z, k)\mu_{k}^2][1+f(z, k)\mu_{k}^2]\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between the pair separation
            and the line of sight

        Returns
        -------
        Spectroscopic galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift, wavenumber and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.istf_spectro_galbias(redshift)
        growth = self.cosmo_dic['f_z'](redshift)
        power = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        pval = ((bias + growth * mu_rsd ** 2.0) *
                (1.0 + growth * mu_rsd ** 2.0)) * power
        return pval

    def fia(self, redshift, k_scale=0.001):
        r"""Intrinsic alignment function.

        Computes the intrinsic alignment function. For v1.0
        we set :math:`\langle L \rangle(z) /L_{\star}(z)=1`.

        .. math::
            f_{\rm IA}(z) &= -\mathcal{A_{\rm IA}}\mathcal{C_{\rm IA}}\
            \frac{\Omega_{m,0}}{D(z)}\
            [(1 + z)/(1 + z_{\rm pivot})]^{\eta_{\rm IA}}\
            [\langle L \rangle(z) /L_{\star}(z)]^{\beta_{\rm IA}}\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to evaluate the intrinsic alignment
        k_scale: float or numpy.ndarray
            Wavenumber(s) at which to evaluate the intrinsic alignment

        Returns
        -------
        Intrinsic alignment function: float or numpy.ndarray
            Value(s) of intrinsic alignment function at
            given redshift(s) and wavenumber(s)
        """
        if self.cosmo_dic['use_gamma_MG']:
            # if gamma_MG parametrization is used
            # the k-dependency in the growth_factor
            # and growth_rate is dropped
            growth = self.cosmo_dic['D_z_k_func_MG'](redshift)
        else:
            growth = self.cosmo_dic['D_z_k_func'](redshift, k_scale)

        if (isinstance(redshift, (list, np.ndarray)) and
                isinstance(wavenumber, (list, np.ndarray))):
            redshift = np.repeat(redshift[:, np.newaxis], len(wavenumber), 1)

        c1 = 0.0134
        pivot_redshift = \
            self.cosmo_dic['nuisance_parameters']['pivot_redshift']
        a1_ia = self.cosmo_dic['nuisance_parameters']['a1_ia']
        eta1_ia = self.cosmo_dic['nuisance_parameters']['eta1_ia']
        beta1_ia = self.cosmo_dic['nuisance_parameters']['beta1_ia']
        omegam = self.cosmo_dic['Omm']
        fia = (-a1_ia * c1 * omegam / growth *
               ((1 + redshift) / (1 + pivot_redshift)) ** eta1_ia *
               self.cosmo_dic['luminosity_ratio_z_func'](redshift) ** beta1_ia)
        return fia

    def Pii_def(self, redshift, k_scale):
        r"""Intrinsic alignment power spectrum.

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Intrinsic alignment power spectrum: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and wavenumber
        """
        pval = self.fia(redshift)**2.0 * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pdeltai_def(self, redshift, k_scale):
        r"""Matter-intrinsic cross-power spectrum.

        Computes the matter-intrinsic power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Matter-intrinsic power spectrum: float or numpy.ndarray
            Value of matter-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.fia(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_phot_def(self, redshift, k_scale):
        r"""Galaxy-intrinsic cross-power spectrum.

        Computes the photometric galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Photometric galaxy-intrinsic power spectrum: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.fia(redshift) * self.compute_phot_galbias(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_spectro_def(self, redshift, k_scale):
        r"""Galaxy-intrinsic cross-power spectrum in redshift space.

        Computes the spectroscopic galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm spectro}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm spectro}(z)P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Spectroscopic galaxy-intrinsic power spectrum: float or numpy.ndarray
            Value of spectroscopic galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.fia(redshift) * self.istf_spectro_galbias(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def noise_Pgg_spectro(self, redshift, k_scale, mu_rsd):
        r"""Noise corrections to spectroscopic Pgg.

        This method is only used for the linear-only case (NL_flag_spectro=0),
        where we assume that the Poissonian contribution has already been
        subtracted and that the non-Poissonian contribution is negligible.
        Therefore this function returns 0 for every redshift, scale, and angle
        to the line of sight.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between wavenumber and the
            line of sight

        Returns
        -------
        Noise contribution for the spectroscopic galaxy power spectrum
        """
        return 0.0

    def Weyl_matter_ratio_def(self, redshift, k_scale, grid=True):
        r"""Weyl matter ratio

        Returns the ratio of the linear Weyl power spectrum to
        the linear matter power spectrum.

        .. math::
            \Gamma^2(z,k)\equiv P_{\Phi+\Psi}(z,k)/P_{mm}(z,k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.
        grid: bool, optional
            If True, use the grid setting in power spectra.
            If False, set grid to False in power spectra.

        Returns
        -------
        \Gamma^2: float or numpy.ndarray
            Value of the ratio of the linear Weyl power spectrum to
            the linear matter power spectrum at a given redshift and
            wavenumber.
        """

        k_min = min(self.cosmo_dic['k_win'])
        k_max = max(self.cosmo_dic['k_win'])

        k_scale_array = np.array(k_scale)
        k_scale_array = np.clip(k_scale_array, k_min, k_max)

        args = [redshift, k_scale_array]
        if not grid:
            args.append(False)

        Pk_weyl = self.cosmo_dic['Pk_weyl'].P(*args)
        Pk_delta = self.cosmo_dic['Pk_delta'].P(*args)

        Gamma2 = Pk_weyl / Pk_delta
        return Gamma2

    def obtain_power_spectra(self):
        """Adds photometric/spectroscopic power spectra to cosmo dictionary.

        Creates interpolators (functions of redshift and scale) for the
        photometric galaxy power spectra (galaxy-galaxy and galaxy-matter)
        and the IA-related power spectra, based on the recipe defined by
        the value of the nonlinear flag, and assigns them to the corresponding
        keys of the cosmo dictionary.

        Assigns the direct function (functions of redshift, scale and angle
        with the line of sight) for the spectroscopic galaxy power spectra
        (galaxy-galaxy and galaxy-matter) based on the recipe defined by the
        value of the nonlinear flag, to the corresponding keys of the cosmo
        dictionary.

        Note: start_jupyter_nb.shinterpolators for v1.0 span the range
        :math:`k=[0.001,100.0]`.
        """

        k_win = self.cosmo_dic['k_win']
        z_win = self.cosmo_dic['z_win']

        spe_bin_edges = np.array([0.90, 1.10, 1.30, 1.50, 1.80])
        z_win_spectro = rb.reduce(z_win, spe_bin_edges[0], spe_bin_edges[-1])

        pksrc_phot = self.pk_source_phot
        pksrc_spectro = self.pk_source_spectro
        pmm_phot = np.array([pksrc_phot.Pmm_phot_def(zz, k_win)
                             for zz in z_win])
        pgg_phot = np.array([pksrc_phot.Pgg_phot_def(zz, k_win)
                             for zz in z_win])
        pgdelta_phot = np.array([pksrc_phot.Pgdelta_phot_def(zz, k_win)
                                 for zz in z_win])
        pii = np.array([pksrc_phot.Pii_def(zz, k_win)
                        for zz in z_win])
        pdeltai = np.array([pksrc_phot.Pdeltai_def(zz, k_win)
                            for zz in z_win])
        pgi_phot = np.array([pksrc_phot.Pgi_phot_def(zz, k_win)
                             for zz in z_win])
        pgi_spectro = np.array([pksrc_phot.Pgi_spectro_def(zz, k_win)
                                for zz in z_win_spectro])

        self.cosmo_dic['Pgg_spectro'] = pksrc_spectro.Pgg_spectro_def
        self.cosmo_dic['Pgdelta_spectro'] = pksrc_spectro.Pgdelta_spectro_def

        self.cosmo_dic['Pmm_phot'] = \
            interpolate.RectBivariateSpline(z_win,
                                            k_win,
                                            pmm_phot,
                                            kx=1, ky=1)
        self.cosmo_dic['Pgg_phot'] = \
            interpolate.RectBivariateSpline(z_win,
                                            k_win,
                                            pgg_phot,
                                            kx=1, ky=1)
        self.cosmo_dic['Pgdelta_phot'] = \
            interpolate.RectBivariateSpline(z_win,
                                            k_win,
                                            pgdelta_phot,
                                            kx=1, ky=1)
        self.cosmo_dic['Pii'] = \
            interpolate.RectBivariateSpline(z_win,
                                            k_win,
                                            pii,
                                            kx=1, ky=1)
        self.cosmo_dic['Pdeltai'] = \
            interpolate.RectBivariateSpline(z_win,
                                            k_win,
                                            pdeltai,
                                            kx=1, ky=1)
        self.cosmo_dic['Pgi_phot'] = \
            interpolate.RectBivariateSpline(z_win,
                                            k_win,
                                            pgi_phot,
                                            kx=1, ky=1)
        self.cosmo_dic['Pgi_spectro'] = \
            interpolate.RectBivariateSpline(z_win_spectro,
                                            k_win,
                                            pgi_spectro,
                                            kx=1, ky=1)
        return

    def rescaled_linear_power_MG(self, redshift, wavenumber):
        r"""Rescaled linear power spectrum due to Modified Gravity

        The rescaling is carried out with the ratio of the growth factors
        squared, as

        .. math::
            P_{\rm lin}^{\rm MG}(k, z)=P_{\rm lin}(k, z)\left[\frac{D_{\rm MG}\
            (z ; \gamma)}{D(z)}\right]^2

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Rescaled power spectrum: float or numpy.ndarray
            Linear power spectrum rescaled by MG for the total matter density
        """
        ratio = (self.cosmo_dic['D_z_k_func_MG'](redshift) /
                 self.cosmo_dic['D_z_k_func'](redshift, 0.05))

        if (isinstance(redshift, (list, np.ndarray)) and
                isinstance(wavenumber, (list, np.ndarray))):
            ratio = np.repeat(ratio[:, np.newaxis], len(wavenumber), 1)

        return (self.cosmo_dic['Pk_delta_Boltzmann'].P(redshift, wavenumber) *
                ratio**2)

    def rescaled_linear_power_cb_MG(self, redshift, wavenumber):
        r"""Rescaled linear cb power spectrum due to Modified Gravity

        The rescaling is carried out with the ratio of the growth factors
        squared, as

        .. math::
            P_{\rm lin}^{\rm MG}(k, z)=P_{\rm lin}(k, z)\left[\frac{D_{\rm MG}\
            (z ; \gamma)}{D(z)}\right]^2

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Rescaled power spectrum: float or numpy.ndarray
            Linear power spectrum rescaled by MG for the cdm+b component
        """
        ratio = (self.cosmo_dic['D_z_k_func_MG'](redshift) /
                 self.cosmo_dic['D_z_k_func'](redshift, 0.05))

        if (isinstance(redshift, (list, np.ndarray)) and
                isinstance(wavenumber, (list, np.ndarray))):
            ratio = np.repeat(ratio[:, np.newaxis], len(wavenumber), 1)

        return (self.cosmo_dic['Pk_cb_Boltzmann'].P(redshift, wavenumber) *
                ratio**2)

    def rescaled_halomodel_power_MG(self, redshift, wavenumber):
        r"""Rescaled halomodel power spectrum due to Modified Gravity

        The rescaling is carried out with the ratio of the growth factors
        squared, as

        .. math::
            P_{\rm lin}^{\rm MG}(k, z)=P_{\rm lin}(k, z)\left[\frac{D_{\rm MG}\
            (z ; \gamma)}{D(z)}\right]^2

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Rescaled power spectrum: float or numpy.ndarray
            Halomodel power spectrum rescaled by MG for the total matter
            density
        """
        ratio = (self.cosmo_dic['D_z_k_func_MG'](redshift) /
                 self.cosmo_dic['D_z_k_func'](redshift, 0.05))

        if (isinstance(redshift, (list, np.ndarray)) and
                isinstance(wavenumber, (list, np.ndarray))):
            ratio = np.repeat(ratio[:, np.newaxis], len(wavenumber), 1)

        return (self.cosmo_dic['Pk_halomodel_recipe_Boltzmann'].P(redshift,
                                                                  wavenumber) *
                ratio**2)

    def MG_mu_def(self, redshift, k_scale, MG_mu):
        r"""Modified gravitational coupling to matter.

        Returns the function :math:`\mu(z, k)` according to the
        Modified Gravity (MG) parametrisation.

        .. math::
            \Psi(z,k) &= -4\pi G\
            \frac{\bar\rho_{\rm m}(z)\delta_{\rm m}(z, k)}{k^2(1+z)^2}\
            \mu(z,k)\\.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate :math:`\mu(z, k)`
        k_scale: float or list or numpy.ndarray
            Wavenumber at which to evaluate :math:`\mu(z, k)`
        MG_mu: float
            Value of constant (for v1.0) :math:`\mu(z, k)`
            function

        Returns
        -------
        Modified gravity parameter: float
            Value of the Modified Gravity :math:`\mu(z, k)` function
            at a given redshift and wavenumber
        """

        return MG_mu

    def MG_sigma_def(self, redshift, k_scale, MG_sigma):
        r"""Modified gravitational coupling to light.

        Returns the function :math:`\Sigma(z, k)` according to the
        Modified Gravity (MG) parametrisation.

        .. math::
            \Phi(z,k)+\Psi(z,k) &= -8\pi G\
            \frac{\bar\rho_{\rm m}(z)\delta_{\rm m}(z,k)}{k^2(1+z)^2}\
            \Sigma(z,k)\\.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate :math:`\Sigma(z, k)`
        k_scale: float
            Wavenumber at which to evaluate :math:`\Sigma(z, k)`
        MG_sigma: float
            Value of constant (for v1.0) :math:`\Sigma(z, k)`
            function

        Returns
        -------
        Modified gravity parameter: float
            Value of the Modified Gravity :math:`\Sigma(z, k)` function
            at a given redshift and wavenumber
        """

        return MG_sigma

    def update_cosmo_dic(self, zs, ks, MG_mu=1.0, MG_sigma=1.0):
        """Updates the cosmology dictionary.

        Updates the dictionary with other cosmological quantities.

        Parameters
        ----------
        zs: list
            list of redshift for the power spectrum
        ks: float
            value of k-scale at which the growth factor
            is evaluated
        MG_mu: float
            constant value of modified gravity mu function
        MG_sigma: float
            constant value of modified gravity sigma function
        """
        # Update dictionary with H(z),
        # r(z), fsigma8, sigma8, f(z), D_A(z)
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        elif self.cosmo_dic['z_win_max'] is None:
            raise Exception('Extended redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.interp_H()
        self.interp_H_Mpc()
        self.interp_comoving_dist()
        self.interp_z_of_r()
        self.interp_transverse_comoving_dist()
        self.interp_transverse_comoving_dist_z12()
        self.cosmo_dic['f_K_z12_func'] = self.f_K_z12_wrapper
        self.interp_fsigma8()
        self.interp_sigma8()
        self.interp_sigmaR()
        self.interp_sigmaR_cb()
        self.interp_growth_rate()
        self.assign_growth_factor()
        self.interp_angular_dist()
        # For the moment we use our own definition
        # of the growth factor
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['sigma8_0'] = \
            self.cosmo_dic['sigma8_z_func'](0)
        self.cosmo_dic['MG_mu'] = lambda x, y: self.MG_mu_def(x, y, MG_mu)
        self.cosmo_dic['MG_sigma'] = lambda x, y: self.MG_sigma_def(x, y,
                                                                    MG_sigma)
        # Create deepcopy objects of the power spectrum interpolators obtained
        # from the Boltzmann solver. If use_gamma_MG=True, then the methods
        # to evaluate the power spectra, i.e. .P, are substituted with the
        # Modified Gravity functions
        self.cosmo_dic['Pk_delta'] = \
            deepcopy(self.cosmo_dic['Pk_delta_Boltzmann'])
        self.cosmo_dic['Pk_cb'] = \
            deepcopy(self.cosmo_dic['Pk_cb_Boltzmann'])
        if self.cosmo_dic['NL_flag_phot_matter'] > 0:
            self.cosmo_dic['Pk_halomodel_recipe'] = \
                deepcopy(self.cosmo_dic['Pk_halomodel_recipe_Boltzmann'])
        if self.cosmo_dic['use_gamma_MG']:
            self.cosmo_dic['Pk_delta'].P = self.rescaled_linear_power_MG
            self.cosmo_dic['Pk_cb'].P = self.rescaled_linear_power_cb_MG
            if self.cosmo_dic['NL_flag_phot_matter'] > 0:
                self.cosmo_dic['Pk_halomodel_recipe'].P = \
                    self.rescaled_halomodel_power_MG
        # Update nonlinear module, by calling the update_dic method
        # of the nonlinear instance
        self.nonlinear.update_dic(self.cosmo_dic)
        # Update dictionary with photo bias function
        # if photo galaxy bias is linear. Otherwise it is done
        # in the update_dic method of the nonlinear instance
        if self.cosmo_dic['NL_flag_phot_bias'] == 0:
            self.create_phot_galbias()
        # Update dictionary with power spectra
        self.obtain_power_spectra()

        if self.cosmo_dic['use_Weyl']:
            self.cosmo_dic['Weyl_matter_ratio'] = \
                self.Weyl_matter_ratio_def

        self.cosmo_dic['noise_Pgg_spectro'] = \
            self.pk_source_spectro.noise_Pgg_spectro
