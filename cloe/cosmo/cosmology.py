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
        Pk_delta: function
            Interpolator function for linear matter :math:`P(k)` from
            Boltzmann code
        Pk_cb: function
            Interpolator function for cdm+b :math:`P(k)` from
            Boltzmann code
        Pk_halomodel_recipe: function
            Interpolator function for nonlinear matter :math:`P(k)` from
            Boltzmann code
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
        f_z: function
            Interpolated growth rate function
        H_z_func: function
            Interpolated function for Hubble parameter
        H_z_func_Mpc: function
            Interpolated function for Hubble parameter :math:`{\rm Mpc^{-1}}`
        D_z_k_func: function
            Interpolated function for growth factor
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
            and 3 IA parameters. The
            initialized values of the fiducial
            cosmology dictionary corresponds to:

                * Photo-z values corresponding to
                :math:`b_{(x,i)}=\sqrt{1+\bar{b}_{(x,i)}}`

                There are 3 bias options (linear, constant, polynomial)

                * Spectroscopic bias values in arXiv:1910.09273

                * IA values in arXiv:1910.09273

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
                          'k_win': None,
                          'comov_dist': None,
                          'angular_dist': None,
                          'H': None,
                          'H_Mpc': None,
                          'fsigma8': None,
                          'sigma8': None,
                          'D_z_k': None,
                          # Interpolators
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
                          'D_z_k_func': None,
                          'sigma8_z_func': None,
                          'fsigma8_z_func': None,
                          'f_z': None,
                          'luminosity_ratio_z_func': None,
                          # NL_boost
                          'NL_boost': None,
                          # NL flags
                          'NL_flag_phot_matter': 0,
                          'NL_flag_spectro': 0,
                          # bias model
                          # (1 => linear interpolation, 2 => constant in bins)
                          'bias_model': 1,
                          # magnification bias model
                          # (1 => linear interpolation, 2 => constant in bins)
                          'magbias_model': 2,
                          # Use Modified Gravity gamma
                          'use_gamma_MG': 0,
                          # Redshift dependent purity correction
                          'f_out_z_dep': False,
                          'nuisance_parameters': {
                             # Intrinsic alignment
                             'aia': 1.72,
                             'nia': -0.41,
                             'bia': 0.0,
                             'pivot_redshift': 0.,
                             # Photometric galaxy bias (IST:F case)
                             'b1_photo': 1.0997727037892875,
                             'b2_photo': 1.220245876862528,
                             'b3_photo': 1.2723993083933989,
                             'b4_photo': 1.316624471897739,
                             'b5_photo': 1.35812370570578,
                             'b6_photo': 1.3998214171814918,
                             'b7_photo': 1.4446452851824907,
                             'b8_photo': 1.4964959071110084,
                             'b9_photo': 1.5652475842498528,
                             'b10_photo': 1.7429859437184225,
                             # Photometric galaxy bias (polynomial case)
                             'b0_poly_photo': 0.830703,
                             'b1_poly_photo': 1.190547,
                             'b2_poly_photo': -0.928357,
                             'b3_poly_photo': 0.423292,
                             # Magnification bias
                             'magnification_bias_1': 0.0,
                             'magnification_bias_2': 0.0,
                             'magnification_bias_3': 0.0,
                             'magnification_bias_4': 0.0,
                             'magnification_bias_5': 0.0,
                             'magnification_bias_6': 0.0,
                             'magnification_bias_7': 0.0,
                             'magnification_bias_8': 0.0,
                             'magnification_bias_9': 0.0,
                             'magnification_bias_10': 0.0,
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
                             # Spectroscopic galaxy bias
                             'b1_spectro_bin1': 1.4614804,
                             'b1_spectro_bin2': 1.6060949,
                             'b1_spectro_bin3': 1.7464790,
                             'b1_spectro_bin4': 1.8988660,
                             'b2_spectro_bin1': 0.0,
                             'b2_spectro_bin2': 0.0,
                             'b2_spectro_bin3': 0.0,
                             'b2_spectro_bin4': 0.0,
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
                             # Shot noise parameters
                             'aP_spectro_bin1': 0.0,
                             'aP_spectro_bin2': 0.0,
                             'aP_spectro_bin3': 0.0,
                             'aP_spectro_bin4': 0.0,
                             'Psn_spectro_bin1': 0.0,
                             'Psn_spectro_bin2': 0.0,
                             'Psn_spectro_bin3': 0.0,
                             'Psn_spectro_bin4': 0.0,
                             # Purity of spectroscopic samples
                             'f_out': 0.0,
                             'f_out_1': 0.0,
                             'f_out_2': 0.0,
                             'f_out_3': 0.0,
                             'f_out_4': 0.0,
                             # Redshift distribution shifts
                             'dz_1_GCphot': 0.0, 'dz_1_WL': 0.0,
                             'dz_2_GCphot': 0.0, 'dz_2_WL': 0.0,
                             'dz_3_GCphot': 0.0, 'dz_3_WL': 0.0,
                             'dz_4_GCphot': 0.0, 'dz_4_WL': 0.0,
                             'dz_5_GCphot': 0.0, 'dz_5_WL': 0.0,
                             'dz_6_GCphot': 0.0, 'dz_6_WL': 0.0,
                             'dz_7_GCphot': 0.0, 'dz_7_WL': 0.0,
                             'dz_8_GCphot': 0.0, 'dz_8_WL': 0.0,
                             'dz_9_GCphot': 0.0, 'dz_9_WL': 0.0,
                             'dz_10_GCphot': 0.0, 'dz_10_WL': 0.0}
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
        # This method should be modified when other photometric flags will
        # be included (e.g. baryonic flag)
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
            P_z_k = self.cosmo_dic['Pk_delta'].P(zs, ks)
            D_z_k = np.sqrt(P_z_k / self.cosmo_dic['Pk_delta'].P(0.0, ks))
            return D_z_k
        except CosmologyError:
            w('Computation error in D(z, k)')

    # This function is deprecated
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

    def growth_rate_cobaya(self):
        r"""Growth rate from Cobaya.

        Calculates growth rate according to

        .. math::
                   f(z) &=f\sigma_8(z) / \sigma_8(z)\\

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates f(z).

        """
        fs8 = self.cosmo_dic['fsigma8_z_func'](self.cosmo_dic['z_win'])
        s8 = self.cosmo_dic['sigma8_z_func'](self.cosmo_dic['z_win'])
        growth = fs8 / s8
        self.cosmo_dic['f_z'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'],
                y=growth, ext=2)

    def growth_rate_MG(self, zs):
        r"""
        Computes the growth rate using :math:`\gamma_{\rm MG}` as

        .. math::
            f(z;\gamma_{\rm MG})=[\Omega_{\rm m}(z)]^{\gamma_{\rm MG}}

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates f(z).

        Parameters
        ----------
        zs: list
            List of redshift for the power spectrum

        """
        f_MG = self.matter_density(zs)**self.cosmo_dic['gamma_MG']
        self.cosmo_dic['f_z'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'],
                y=f_MG, ext=2)

    def _growth_integrand_MG(self, z_prime):
        r"""Integrand function for the :obj:`growth_factor_MG`.

        .. math::
              \frac{f(z';\gamma_{\rm MG})}{1+z'}

        Parameters
        ----------
        z_prime: float
           Integrand variable (redshift)
        """
        return self.cosmo_dic['f_z'](z_prime) / (1.0 + z_prime)

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

    def interp_growth_factor(self):
        """Interpolates the growth factor.

        Adds an interpolator for the growth factor (function of redshift and
        scale) to the cosmo dictionary.
        """
        z_win = self.cosmo_dic['z_win']
        k_win = self.cosmo_dic['k_win']

        if self.cosmo_dic['use_gamma_MG']:
            self.cosmo_dic['D_z_k_func'] = \
                interpolate.UnivariateSpline(z_win, self.growth_factor_MG())
        else:
            growth_grid = self.growth_factor(z_win, k_win)
            self.cosmo_dic['D_z_k_func'] = \
                interpolate.RectBivariateSpline(z_win,
                                                k_win,
                                                growth_grid,
                                                kx=3, ky=3)

    def interp_comoving_dist(self):
        """Interpolates the comoving distance.

        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explicitly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates comoving distance as a function of redshift.
        """

        self.cosmo_dic['r_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['comov_dist'], ext=2)

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
            y=self.cosmo_dic['z_win'], ext='zeros')

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
        x_int = self.cosmo_dic['z_win']

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

    def create_phot_galbias(self, model=None, x_values=[0.0, 4.0],
                            y_values=[1.0, 1.0]):
        r"""Creates the photometric galaxy bias.

        Creates the photometric galaxy bias as
        function/interpolator of the redshift.
        The function is stored in the cosmo dictionary 'b_inter'.

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
            self.cosmo_dic['b_inter'] \
                = self.istf_phot_galbias_interpolator(
                    self.cosmo_dic['redshift_bins_means_phot'])
        elif bias_model == 2:
            self.cosmo_dic['b_inter'] \
                = rb.linear_interpolator(x_values, y_values)
        elif bias_model == 3:
            self.cosmo_dic['b_inter'] = self.poly_phot_galbias
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

        istf_bias_list = [nuisance_par[f'b{idx}_photo']
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
        return nuisance['b0_poly_photo'] + \
            nuisance['b1_poly_photo'] * redshift + \
            nuisance['b2_poly_photo'] * np.power(redshift, 2) + \
            nuisance['b3_poly_photo'] * np.power(redshift, 3)

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

        return self.cosmo_dic['b_inter'](redshift)

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
            growth = self.cosmo_dic['D_z_k_func'](redshift)
        else:
            growth = self.cosmo_dic['D_z_k_func'](redshift, k_scale)
            z_is_array = isinstance(redshift, np.ndarray)
            k_is_array = isinstance(k_scale, np.ndarray)
            if k_is_array and not z_is_array:
                growth = growth[0]
            elif z_is_array and not k_is_array:
                growth = growth[:, 0]
            elif not z_is_array and not k_is_array:
                growth = growth[0, 0]
            else:
                redshift = redshift.reshape(-1, 1)

        c1 = 0.0134
        pivot_redshift = \
            self.cosmo_dic['nuisance_parameters']['pivot_redshift']
        aia = self.cosmo_dic['nuisance_parameters']['aia']
        nia = self.cosmo_dic['nuisance_parameters']['nia']
        bia = self.cosmo_dic['nuisance_parameters']['bia']
        omegam = self.cosmo_dic['Omm']
        fia = (-aia * c1 * omegam / growth *
               ((1 + redshift) / (1 + pivot_redshift)) ** nia *
               self.cosmo_dic['luminosity_ratio_z_func'](redshift) ** bia)
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
        self.interp_H()
        self.interp_H_Mpc()
        self.interp_comoving_dist()
        self.interp_z_of_r()
        self.interp_transverse_comoving_dist()
        self.interp_transverse_comoving_dist_z12()
        self.cosmo_dic['f_K_z12_func'] = self.f_K_z12_wrapper
        self.interp_fsigma8()
        self.interp_sigma8()
        if self.cosmo_dic['use_gamma_MG']:
            self.growth_rate_MG(zs)
        else:
            self.growth_rate_cobaya()
        self.interp_growth_factor()
        self.interp_angular_dist()
        # For the moment we use our own definition
        # of the growth factor
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['sigma8_0'] = \
            self.cosmo_dic['sigma8_z_func'](0)
        self.cosmo_dic['MG_mu'] = lambda x, y: self.MG_mu_def(x, y, MG_mu)
        self.cosmo_dic['MG_sigma'] = lambda x, y: self.MG_sigma_def(x, y,
                                                                    MG_sigma)
        # Update nonlinear module, by calling the update_dic method
        # of the nonlinear instance
        self.nonlinear.update_dic(self.cosmo_dic)
        # Update dictionary with bias function and power spectra
        self.create_phot_galbias()
        self.obtain_power_spectra()
