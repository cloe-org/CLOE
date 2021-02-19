# -*- coding: utf-8 -*-
"""Cosmology

Class to store cosmological parameters and functions.
"""

import numpy as np
from likelihood.non_linear.nonlinear import Nonlinear
from scipy import interpolate
from astropy import constants as const
import sys

class CosmologyError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class Cosmology:
    r"""
    Class for cosmological observables
    """

    def __init__(self):
        r"""
        List of cosmological parameters implemented

        Parameters
        ----------
        H0: float
            Present-day Hubble constant :math:`(km·s^{-1}·Mpc^{-1})`
        H0_Mpc: float
            Present-day Hubble constant :math:`(Mpc^{-1})`
        omch2: float
            Present-day CDM energy density
            Omega_CDM * (H0/100)**2
        ombh2: float
            Present-day baryon energy density
            Omega_baryon * (H0/100)**2
        omkh2: float
            Present-day curvature energy density
            Omega_k * (H0/100)**2
        Omc: float
            Present-day CDM energy density
            Omega_CDM
        Omb: float
            Present-day baryon energy density
            Omega_baryon
        Omk: float
            Present-day curvature energy density
            Omega_k
        As: float
            amplitude of the primordial power spectrum
        ns: float
            spectral tilt of the primordial
            power spectrum
        sigma_8_0: float
            sigma8 evaluated at z = 0
        w: float
           Dark energy equation of state
        wa: float
           Dark energy equation of state
        omnuh2: float
            Present-day massive neutrinos energy density
            Omega_neutrinos * (H0/100)**2
        Omnu: float
            Present-day massive neutrinos energy density
            Omega_neutrinos
        Omm: float
            Present-day total matter energy density
            Omega_m
            Assumes sum of baryons, CDM and neutrinos
        Omc: float
            Present-day cold dark matter energy density
            Omega_cdm
        Omk: float
            Present-day curvature energy density
            Omega_k
        mnu: float
            Sum of massive neutrino species masses (eV)
        comov_dist: list
            Value of comoving distances at redshifts z_win
        angular_dist: list
            Value of angular diameter distances at redshifts z_win
        H: list
            Hubble function evaluated at redshifts z_win
        H_Mpc: list
            Hubble function evaluated at redshifts z_win in units
            of :math:`Mpc^{-1}`
        Pk_interpolator: function
            Interpolator function for power spectrum from Boltzmann code
        Pk_delta: function
            Interpolator function for delta from Boltzmann code
        fsigma8: list
            fsigma8 function evaluated at z
        sigma_8: list
            sigma8 function evaluated at z
        c: float
            Speed-of-light in units of :math:`km·s^{-1}`
        r_z_func: function
            Interpolated function for comoving distance
        d_z_func: function
            Interpolated function for angular diameter distance
        sigma8_z_func: function
            Interpolated function for sigma8
        fsigma8_z_func: function
            Interpolated function for fsigma8
        f_z: function
            Interpolated growth rate function
        H_z_func: function
            Interpolated function for Hubble parameter
        H_z_func_Mpc: function
            Interpolated function for Hubble parameter in :math:`Mpc^{-1}`
        z_win: list
            Array of redshifts ar which H and comov_dist are evaluated at
        k_win: list
            Array of k values which will be used to evaluate galaxy power
            spectra
        Pgg_phot: function
            Galaxy-galaxy power spectrum for GC-phot
        Pgdelta_phot: function
            Galaxy-matter power spectrum for GC-phot
        Pgg_spec: function
            Galaxy-galaxy power spectrum for GC-spec
        Pgdelta_spec: function
            Galaxy-matter power spectrum for GC-spec
        Pii: function
            Intrinsic alignment (intrinsic-intrinsic) power spectrum
        Pdeltai: function
            Density-intrinsic cross-spectrum
        Pgi_phot: function
            Photometric galaxy-intrinsic cross-spectrum
        Pgi_spec: function
            Spectroscopic galaxy-intrinsic cross-spectrum
        MG_mu: function
            mu function from Modified Gravity parametrization
        MG_sigma: function
            sigma function from Modified Gravity parametrization
        NL_boost: float
            Non-linear boost factor
        nuisance_parameters: dict
            Contains all nuisance bias parameters
            and IA parameters which are sampled over.
            At the moment, we have implemented
            10 constant bias for photo-z
            recipe and 4 for spec recipe,
            and 3 IA parameters. The
            initialized values of the fiducial
            cosmology dictionary corresponds to:

                * Photo-z values corresponding to
                :math:`b_{(x,i)}=\sqrt{1+\bar{b}_{(x,i)}}`

                * Spectroscopic bias values in arXiv:1910.09273

                * IA values in arXiv:1910.09273
        """
        # (GCH): initialize cosmo dictionary
        # (ACD): Added speed of light to dictionary.!!!Important:it's in units
        # of km/s to be dimensionally consistent with H0.!!!!
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
                          'mnu': 0.06,
                          'tau': 0.07,
                          'nnu': 3.046,
                          'ns': 0.96,
                          'As': 2.1e-9,
                          'sigma_8_0': 0.816,
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
                          'sigma_8': None,
                          'D_z_k': None,
                          # Interpolators
                          'Pk_interpolator': None,
                          'Pk_delta': None,
                          'Pgg_phot': None,
                          'Pgdelta_phot': None,
                          'Pgg_spec': None,
                          'Pgdelta_spec': None,
                          'Pii': None,
                          'Pdeltai': None,
                          'Pgi_phot': None,
                          'Pgi_spec': None,
                          'r_z_func': None,
                          'd_z_func': None,
                          'H_z_func': None,
                          'H_z_func_Mpc': None,
                          'sigma8_z_func': None,
                          'fsigma8_z_func': None,
                          'f_z': None,
                          # NL_boost
                          'NL_boost': None,
                          'nuisance_parameters': {
                             'like_selection': 2,
                             'full_photo': True,
                             'NL_flag': 1,
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
                             'b1_spec': 1.4614804,
                             'b2_spec': 1.6060949,
                             'b3_spec': 1.7464790,
                             'b4_spec': 1.8988660,
                             'aia': 1.72,
                             'nia': -0.41,
                             'bia': 0.0}}

        self.cosmo_dic['H0_Mpc'] = (self.cosmo_dic['H0'] /
                                    const.c.to('km/s').value)
        self.nonlinear = Nonlinear(self.cosmo_dic)

    def growth_factor(self, zs, ks):
        r"""
        Computes growth factor according to

        .. math::
            D(z, k) &=\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts for the power spectrum
        ks: list
            list of modes for the power spectrum

        Returns
        -------
        D_z_k: numpy.ndarray
            Growth factor as function of redshift and k-mode

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        try:
            D_z_k = self.cosmo_dic['Pk_delta'].P(zs, ks)
            D_z_k = np.sqrt(D_z_k / self.cosmo_dic['Pk_delta'].P(0.0, ks))
            return D_z_k
        except CosmologyError:
            print('Computation error in D(z, k)')

    # ATTENTION !!!
    # THIS FUNCTION IS DEPRECATED
    def growth_rate(self, zs, ks):
        r"""Growth Rate

        Adds an interpolator for the growth rate (this function is actually
        deprecated since we use the growth rate directly from  Cobaya)

        .. math::
            f(z, k) &=-\frac{(1+z)}{D(z,k)}\frac{dD(z, k)}{dz}\\

        Parameters
        ----------
        zs: list
            list of redshift for the power spectrum
        ks: list
            list of modes for the power spectrum

        Returns
        -------
        f_z_k: object
            Interpolator growth rate as function of redshift and k-mode

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        # I assume 1+z=1/a where a: scale factor
        D_z_k = self.growth_factor(zs, ks)
        # This will work when k is fixed, not an array
        try:
            f_z_k = -(1 + zs) * np.gradient(D_z_k, zs[1] - zs[0]) / D_z_k
            return interpolate.InterpolatedUnivariateSpline(
                x=zs, y=f_z_k, ext=2)
        except CosmologyError:
            print('Computation error in f(z, k)')
            print('ATTENTION: Check k is a value, not a list')

    def growth_rate_cobaya(self):
        r"""Growth Rate Cobaya

        Calculates growth rate according to

        .. math::
                   f(z) &=f\sigma_8(z) / \sigma_8(z)\\

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates f(z)
        """
        fs8 = self.cosmo_dic['fsigma8_z_func'](self.cosmo_dic['z_win'])
        s8 = self.cosmo_dic['sigma8_z_func'](self.cosmo_dic['z_win'])
        growth = fs8 / s8
        self.cosmo_dic['f_z'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'],
                y=growth, ext=2)

    def interp_comoving_dist(self):
        """Interp Comoving Dist

        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates comoving distance as a function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['r_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['comov_dist'], ext=2)

    def interp_angular_dist(self):
        """Interp Angular Dist

        Adds an interpolator for angular distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Updates 'key' in the cosmo_dic attribute of the class by adding an
        interpolator object of angular diameter distance
        as a function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['d_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['angular_dist'], ext=2)

    def interp_H(self):
        """Interp H

        Adds an interpolator for the Hubble parameter to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates the Hubble parameter
        H(z) as a function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['H_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['H'], ext=2)

    def interp_H_Mpc(self):
        """Interp H Mpc

        Adds an interpolator for the Hubble parameter in Mpc to the
        dictionary so that it can be evaluated at redshifts not
        explictly supplied to Cobaya.

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates the Hubble parameter
        H(z) in Mpc as a function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['H_z_func_Mpc'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['H_Mpc'], ext=2)

    def interp_sigma8(self):
        r"""Interp Sigma8

        Adds an interpolator for the matter fluctuation
        parameter :math:`\sigma_8` to the dictionary so that it
        can be evaluated at redshifts not explictly supplied to Cobaya

        Updates 'key' in the cosmo_dic attribute of the class
        by adding an interpolator object
        which interpolates :math:`\sigma_8` as a function of redshift
        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['sigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['sigma_8'], ext=2)

    def interp_fsigma8(self):
        r"""Interp fsigma8

        Adds an interpolator for :math:`f\sigma_8` to the dictionary
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
        self.cosmo_dic['fsigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['fsigma8'], ext=2)

    def istf_phot_galbias(self, redshift, bin_edge_list=[0.001, 0.418, 0.560,
                                                         0.678, 0.789, 0.900,
                                                         1.019, 1.155, 1.324,
                                                         1.576, 2.50]):
        r"""Istf Phot Galbias

        Updates galaxy bias for the photometric GC probes at a given
        redshift z

        Note: for redshifts above the final bin (z > 2.5), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.001), we use the bias of the first bin.

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of tomographic redshift bin edges for photometric GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        bi_val: float
            Value of photometric galaxy bias at input redshift
        """
        istf_bias_list = [self.cosmo_dic['nuisance_parameters']['b1_photo'],
                          self.cosmo_dic['nuisance_parameters']['b2_photo'],
                          self.cosmo_dic['nuisance_parameters']['b3_photo'],
                          self.cosmo_dic['nuisance_parameters']['b4_photo'],
                          self.cosmo_dic['nuisance_parameters']['b5_photo'],
                          self.cosmo_dic['nuisance_parameters']['b6_photo'],
                          self.cosmo_dic['nuisance_parameters']['b7_photo'],
                          self.cosmo_dic['nuisance_parameters']['b8_photo'],
                          self.cosmo_dic['nuisance_parameters']['b9_photo'],
                          self.cosmo_dic['nuisance_parameters']['b10_photo']]

        print("spec galbias redshift", redshift)

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
        elif redshift >= bin_edge_list[-1]:
            bi_val = istf_bias_list[-1]
        elif redshift < bin_edge_list[0]:
            bi_val = istf_bias_list[0]
        return bi_val

    def istf_spec_galbias(self, redshift, bin_edge_list=[0.90, 1.10, 1.30,
                                                         1.50, 1.80]):
        """Istf Spec Galbias

        Updates galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift, according to default recipe.

        Note: for redshifts above the final bin (z > 1.80), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.90), we use the bias of the first bin.

        Attention: this will change in the future

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of redshift bin edges for spectroscopic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        bi_val: float
            Value of spectroscopic galaxy bias at input redshift
        """

        istf_bias_list = [self.cosmo_dic['nuisance_parameters']['b1_spec'],
                          self.cosmo_dic['nuisance_parameters']['b2_spec'],
                          self.cosmo_dic['nuisance_parameters']['b3_spec'],
                          self.cosmo_dic['nuisance_parameters']['b4_spec']]

        if bin_edge_list[0] <= redshift <= bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
                elif redshift == bin_edge_list[-1]:
                    bi_val = istf_bias_list[-1]
        elif redshift > bin_edge_list[-1]:
            # (SJ): let us throw an exception instead
            # bi_val = istf_bias_list[-1]
            raise Exception('Spectroscopic galaxy bias cannot be obtained '
                            'as redshift is above the highest bin edge')
        elif redshift < bin_edge_list[0]:
            # bi_val = istf_bias_list[0]
            raise Exception('Spectroscopic galaxy bias cannot be obtained '
                            'as redshift is below the lowest bin edge.')
        return bi_val

    def Pgg_phot_def(self, redshift, k_scale):
        r"""Pgg Phot Def

        Computes the galaxy-galaxy power spectrum for the photometric probe.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and k-mode for galaxy
            clustering photometric
        """
        pval = ((self.istf_phot_galbias(redshift) ** 2.0) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgg_spec_def(self, redshift, k_scale, mu_rsd):
        r"""Pgg Spec Def

        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spec}(z, k) &=\
            [b_{\rm g}^{\rm spec}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float or numpy.ndarray
            cosine of the angle between the pair separation and
            the line of sight

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.istf_spec_galbias(redshift)
        growth = self.cosmo_dic['f_z'](redshift)
        power = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def Pgd_phot_def(self, redshift, k_scale):
        r"""Pgd Phot Def

        Computes the galaxy-matter power spectrum for the photometric probe.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_g^{\rm photo}(z)] P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and k-mode for galaxy clustering
            photometric
        """
        pval = (self.istf_phot_galbias(redshift) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgd_spec_def(self, redshift, k_scale, mu_rsd):
        r"""Pgd Spec Def

        Computes the redshift-space galaxy-matter power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm g \delta}^{\rm spec}(z, k) &=\
            [b_{\rm g}^{\rm spec}(z) + f(z, k)\mu_{k}^2][1 + f(z, k)\mu_{k}^2]\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float or numpy.ndarray
            cosine of the angle between the pair separation
            and the line of sight

        Returns
        -------
        pval: float or numpy.ndarray.
            Value of galaxy-matter power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.istf_spec_galbias(redshift)
        growth = self.cosmo_dic['f_z'](redshift)
        power = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        pval = ((bias + growth * mu_rsd ** 2.0) *
                (1.0 + growth * mu_rsd ** 2.0)) * power
        return pval

    def fia(self, redshift, k_scale=0.001):
        r"""Fia

        Computes the intrinsic alignment function. For v1.0
        we set :math:`\beta_{\rm IA}=0`.

        .. math::
            f_{\rm IA}(z) &= -\mathcal{A_{\rm IA}}\mathcal{C_{\rm IA}}\
            \frac{\Omega_{m,0}}{D(z)}(1 + z)^{\eta_{\rm IA}}\
            [\langle L \rangle(z) /L_{\star}(z)]^{\beta_{\rm IA}}\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.

        Returns
        -------
        fia: float or numpy.ndarray
            Value of intrinsic alignment function at
            a given redshift
        """
        c1 = 0.0134
        aia = self.cosmo_dic['nuisance_parameters']['aia']
        nia = self.cosmo_dic['nuisance_parameters']['nia']
        bia = self.cosmo_dic['nuisance_parameters']['bia']
        omegam = self.cosmo_dic['Omm']
        # SJ: temporary lum for now, to be read in from IST:forecast file
        lum = 1.0
        fia = (-aia * c1 * omegam / self.growth_factor(redshift, k_scale) *
               (1 + redshift)**nia * lum**bia)
        return fia

    def Pii_def(self, redshift, k_scale):
        r"""Pii Def

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and k-mode
        """
        pval = self.fia(redshift)**2.0 * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pdeltai_def(self, redshift, k_scale):
        r"""Pdeltai Def

        Computes the density-intrinsic power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and k-mode
        """
        pval = self.fia(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_phot_def(self, redshift, k_scale):
        r"""Pgi Phot Def

        Computes the photometric galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and k-mode
        """
        pval = self.fia(redshift) * self.istf_phot_galbias(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_spec_def(self, redshift, k_scale):
        r"""Pgi Spec Def

        Computes the spectroscopic galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm spec}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm spec}(z)P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of spectroscopic galaxy-intrinsic power spectrum
            at a given redshift and k-mode
        """
        pval = self.fia(redshift) * self.istf_spec_galbias(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def interp_phot_galaxy_spectra(self):
        """Interp Phot Galaxy Spectra

        Creates interpolators for the photometric galaxy
        clustering and galaxy-matter power spectra, and adds them to cosmo_dic.
        Note: the interpolators for v1.0 span the range :math:`k=[0.001,100.0]`

        Updates 'keys' of the cosmo_dic attribute of the class, adding
        interpolator objects (interpolates photometric galaxy
        clustering and galaxy-matter power spectra as a
        function of redshift and k-mode)
        """
        # AP: Removed the interpolation of the spectroscopic galaxy power
        # spectra and renamed this method to reflect that the interpolation
        # is carrioed out only on the photometric spectra. This is because,
        # in order to interface the spec module with Pgg_spec we need it
        # in redshift-space , i.e. Pgg_spec(z, k, rsd_mu), and a 3-d
        # interpolation drastically increases the evaluation time.
        # Pgg_spec and Pgd_spec are added to the cosmo dic in the method
        # update_cosmo_dic
        ks_base = self.cosmo_dic['k_win']
        zs_base = self.cosmo_dic['z_win']

        pgg_phot = np.array([self.Pgg_phot_def(zz, ks_base) for zz in zs_base])
        pgdelta_phot = np.array([self.Pgd_phot_def(zz, ks_base)
                                 for zz in zs_base])
        pii = np.array([self.Pii_def(zz, ks_base) for zz in zs_base])
        pdeltai = np.array([self.Pdeltai_def(zz, ks_base) for zz in zs_base])
        pgi_phot = np.array([self.Pgi_phot_def(zz, ks_base) for zz in zs_base])
        # pgi_spec = np.array([self.Pgi_spec_def(zz, ks_base) for zz in zs_base])

        self.cosmo_dic['Pgg_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgg_phot,
                                            kx=1, ky=1)
        self.cosmo_dic['Pgdelta_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgdelta_phot,
                                            kx=1, ky=1)
        self.cosmo_dic['Pii'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pii,
                                            kx=1, ky=1)
        self.cosmo_dic['Pdeltai'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pdeltai,
                                            kx=1, ky=1)
        self.cosmo_dic['Pgi_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgi_phot,
                                            kx=1, ky=1)
        # self.cosmo_dic['Pgi_spec'] = \
        #     interpolate.RectBivariateSpline(zs_base,
        #                                     ks_base,
        #                                     pgi_spec,
        #                                     kx=1, ky=1)
        return

    def MG_mu_def(self, redshift, k_scale, MG_mu):
        r"""MG Mu Def

        Returns the function :math:`\mu(z, k)` according to the
        Modified Gravity (MG) parametrization

        .. math::
            \Psi(z,k) &= -4\pi G\
            \frac{\bar\rho_{\rm m}(z)\delta_{\rm m}(z, k)}{k^2(1+z)^2}\
            \mu(z,k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate :math:`\mu(z, k)`.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate :math:`\mu(z, k)`.
        MG_mu: float
            Value of constant (for v1.0) :math:`\mu(z, k)`
            function.

        Returns
        -------
        MG_mu: float
            Value of the Modified Gravity :math:`\mu(z, k)` function
            at a given redshift and k-mode
        """

        return MG_mu

    def MG_sigma_def(self, redshift, k_scale, MG_sigma):
        r"""MG Sigma Def

        Returns the function :math:`\Sigma(z, k)` according to the
        Modified Gravity (MG) parametrization

        .. math::
            \Phi(z,k)+\Psi(z,k) &= -8\pi G\
            \frac{\bar\rho_{\rm m}(z)\delta_{\rm m}(z,k)}{k^2(1+z)^2}\
            \Sigma(z,k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate :math:`\Sigma(z, k)`.
        k_scale: float
            k-mode at which to evaluate :math:`\Sigma(z, k)`.
        MG_sigma: float
            Value of constant (for v1.0) :math:`\Sigma(z, k)`
            function

        Returns
        -------
        MG_sigma: float
            Value of the Modified Gravity :math:`\Sigma(z, k)` function
            at a given redshift and k-mode
        """

        return MG_sigma

    def update_cosmo_dic(self, zs, ks, MG_mu=1.0, MG_sigma=1.0):
        """Update Cosmo Dic

        Update the dictionary with other cosmological quantities

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
        NL_flag: string
            flag for non-linear boost factor
        """
        # (GCH): update dictionary with H(z)
        # r(z), fsigma8, sigma8, f(z), D_A(z),
        # photo-spectra
        self.interp_H()
        self.interp_H_Mpc()
        self.interp_comoving_dist()
        self.interp_fsigma8()
        self.interp_sigma8()
        self.growth_rate_cobaya()
        self.interp_angular_dist()
        self.interp_phot_galaxy_spectra()
        # (GCH): As mentioned in interp_phot_galaxy_spectra
        # spec spectra are not interpolated so they are added
        # here in this method
        self.cosmo_dic['Pgg_spec'] = self.Pgg_spec_def
        self.cosmo_dic['Pgi_spec'] = self.Pgi_spec_def
        self.cosmo_dic['Pgdelta_spec'] = self.Pgd_spec_def
        # (GCH): for the moment we use our own definition
        # of the growth factor
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['sigma_8_0'] = \
            self.cosmo_dic['sigma8_z_func'](0)
        self.cosmo_dic['MG_mu'] = lambda x, y: self.MG_mu_def(x, y, MG_mu)
        self.cosmo_dic['MG_sigma'] = lambda x, y: self.MG_sigma_def(x, y,
                                                                    MG_sigma)
        self.cosmo_dic = self.nonlinear.update_dic(self.cosmo_dic)
