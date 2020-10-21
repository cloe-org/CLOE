# -*- coding: utf-8 -*-
"""Cosmology

Class to store cosmological parameters and functions.
"""

import numpy as np
from scipy import interpolate
from astropy import constants as const
from scipy import interpolate


class CosmologyError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Cosmology:
    """
    Class for cosmological observables
    """

    def __init__(self):
        """
        Parameters
        ----------
        None

        Cosmology dictionary parameters
        -------------------------------
        H0: float
            Present-day Hubble constant (km s^{-1} Mpc^{-1})
        omch2: float
            Present-day Omega_CDM * (H0/100)**2
        ombh2: float
            Present-day Omega_baryon * (H0/100)**2
        omnuh2: float
            Present-day Omega_neutrinos * (H0/100)**2
        mnu: float
            Sum of massive neutrino species masses (eV)
        comov_dist: array-like
            Value of comoving distances at redshifts z_win
        angular_dist: array-like
            Value of angular diameter distances at redshifts z_win
        H: array-like
            Hubble function evaluated at redshifts z_win
        Pk_interpolator: function
            Interpolator function for power spectrum from Boltzmann code
        Pk_delta: function
            Interpolator function for delta from Boltzmann code
        fsigma8: array
            fsigma8 function evaluated at z
        sigma_8: array
            sigma8 functione valuated at z
        b_gal: float
            Galaxy bias
        c: float
            Speed-of-light in units of km s^{-1}
        r_z_func: function
            Interpolated function for comoving distance
        d_z_func: function
            Interpolated function for angular diameter distance
        sigma8_z_func: function
            Interpolated function for angular sigma8
        fsigma8_z_func: function
            Interpolated function for fsigma8
        H_z_func: function
            Interpolated function for Hubble parameter
        z_win: array-like
            Array of redshifts ar which H and comov_dist are evaluated at
        k_win: array-like
            Array of k values which will be used to evaluate gaalxy power
            spectra
        Pgg_phot: function
            Galaxy-galaxy power spectrum for GC-phot.
        Pgdelta_phot: function
            Galaxy-matter power spectrum for GC-phot.
        Pgg_spec: function
            Galaxy-galaxy power spectrum for GC-spec.
        Pgdelta_spec: function
            Galaxy-matter power spectrum for GC-spec.
        """
        # (GCH): initialize cosmo dictionary
        # (ACD): Added speed of light to dictionary.!!!Important:it's in units
        # of km/s to be dimensionally consistent with H0.!!!!
        # SJ: temporary modification to b_gal
        self.cosmo_dic = {'H0': 67.5,
                          'omch2': 0.122,
                          'ombh2': 0.022,
                          'omnuh2': 0.00028,
                          'mnu': 0.06,
                          'tau': 0.07,
                          'nnu': 3.046,
                          'ns': 0.9674,
                          'As': 2.1e-9,
                          'comov_dist': None,
                          'angular_dist': None,
                          'H': None,
                          'Pk_interpolator': None,
                          'Pk_delta': None,
                          'Pgg_phot': None,
                          'Pgdelta_phot': None,
                          'Pgg_spec': None,
                          'Pgdelta_spec': None,
                          'fsigma8': None,
                          'b_gal': 1.0,
                          # 'b_gal': None,
                          'sigma_8': None,
                          'c': const.c.to('km/s').value,
                          'z_win': None,
                          'k_win': None,
                          'r_z_func': None,
                          'd_z_func': None,
                          'H_z_func': None,
                          'sigma8_z_func': None,
                          'fsigma8_z_func': None}

    def growth_factor(self, zs, ks):
        """
        Calculates growth factor according to

        .. math::
                   D(z, k) &=\sqrt{P(z, k)/P(z=0, k)}\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        growth factor

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

    def growth_rate(self, zs, ks):
        """
        Calculates growth rate according to

        .. math::
                   f(z, k) &=-(1+z)/D dD(z, k)/dz\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        interpolator growth rate

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

    def interp_comoving_dist(self):
        """
        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator comoving distance as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['r_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['comov_dist'], ext=2)

    def interp_angular_dist(self):
        """
        Adds an interpolator for angular distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator angular diameter distance  as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['d_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['angular_dist'], ext=2)

    def interp_H(self):
        """
        Adds an interpolator for the Hubble parameter to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator H as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['H_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['H'], ext=2)

    def interp_sigma8(self):
        """
        Adds an interpolator for sigma8 to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator sigma8 as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['sigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['sigma_8'], ext=2)

    def interp_fsigma8(self):
        """
        Adds an interpolator for fsigma8 to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator fsigma8 as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['fsigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['fsigma8'], ext=2)

    def generic_istf_bin_bias_calc(self, bin_mean_redshift):
        """
        Calculates galaxy bias according to default recipe, for a given
        redshift bin.

        .. math::
                   b_{x,i} = \sqrt{1+\bar{z}_{x,i}}\\

        Parameters
        ----------
        bin_mean_redshift: float
            Mean redshift of particular redshift bin.

        Returns
        -------
        Value of bias in redshift bin with mean given by bin_mean_redshift.
        """
        bias = np.sqrt(1.0 + bin_mean_redshift)
        return bias

    def istf_phot_galbias(self, redshift, bin_edge_list=[0.001, 0.418, 0.560,
                                                         0.678, 0.789, 0.900,
                                                         1.019, 1.155, 1.324,
                                                         1.576, 2.50]):
        """
        Calculates galaxy bias for the photometric GC probe, at given redshift,
        according to default recipe.

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of tomographic redshift bin edges for photometic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        Value of galaxy bias at input redshift.
        """
        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = self.generic_istf_bin_bias_calc(
                        (bin_edge_list[i] + bin_edge_list[i + 1]) / 2.0)
        elif redshift >= bin_edge_list[-1]:
            bi_val = self.generic_istf_bin_bias_calc((bin_edge_list[-1] +
                                                      bin_edge_list[-2]) / 2.0)
        elif redshift < bin_edge_list[0]:
            bi_val = self.generic_istf_bin_bias_calc((bin_edge_list[0] +
                                                      bin_edge_list[1]) / 2.0)
        return bi_val

    def istf_spec_galbias(self, redshift, bin_edge_list=[0.90, 1.10, 1.30,
                                                         1.50, 1.80]):
        """
        Calculates galaxy bias for the spectroscopic GC probe, at given
        redshift, according to default recipe.

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of tomographic redshift bin edges for spectroscopic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        Value of galaxy bias at input redshift.
        """
        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = self.generic_istf_bin_bias_calc(
                        (bin_edge_list[i] + bin_edge_list[i + 1]) / 2.0)
        elif redshift >= bin_edge_list[-1]:
            bi_val = self.generic_istf_bin_bias_calc((bin_edge_list[-1] +
                                                      bin_edge_list[-2]) / 2.0)
        elif redshift < bin_edge_list[0]:
            bi_val = self.generic_istf_bin_bias_calc((bin_edge_list[0] +
                                                      bin_edge_list[1]) / 2.0)
        return bi_val

    def Pgg_phot_def(self, redshift, k_scale):
        """
        Calculates the galaxy-galaxy power spectrum for the photometric probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        Value of G-G power spectrum at given k and redshift.
        """
        pval = ((self.istf_phot_galbias(redshift) ** 2.0) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgg_spec_def(self, redshift, k_scale):
        """
        Calculates the galaxy-galaxy power spectrum for the spectroscopic
        probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        Value of G-G power spectrum at given k and redshift.
        """
        pval = ((self.istf_spec_galbias(redshift) ** 2.0) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgd_phot_def(self, redshift, k_scale):
        """
        Calculates the galaxy-matter power spectrum for the photometric probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        Value of G-delta power spectrum at given k and redshift.
        """
        pval = (self.istf_phot_galbias(redshift) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgd_spec_def(self, redshift, k_scale):
        """
        Calculates the galaxy-matter power spectrum for the spectroscopic
        probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        Value of G-delta power spectrum at given k and redshift.
        """
        pval = (self.istf_spec_galbias(redshift) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def interp_galaxy_spectra(self):

        ks_base = self.cosmo_dic['k_win']
        zs_base = self.cosmo_dic['z_win']

        zs_interp = np.tile(zs_base, len(ks_base))
        ks_interp = np.repeat(ks_base, len(zs_base))

        pgg_phot = []
        pgg_spec = []
        pgdelta_phot = []
        pgdelta_spec = []
        for index in range(len(ks_interp)):
            pgg_phot.append(self.Pgg_phot_def(zs_interp[index],
                                              ks_interp[index]))
            pgg_spec.append(self.Pgg_spec_def(zs_interp[index],
                                              ks_interp[index]))
            pgdelta_phot.append(self.Pgd_phot_def(zs_interp[index],
                                                  ks_interp[index]))
            pgdelta_spec.append(self.Pgd_spec_def(zs_interp[index],
                                                  ks_interp[index]))
        self.cosmo_dic['Pgg_phot'] = interpolate.interp2d(x=zs_interp,
                                                          y=ks_interp,
                                                          z=pgg_phot,
                                                          bounds_error=True)
        self.cosmo_dic['Pgdelta_phot'] = interpolate.interp2d(x=zs_interp,
                                                              y=ks_interp,
                                                              z=pgdelta_phot,
                                                              bounds_error=True
                                                              )
        self.cosmo_dic['Pgg_spec'] = interpolate.interp2d(x=zs_interp,
                                                          y=ks_interp,
                                                          z=pgg_spec,
                                                          bounds_error=True)
        self.cosmo_dic['Pgdelta_spec'] = interpolate.interp2d(x=zs_interp,
                                                              y=ks_interp,
                                                              z=pgdelta_spec,
                                                              bounds_error=True
                                                              )
        return

    def update_cosmo_dic(self, zs, ks):
        """
        Update the dictionary with other cosmological quantities

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        None

        """
        # (GCH): this function is superfluous
        # just in case we want to have always
        # an updated dictionary with D_z, f, H(z), r(z)
        self.interp_H()
        self.interp_comoving_dist()
        self.interp_fsigma8()
        self.interp_sigma8()
        self.interp_angular_dist()
        self.interp_galaxy_spectra()
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['f_z_k'] = self.growth_rate(zs, ks)
