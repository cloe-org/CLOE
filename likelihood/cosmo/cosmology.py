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

        """
        # (GCH): initialize cosmo dictionary
        # (ACD): Added speed of light to dictionary.!!!Important:it's in units
        # of km/s to be dimensionally consistent with H0.!!!!
        # SJ: temporary modifications to zk, b_gal, and sigma_8
        self.cosmo_dic = {'H0': 67.5,
                          'omch2': 0.122,
                          'ombh2': 0.022,
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
                          'fsigma8': None,
                          'zk': 1.0,
                          # 'zk': None,
                          'b_gal': 1.0,
                          # 'b_gal': None,
                          'sigma_8': 1.0,
                          # 'sigma_8': None,
                          'c': const.c.to('km/s').value,
                          'z_win': None,
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
        growth rate

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        # I assume 1+z=1/a where a: scale factor
        D_z_k = self.growth_factor(zs, ks)
        # This will work when k is fixed, not an array
        try:
            f_z_k = -(1 + zs) * np.gradient(D_z_k) / D_z_k
            return f_z_k
        except CosmologyError:
            print('Computation error in f(z, k)')
            print('ATTENTION: Check k is a value, not a list')

    def interp_comoving_dist(self):
        """
        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Parameters
        ----------
        zs: array
            list of redshift comoving distance is evaluated at.
        Returns
        -------
        None

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

        Parameters
        ----------
        zs: array
            list of redshift comoving distance is evaluated at.
        Returns
        -------
        None

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

        Parameters
        ----------
        zs: array
            list of redshift comoving distance is evaluated at.
        Returns
        -------
        None

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

        Parameters
        ----------
        zs: array
            list of redshift comoving distance is evaluated at.
        Returns
        -------
        None

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['sigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['sigma_8'], ext=2)

    def interp_fsigma8(self):
        """
        Adds an interpolator for sigma8 to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Parameters
        ----------
        zs: array
            list of redshift comoving distance is evaluated at.
        Returns
        -------
        None

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['fsigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['fsigma8'], ext=2)

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
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['f_z_k'] = self.growth_rate(zs, ks)
