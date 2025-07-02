"""
MISCELLANEOUS

This module contains functions from :obj:`cosmology.py` that are
required in the nonlinear module (only temporary to mimic the
linear implementation made by IST:L).
"""

import numpy as np
import cloe.auxiliary.redshift_bins as rb
import fastpt as fpt
from scipy import interpolate


class Misc:
    r"""
    Class for storing miscellanous routines.
    """

    def __init__(self, cosmo_dic):
        """Initialises the :obj:`Misc` class.

        Constructor of the class Misc.

        Parameters
        ----------
        cosmo_dic: dict
            External dictionary from nonlinear module
        """
        self.theory = cosmo_dic

    def update_dic(self, cosmo_dic):
        """
        Updates theory with an external cosmo dictionary.
        """
        self.theory = cosmo_dic

        # This parameter sets the FAST-PT quantities needed in initialization
        if self.theory['IA_flag'] == 1:
            to_do = ['IA']
            pad_factor = 1
            n_pad = int(pad_factor * len(self.theory['k_win']))
            self.f_pt = fpt.FASTPT(self.theory['k_win'],
                                   to_do=to_do,
                                   low_extrap=-5,
                                   high_extrap=3,
                                   n_pad=n_pad)

    def fia(self, redshift, wavenumber=0.001):
        r"""Intrinsic alignment function.

        Computes the intrinsic alignment function for the NLA model. For v1.0
        we set :math:`\langle L \rangle(z) /L_{\star}(z)=1`.

        .. math::
            f_{\rm IA}(z) &= -\mathcal{A_{\rm IA}}\mathcal{C_{\rm IA}}\
            \frac{\Omega_{\rm m,0}}{D(z)}
            [(1 + z)/(1 + z_{\rm pivot})]^{\eta_{\rm IA}}\
            [\langle L \rangle(z) /L_{\star}(z)]^{\beta_{\rm IA}}\\.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to evaluate the intrinsic alignment
        wavenumber: float or numpy.ndarray
            Wavenumber(s) at which to evaluate the intrinsic alignment
            Default value set to wavenumber=0.001.

        Returns
        -------
        Intrinsic alignment function: float or numpy.ndarray
            Value(s) of intrinsic alignment function at
            given redshift(s) and wavenumber(s)
        """
        if self.theory['use_gamma_MG']:
            # if gamma_MG parametrization is used
            # the k-dependency in the growth_factor
            # and growth_rate is dropped
            growth = self.theory['D_z_k_func_MG'](redshift)
        else:
            growth = self.theory['D_z_k_func'](redshift, wavenumber)

        if (isinstance(redshift, (list, np.ndarray)) and
                isinstance(wavenumber, (list, np.ndarray))):
            redshift = np.repeat(redshift[:, np.newaxis], len(wavenumber), 1)

        c1 = 0.0134
        pivot_redshift = self.theory['nuisance_parameters']['pivot_redshift']
        a1_ia = self.theory['nuisance_parameters']['a1_ia']
        eta1_ia = self.theory['nuisance_parameters']['eta1_ia']
        beta1_ia = self.theory['nuisance_parameters']['beta1_ia']
        omegam = self.theory['Omm']

        fia = (-a1_ia * c1 * omegam / growth *
               ((1 + redshift) / (1 + pivot_redshift)) ** eta1_ia *
               self.theory['luminosity_ratio_z_func'](redshift) ** beta1_ia)
        return fia

    def ia_tatt_terms(self, wavenumber=0.001, C_window=.75):
        r"""ia_tatt_terms

        Computes the terms of the IA TATT model, at 1-loop order.
        For reference on the equations, see arxiv:1708.09247.

        Parameters
        ----------
        wavenumber: float or numpy.ndarray
            wavemode(s) at which to evaluate the intrinsic alignment.
            Default value set to wavenumber=0.001.
        C_window: float
            It removes high frequency modes to avoid ringing effects.
            Default value set to C_window = 0.75 (tested with fast-pt).

        Returns
        -------
        a00e, c00e, a0e0e, a0b0b, ae2e2, ab2b2, a0e2, b0e2, d0ee2, d0bb2:
        numpy.ndarray
            Value(s) of the intrinsic alignment 1-loop order terms as a
            function of the wavemode(s) at a fixed redshift z = 0.
        """

        P_window = None

        a00e, c00e, a0e0e, a0b0b = self.f_pt.IA_ta(self.theory['Pk_delta'].
                                                   P(0, wavenumber),
                                                   P_window=P_window,
                                                   C_window=C_window)

        ae2e2, ab2b2 = self.f_pt.IA_tt(self.theory['Pk_delta'].P(
            0, wavenumber), P_window=P_window, C_window=C_window)

        a0e2, b0e2, d0ee2, d0bb2 = self.f_pt.IA_mix(self.theory['Pk_delta'].
                                                    P(0, wavenumber),
                                                    P_window=P_window,
                                                    C_window=C_window)

        a00e = interpolate.interp1d(wavenumber, a00e, kind='linear',
                                    fill_value='extrapolate')

        c00e = interpolate.interp1d(wavenumber, c00e, kind='linear',
                                    fill_value='extrapolate')

        a0e0e = interpolate.interp1d(wavenumber, a0e0e, kind='linear',
                                     fill_value='extrapolate')

        a0b0b = interpolate.interp1d(wavenumber, a0b0b, kind='linear',
                                     fill_value='extrapolate')

        ae2e2 = interpolate.interp1d(wavenumber, ae2e2, kind='linear',
                                     fill_value='extrapolate')

        ab2b2 = interpolate.interp1d(wavenumber, ab2b2, kind='linear',
                                     fill_value='extrapolate')

        a0e2 = interpolate.interp1d(wavenumber, a0e2, kind='linear',
                                    fill_value='extrapolate')

        b0e2 = interpolate.interp1d(wavenumber, b0e2, kind='linear',
                                    fill_value='extrapolate')

        d0ee2 = interpolate.interp1d(wavenumber, d0ee2, kind='linear',
                                     fill_value='extrapolate')

        d0bb2 = interpolate.interp1d(wavenumber, d0bb2, kind='linear',
                                     fill_value='extrapolate')

        return a00e, c00e, a0e0e, a0b0b, ae2e2, ab2b2, a0e2, b0e2, d0ee2, d0bb2

    def normalize_tatt_parameters(self, redshift, wavenumber=0.001):
        r"""normalize_tatt_parameters

        Computes the normalized TATT parameters C1, C1d and C2.
        (arxiv:1708.09247)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to evaluate the intrinsic alignment.
        wavenumber: float or numpy.ndarray
            wavemode(s) at which to evaluate the intrinsic alignment.
            Default value set to wavenumber=0.001.

        Returns
        -------
        C1, C1d, C2: float or numpy.ndarray
            Value(s) of the normalized TATT parameters C1, C1d and C2
        """

        omegam = self.theory['Omm']
        growth = self.theory['D_z_k_func'](redshift, wavenumber)
        a1_ia = self.theory['nuisance_parameters']['a1_ia']
        a2_ia = self.theory['nuisance_parameters']['a2_ia']
        b1_ia = self.theory['nuisance_parameters']['b1_ia']
        eta1_ia = self.theory['nuisance_parameters']['eta1_ia']
        eta2_ia = self.theory['nuisance_parameters']['eta2_ia']
        c1_bar = 0.0134
        pivot_redshift = self.theory['nuisance_parameters']['pivot_redshift']
        c1 = -a1_ia * c1_bar * omegam * \
            ((1 + redshift) / (1 + pivot_redshift)) ** eta1_ia / growth
        c2 = a2_ia * 5 * c1_bar * omegam * \
            ((1 + redshift) / (1 + pivot_redshift)) ** eta2_ia / (growth**2)
        c1d = b1_ia * c1

        return c1, c1d, c2

    def istf_spectro_galbias(self, redshift):
        """IST:F Spectroscopic galaxy bias.

        Gets the galaxy bias for the spectroscopic galaxy
        clustering probe, at given redshift(s), according to
        the linear recipe used for version 1.0 of CLOE
        (default recipe).

        Attention: this function is going to be removed from the
        nonlinear module. In the future we are not going to employ
        the same values used by IST:F, rather we are aiming at having
        a proper nonlinear galaxy bias model with multiple parameters
        at each considered redshift bin.

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
            and last element of ``bin_edges``
        """

        bin_edges = self.theory['redshift_bins_means_spectro']

        nuisance_src = self.theory['nuisance_parameters']

        try:
            redshift_bin = rb.find_bin(redshift, bin_edges, False)
            bi_val = np.array([nuisance_src[f'b1_spectro_bin{i}']
                              for i in np.nditer(redshift_bin)])
            return bi_val[0] if np.isscalar(redshift) else bi_val
        except (ValueError, KeyError):
            raise ValueError('Spectroscopic galaxy bias cannot be obtained. '
                             'Check that redshift is inside the bin edges'
                             'and valid bi_spectro\'s are provided.')
