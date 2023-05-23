"""
module: miscellanous

Contains functions from cosmology.py that are required
in the nonlinear module (only temporary to mimic the
linear implementation made by IST:L)
"""

import numpy as np
import cloe.auxiliary.redshift_bins as rb


class Misc:
    r"""
    Class for storing miscellanous routines
    """

    def __init__(self, cosmo_dic):
        """Initialize

        Constructor of the class Misc

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from nonlinear module
        """
        self.theory = cosmo_dic

    def update_dic(self, cosmo_dic):
        """
        Update theory with an external cosmo dictionary
        """
        self.theory = cosmo_dic

    def fia(self, redshift, wavenumber=0.001):
        r"""Fia

        Computes the intrinsic alignment function. For v1.0
        we set :math:`\langle L \rangle(z) /L_{\star}(z)=1`.

        .. math::
            f_{\rm IA}(z) &= -\mathcal{A_{\rm IA}}\mathcal{C_{\rm IA}}\
            \frac{\Omega_{m,0}}{D(z)}(1 + z)^{\eta_{\rm IA}}\
            [\langle L \rangle(z) /L_{\star}(z)]^{\beta_{\rm IA}}\\

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to evaluate the intrinsic alignment.
        wavenumber: float or numpy.ndarray
            wavenumber(s) at which to evaluate the intrinsic alignment.

        Returns
        -------
        fia: float or numpy.ndarray
            Value(s) of intrinsic alignment function at
            given redshift(s) and wavenumber(s)
        """
        if self.theory['use_gamma_MG']:
            # if gamma_MG parametrization is used
            # the k-dependency in the growth_factor
            # and growth_rate is dropped
            growth = self.theory['D_z_k_func'](redshift)
        else:
            growth = self.theory['D_z_k_func'](redshift, wavenumber)
            redshift_is_array = isinstance(redshift, np.ndarray)
            wavenumber_is_array = isinstance(wavenumber, np.ndarray)
            if wavenumber_is_array and not redshift_is_array:
                growth = growth[0]
            elif redshift_is_array and not wavenumber_is_array:
                growth = growth[:, 0]
            elif not redshift_is_array and not wavenumber_is_array:
                growth = growth[0, 0]
            else:
                redshift = redshift.reshape(-1, 1)

        c1 = 0.0134
        aia = self.theory['nuisance_parameters']['aia']
        nia = self.theory['nuisance_parameters']['nia']
        bia = self.theory['nuisance_parameters']['bia']
        omegam = self.theory['Omm']
        fia = (-aia * c1 * omegam / growth *
               (1 + redshift) ** nia *
               self.theory['luminosity_ratio_z_func'](redshift) ** bia)
        return fia

    def istf_spectro_galbias(self, redshift):
        """Istf Spectro Galbias

        Gets galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift(s), according to the linear recipe
        used for version 1.0 of CLOE (default recipe).

        Attention: this function is going to be removed from the
        nonlinear module. In the future we are not going to employ
        the same values used by IST:F, rather we are aiming at having
        a proper nonlinear galaxy bias model with multiple parameters
        at each considered redshift bin

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias.

            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        bi_val: float or numpy.ndarray
            Value(s) of spectroscopic galaxy bias at input redshift(s)

        Raises
        ------
        ValueError
            If redshift is outside of the bounds defined by the first
            and last element of bin_edges
        """

        bin_edges = self.theory['redshift_bins']

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

    def istf_phot_galbias(self, redshift):
        r"""Istf Phot Galbias

        Gets galaxy bias(es) for the photometric GC probes by
        interpolation at a given redshift

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias.

        Returns
        -------
        bi_val: float or numpy.ndarray
            Value(s) of photometric galaxy bias at input redshift(s)
        """

        return self.theory['b_inter'](redshift)
