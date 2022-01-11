"""
module: miscellanous

Contains functions from cosmology.py that are required
in the nonlinear module (only temporary to mimic the
linear implementation made by IST:L)
"""

import numpy as np
import likelihood.auxiliary.redshift_bins as rb


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

    def fia(self, redshift, k_scale=0.001):
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
        k_scale: float or numpy.ndarray
            k-mode(s) at which to evaluate the intrinsic alignment.

        Returns
        -------
        fia: float or numpy.ndarray
            Value(s) of intrinsic alignment function at
            given redshift(s) and k-mode(s)
        """
        growth = self.theory['D_z_k_func'](redshift, k_scale)
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
        aia = self.theory['nuisance_parameters']['aia']
        nia = self.theory['nuisance_parameters']['nia']
        bia = self.theory['nuisance_parameters']['bia']
        omegam = self.theory['Omm']
        lum = 1.0
        fia = (-aia * c1 * omegam / growth *
               (1 + redshift) ** nia * lum ** bia)
        return fia

    def istf_spectro_galbias(self, redshift, bin_edges=None):
        """Istf Spectro Galbias

        Gets galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift(s), according to the linear recipe
        used for version 1.0 of CLOE (default recipe).

        Attention: this function is going to be removed from the
        non-linear module. In the future we are not going to employ
        the same values used by IST:F, rather we are aiming at having
        a proper non-linear galaxy bias model with multiple parameters
        at each considered redshift bin

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias.
        bin_edges: numpy.ndarray
            Array of redshift bin edges for spectroscopic GC probe.
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

        if bin_edges is None:
            bin_edges = np.array([0.90, 1.10, 1.30, 1.50, 1.80])

        nuisance_src = self.theory['nuisance_parameters']

        try:
            z_bin = rb.find_bin(redshift, bin_edges, False)
            bi_val = np.array([nuisance_src[f'b{i}_spectro']
                              for i in np.nditer(z_bin)])
            return bi_val[0] if np.isscalar(redshift) else bi_val
        except (ValueError, KeyError):
            raise ValueError('Spectroscopic galaxy bias cannot be obtained. '
                             'Check that redshift is inside the bin edges'
                             'and valid bi_spectro\'s are provided.')

    def istf_phot_galbias(self, redshift, bin_edges=None):
        r"""Istf Phot Galbias

        Gets galaxy bias(es) for the photometric GC probes by
        interpolation at a given redshift z

        Note: for redshifts above the final bin (z > 2.5), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.001), we use the bias of the first bin.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias.
        bin_edges: numpy.ndarray
            Array of tomographic redshift bin edges for photometric GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        bi_val: float or numpy.ndarray
            Value(s) of photometric galaxy bias at input redshift(s)
        """

        if bin_edges is None:
            bin_edges = np.array([0.001, 0.418, 0.560, 0.678, 0.789,
                                  0.900, 1.019, 1.155, 1.324, 1.576, 2.50])

        z_in_range = rb.coerce(redshift, bin_edges)
        return self.theory['b_inter'](z_in_range)
