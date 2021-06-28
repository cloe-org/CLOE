"""
module: miscellanous

Contains functions from cosmology.py that are required
in the nonlinear module (only temporary to mimic the
linear implementation made by IST:L)
"""

# Global
import numpy as np


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
            Redshift at which to evaluate the power spectrum.

        Returns
        -------
        fia: float or numpy.ndarray
            Value of intrinsic alignment function at
            a given redshift
        """
        growth = self.theory['D_z_k_func'](redshift, k_scale)
        ztype = isinstance(redshift, np.ndarray)
        ktype = isinstance(k_scale, np.ndarray)
        if (not(ztype) and not(ktype)):
            growth = growth[0, 0]
        elif (ztype ^ ktype):
            growth = growth[0]

        c1 = 0.0134
        aia = self.theory['nuisance_parameters']['aia']
        nia = self.theory['nuisance_parameters']['nia']
        bia = self.theory['nuisance_parameters']['bia']
        omegam = self.theory['Omm']
        lum = 1.0
        fia = (-aia * c1 * omegam / growth *
               (1 + redshift)**nia * lum**bia)
        return fia

    def istf_spec_galbias(self, redshift, bin_edge_list=[0.90, 1.10, 1.30,
                                                         1.50, 1.80]):
        """Istf Spec Galbias

        Gets galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift, according to the linear recipe
        used for version 1.0 of CLOE (default recipe).

        Attention: this function is going to be removed from the
        non-linear module. In the future we are not going to employ
        the same values used by IST:F, rather we are aiming at having
        a proper non-linear galaxy bias model with multiple parameters
        at each considered redshift bin

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

        Raises
        ------
        ValueError
            If redshift is outside of the bounds defined by the first
            and last element of bin_edge_list
        """

        istf_bias_list = [self.theory['nuisance_parameters']['b1_spec'],
                          self.theory['nuisance_parameters']['b2_spec'],
                          self.theory['nuisance_parameters']['b3_spec'],
                          self.theory['nuisance_parameters']['b4_spec']]

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
        elif redshift >= bin_edge_list[-1]:
            raise Exception('Spectroscopic galaxy bias cannot be obtained '
                            'as redshift is above the highest bin edge')
        elif redshift < bin_edge_list[0]:
            raise Exception('Spectroscopic galaxy bias cannot be obtained '
                            'as redshift is below the lowest bin edge.')
        return bi_val

    def istf_phot_galbias(self, redshift, bin_edge_list=[0.001, 0.418, 0.560,
                                                         0.678, 0.789, 0.900,
                                                         1.019, 1.155, 1.324,
                                                         1.576, 2.50]):
        r"""Istf Phot Galbias

        Gets galaxy bias for the photometric GC probes by
        interpolation at a given redshift z

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

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            z_in_range = redshift
        elif redshift >= bin_edge_list[-1]:
            z_in_range = bin_edge_list[-1]
        elif redshift < bin_edge_list[0]:
            z_in_range = bin_edge_list[0]

        bi_val = self.theory['b_inter'](z_in_range)

        return bi_val
