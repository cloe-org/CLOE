"""
module: spec

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np


class Spec:
    """
    Class for GC spectroscopy observable
    """

    def __init__(self, cosmo_dic, fiducial_dic):
        """
        Parameters
        ----------
        None
        """
        self.theory = cosmo_dic
        self.fiducial = fiducial_dic

    def scaling_factor_perp(self, z):
        return theory['d_z_func'](z) / fiducial['d_z_func'](z)

    def scaling_factor_parall(self, z):
        return theory['H_z_func'](z) / fiducial['H_z_func'](z)

    def get_k(self, k_prime, mu_prime, z):
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - (mu_prime)**2))**(1. / 2)

    def get_mu(self, mu_prime, z):
        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - (mu_prime)**2))**(-1. / 2)

    def multipole_spectra(self):
        r"""
        Computation of Eqns 25 - 27 of Euclid IST:L documentation
        (corresponding to multipole power spectra pre geometric distortions),
        found below.

        https://www.overleaf.com/read/pvfkzvvkymbj

        Eqns 25 - 27 shown in latex format below.

        .. math::
            P_0(k) &=\left(1+\frac{2}{3}\beta+\frac{1}{5}\beta^2\right)\,P(k)\\
            P_2(k) &=\left(\frac{4}{3}\beta+\frac{4}{7}\beta^2\right)\,P(k)\\
            P_4(k) &=\frac{8}{35}\beta^2\,P(k)

        """

        # SJ: Set up power spectrum [note weirdly Cobaya has it as P(z,k)
        # SJ: instead of the more common P(k,z)]
        # SJ: For now, fix redshift (and scale),
        # fix galaxy bias = 1, and fix sigma8(z) = 1
        # Cobaya does not seem to allow for either growth
        # rate alone or sigma8 alone
        # to be called yet (only their combination).
        # compute Eqns 25-27
        beta = self.theory['fsigma8'] / self.theory['sigma_8'] / \
            self.theory['b_gal']
        Pk_gal = (self.theory['b_gal']**2.0) * \
            self.theory['Pk_interpolator'].P(self.theory['zk'], 0.02)
        self.P0k = (1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta**2.0) * Pk_gal
        self.P2k = (4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2.0) * Pk_gal
        self.P4k = (8.0 / 35.0 * beta**2.0) * Pk_gal

        # (GCH): maybe save as attributes P0k, P2k, P4k?

    def loglike(self):
        """
        Returns
        -------
        loglike: float
                loglike for GC spectroscopy observable
                ln(likelihood) = -1/2 chi**2
        """
        # SJ: This will be the log-likelihood;
        # for now just return P(z,k) for fixed z and k.
        self.multipole_spectra()
        return self.theory['Pk_interpolator'].P(0.5, 0.02)
