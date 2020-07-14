"""
module: spec

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np
from ..cosmo.cosmology import Cosmology
from scipy.special import legendre
from scipy import integrate


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
        self.cosmo_dic = cosmo_dic
        self.fiducial = fiducial_dic

    def scaling_factor_perp(self, z):
        return self.theory['d_z_func'](z) / self.fiducial['d_z_func'](z)

    def scaling_factor_parall(self, z):
        return self.fiducial['H_z_func'](z) / self.theory['H_z_func'](z)

    def get_k(self, k_prime, mu_prime, z):
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - (mu_prime)**2))**(1. / 2)

    def get_mu(self, mu_prime, z):
        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - (mu_prime)**2))**(-1. / 2)

    # SJ: Only linear galaxy spectrum for now as a placeholder.
    # SJ: To be extended (i.e new function) by IST:NL.
    def pkgal_linear(self, mu, z, k):
        r"""
        Computation of P_gg appearing in Eqn 37 of IST:L document
        (found below). Here, we express it using  linear theory (Eqn 25).

        https://www.overleaf.com/read/pvfkzvvkymbj

        .. math::
            P_{\rm gg}\left(k(k',\mu_k'),\mu_k(\mu_k');z\right) = \
            \left(b(z) + f(z)({\mu_k}')^2\right)^2 P_{\rm mm}(k';z)

        """

        growth = self.theory['fsigma8'][0] / self.theory['sigma_8']
        pkgal = self.theory['Pk_interpolator'].P(z, k) * \
            (self.theory['b_gal'] + growth * mu**2.0)**2.0

        return pkgal

    def multipole_spectra_integrand(self, mu, z, k, m):
        r"""
        Computation of integrand of Eqn 37 of
        Euclid IST:L documentation (corresponding
        to multipole power spectra), found below.
        Note we consider ell = m in the code.

        https://www.overleaf.com/read/pvfkzvvkymbj

        .. math::
            L_\ell(\mu_k') \
            P_{\rm gg}\left(k(k',\mu_k'),\mu_k(\mu_k');z\right)

        """

        legendrepol = legendre(m)(mu)
        integrand = self.pkgal_linear(self.get_mu(mu, z), z,
                                      self.get_k(k, mu, z)) * legendrepol

        return integrand

    def multipole_spectra(self, z, k, m):
        r"""
        Computation of Eqn 37 of Euclid IST:L documentation
        (corresponding to multipole power spectra), found
        below. Note we consider ell = m in the code.

        https://www.overleaf.com/read/pvfkzvvkymbj

        .. math::
            P_{{\rm obs},\ell}(k';z)=\frac{1}{q_\perp^2 q_\parallel} \
            \frac{2\ell+1}{2}\int^1_{-1} L_\ell(\mu_k') \
            P_{\rm gg}\left(k(k',\mu_k'),\mu_k(\mu_k') \
            \PNT{;z}\right)\,{\rm d}\mu_k'

        """

        prefactor = 1.0 / self.scaling_factor_parall(z) / \
            (self.scaling_factor_perp(z))**2.0 * (2.0 * m + 1.0) / 2.0

        integral = prefactor * integrate.quad(self.multipole_spectra_integrand,
                                              a=-1.0, b=1.0, args=(z, k, m))[0]

        return integral

    def multipole_spectra_noap(self):
        r"""
        Computation of Eqns 28 - 30 of Euclid IST:L documentation
        (corresponding to multipole power spectra pre geometric distortions),
        found below.

        https://www.overleaf.com/read/pvfkzvvkymbj

        Eqns 28 - 30 shown in latex format below.

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
        # compute Eqns 28-30
        beta = self.theory['fsigma8'] / self.theory['sigma_8'] / \
            self.theory['b_gal']
        # beta = 1.0
        Pk_gal = (self.theory['b_gal']**2.0) * \
            self.theory['Pk_interpolator'].P(self.theory['zk'], 0.02)
        # Pk_gal = 1.0
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
        self.multipole_spectra_noap()
        return self.theory['Pk_interpolator'].P(0.5, 0.02)
