"""
module: spec

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np
from ..cosmo.cosmology import Cosmology
from scipy.special import legendre
from scipy import integrate
from scipy import interpolate
import os.path


class Spec:
    r"""
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
        r"""
        Computation of the perpendicular scaling factor

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.

        Returns
        -------
        scaling_factor_perp: float
           Value of the scaling_factor_perp at given redshift

        Notes
        -----
        .. math::
            q_{\perp} &= \frac{D_{\rm M}(z)}{D'_{\rm M}(z)}\\
        """

        return self.theory['d_z_func'](z) / self.fiducial['d_z_func'](z)

    def scaling_factor_parall(self, z):
        r"""
        Computation of the parallel scaling factor

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.

        Returns
        -------
        scaling_factor_perp: float
           Value of the scaling_factor_perp at given redshift

        Notes
        -----
        .. math::
            q_{\parallel} &= \frac{H'(z)}{H(z)}\\
        """
        return self.fiducial['H_z_func'](z) / self.theory['H_z_func'](z)

    def get_k(self, k_prime, mu_prime, z):
        r"""
        Computation of the wavenumber k

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        k_prime: float
           Fiducial Scale (wavenumber) at which to evaluate the galaxy power
        mu_prime: float
           Fiducial Cosine of the angle between the wavenumber and
           LOS (AP-distorted).

        Returns
        -------
        get_k: float
           Value of the scalar wavenumber  at given redshift
           cosine of the angle and fiducial wavenumber

        Notes
        -----
        .. math::
            k(k',\mu_k', ) &= k' \left[q_{\parallel}^{-2} \,
            (\mu_k')^2 + \
            q_{\perp}^{-2} \left( 1 - (\mu_k')^2 \right)\right]^{1/2}\\
        """
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - (mu_prime)**2))**(1. / 2)

    def get_mu(self, mu_prime, z):
        r"""
        Computation of the cosine of the angle

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        mu_prime: float
           Fiducial Cosine of the angle between the wavenumber
           and LOS (AP-distorted).

        Returns
        -------
        get_mu: float
           Value of the cosine of the angle at given redshift
           and fiducial wavenumber

        Notes
        -----
        .. math::
            \mu_k(\mu_k') &= \mu_k' \, q_{\parallel}^{-1}
            \left[ q_{\parallel}^{-2}\,
            (\mu_k')^2 + q_{\perp}^{-2}
            \left( 1 - (\mu_k')^2 \right) \right]^{-1/2}\\
        """

        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - (mu_prime)**2))**(-1. / 2)

    def multipole_spectra_integrand(self, rsd_mu, z, k, m):
        r"""
        Computation of multipole power spectrum integrand
        Note we consider ell = m in the code


        Parameters
        ----------
        rsd_mu: float
           Cosine of the angle between the wavenumber and LOS (AP-distorted).
        z: float
            Redshift at which to evaluate power spectrum.
        k: float
            Scale (wavenumber) at which to evaluate power spectrum.
        m: float
            Order of the Legendre expansion.


        Returns
        -------
        integrand: float
        Integrand of multipole power spectrum

        Notes
        -----
        .. math::
            L_\ell(\mu_k')P_{\rm gg}\left(k(k',\mu_k'),\mu_k(\mu_k');z\right)\\
        """

        legendrepol = legendre(m)(rsd_mu)
        integrand = (self.theory['Pgg_spec'](z, self.get_k(k, rsd_mu, z),
                                             self.get_mu(rsd_mu, z)) *
                     legendrepol)

        return integrand

    def multipole_spectra(self, z, k, m):
        r"""
        Computation of multipole power spectra
        Note we consider ell = m in the code.


        Parameters
        ----------
        z: float
        Redshift at which to evaluate power spectrum.
        k: float
        Scale (wavenumber) at which to evaluate power spectrum.
        m: float
        Order of the Legendre expansion.


        Returns
        -------
        integral: float
        Multipole power spectrum

        Notes
        -----
        .. math::
            P_{{\rm obs},\ell}(k';z)=\frac{1}{q_\perp^2 q_\parallel} \
            \frac{2\ell+1}{2}\int^1_{-1} L_\ell(\mu_k') \
            P_{\rm gg}\left(k(k',\mu_k'),\mu_k(\mu_k') \
            \PNT{;z}\right)\,{\rm d}\mu_k'\\
        """

        prefactor = 1.0 / self.scaling_factor_parall(z) / \
            (self.scaling_factor_perp(z))**2.0 * (2.0 * m + 1.0) / 2.0

        integral = prefactor * integrate.quad(self.multipole_spectra_integrand,
                                              a=-1.0, b=1.0, args=(z, k, m))[0]

        return integral
