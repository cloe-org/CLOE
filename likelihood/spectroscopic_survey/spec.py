"""
module: spec

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np
from likelihood.cosmo.cosmology import Cosmology
from scipy.special import legendre
from scipy import integrate
from scipy import interpolate
import os.path


class Spec:
    r"""
    Class for Galaxy clustering spectroscopic observable
    """

    def __init__(self, cosmo_dic, fiducial_dic):
        """Initialize

        Constructor of the class Spec

        Parameters
        ----------
        cosmo_dic: dict
            cosmological dictionary from cosmo
        fiducial_dic: dict
            fiducial dictionary
        """
        self.theory = cosmo_dic
        self.fiducial = fiducial_dic

    def scaling_factor_perp(self, z):
        r"""Scaling Factor Perp

        Computation of the perpendicular scaling factor

        .. math::
            q_{\perp}(z) &= \frac{D_{\rm M}(z)}{D'_{\rm M}(z)}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.

        Returns
        -------
        scaling_factor_perp: float
           Value of the perpendicular scaling factor at given redshift
        """

        return self.theory['d_z_func'](z) / self.fiducial['d_z_func'](z)

    def scaling_factor_parall(self, z):
        r"""Scaling Factor Parall

        Computation of the parallel scaling factor

        .. math::
            q_{\parallel}(z) &= \frac{H'(z)}{H(z)}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.

        Returns
        -------
        scaling_factor_parall: float
           Value of the the parallel scaling factor at a given redshift
        """
        return self.fiducial['H_z_func'](z) / self.theory['H_z_func'](z)

    def get_k(self, k_prime, mu_prime, z):
        r"""Get k

        Computation of the wavenumber k

        .. math::
            k(k',\mu_k', z) &= k' \left[[q_{\parallel}(z)]^{-2} \,
            (\mu_k')^2 + \
            [q_{\perp}(z)]^{-2} \left( 1 - (\mu_k')^2 \right)\right]^{1/2}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        k_prime: float
           Fiducial scale (wavenumber) at which to evaluate the galaxy power
        mu_prime: float
           Fiducial cosine of the angle between the wavenumber and
           line of sight (Alcock–Paczynski distorted).

        Returns
        -------
        get_k: float
           Value of the scalar wavenumber  at given redshift
           cosine of the angle and fiducial wavenumber
        """
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - (mu_prime)**2))**(1. / 2)

    def get_mu(self, mu_prime, z):
        r"""Get Mu

        Computation of the cosine of the angle

        .. math::
            \mu_k(\mu_k') &= \mu_k' \, [q_{\parallel}(z)]^{-1}
            \left[ [q_{\parallel}(z)]^{-2}\,
            (\mu_k')^2 + [q_{\perp}(z)]^{-2}
            \left( 1 - (\mu_k')^2 \right) \right]^{-1/2}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        mu_prime: float
           Fiducial cosine of the angle between the wavenumber
           and line of sight (Alcock–Paczynski distorted).

        Returns
        -------
        get_mu: float
           Value of the cosine of the angle at given redshift
           and fiducial wavenumber
        """

        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - (mu_prime)**2))**(-1. / 2)

    def multipole_spectra_integrand(self, mu_rsd, z, k, m):
        r"""Multipole Spectra Integrand

        Computation of multipole power spectrum integrand.
        Note: we consider :math:`\ell = m` in the code

        .. math::
            L_\ell(\mu_k')P_{\rm gg}^{\rm spec}\
            \left[k(k',\mu_k'),\mu_k(\mu_k');z\right]\\


        Parameters
        ----------
        mu_rsd: float
           Cosine of the angle between the wavenumber and
           line of sight (Alcock–Paczynski distorted).
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
        """
        if self.theory['Pgg_spec'] is None:
            raise Exception('Pgg_spec is not defined inside the cosmo dic. '
                            'Run update_cosmo_dic() method first.')
        legendrepol = legendre(m)(mu_rsd)
        integrand = (self.theory['Pgg_spec'](z, self.get_k(k, mu_rsd, z),
                                             self.get_mu(mu_rsd, z)) *
                     legendrepol)

        return integrand

    def multipole_spectra(self, z, k, m):
        r"""Multipole Spectra

        Computation of multipole power spectra.
        Note: we consider :math:`\ell = m` in the code.

        .. math::
            P_{{\rm obs},\ell}(k';z)=\frac{1}{[q_\perp(z)]^2 q_\parallel(z)} \
            \frac{2\ell+1}{2}\int^1_{-1} L_\ell(\mu_k') \
            P_{\rm gg}^{\rm spec}\left[k(k',\mu_k'),\mu_k(\mu_k') \
            {;z}\right]\,{\rm d}\mu_k'\\


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
        """

        prefactor = 1.0 / self.scaling_factor_parall(z) / \
            (self.scaling_factor_perp(z))**2.0 * (2.0 * m + 1.0) / 2.0

        integral = prefactor * integrate.quad(self.multipole_spectra_integrand,
                                              a=-1.0, b=1.0, args=(z, k, m))[0]

        return integral
