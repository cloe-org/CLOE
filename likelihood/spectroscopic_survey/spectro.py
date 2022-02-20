"""
module: spectro

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np
from scipy.special import legendre
from scipy import integrate


class Spectro:
    r"""
    Class for Galaxy clustering spectroscopic observable
    """

    def __init__(self, cosmo_dic, fiducial_dic):
        """Initialize

        Constructor of the class Spectro

        Parameters
        ----------
        cosmo_dic: dict
            cosmological dictionary from cosmo
        fiducial_dic: dict
            fiducial dictionary
        """
        self.theory = cosmo_dic
        self.fiducial = fiducial_dic

        mu_min = -1.0
        mu_max = 1.0
        mu_samp = 2001
        self.mu_grid = np.linspace(mu_min, mu_max, mu_samp)
        leg_m_max = 10
        self.dict_m_legendrepol = \
            {m: legendre(m)(self.mu_grid) for m in range(leg_m_max)}

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
        mu_prime: float or numpy.ndarray of float
           Fiducial cosine of the angle between the wavenumber and
           line of sight (Alcock–Paczynski distorted).

        Returns
        -------
        get_k: float or numpy.ndarray of float
           Value of the scalar wavenumber  at given redshift
           cosine of the angle and fiducial wavenumber
        """
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - mu_prime**2))**(1. / 2)

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
        mu_prime: float or numpy.ndarray of float
           Fiducial cosine of the angle between the wavenumber
           and line of sight (Alcock–Paczynski distorted).

        Returns
        -------
        get_mu: float or numpy.ndarray of float
           Value of the cosine of the angle at given redshift
           and fiducial wavenumber
        """

        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - mu_prime**2))**(-1. / 2)

    def multipole_spectra_integrand(self, mu_rsd, z, k, ms):
        r"""Multipole Spectra Integrand

        Computation of multipole power spectra integrand
        over the whole :math:`\mu_k'` grid [-1, 1].
        Note: we name :math:`\ell = m` in the code

        .. math::
            L_\ell(\mu_k')P_{\rm gg}^{\rm spectro}\
            \left[k(k',\mu_k'),\mu_k(\mu_k');z\right]\\

        Parameters
        ----------
        mu_rsd: numpy.ndarray of float
           Cosines of the angles between the wavenumber and
           line of sight (Alcock–Paczynski distorted).
           Warning: only mu_rsd = self.mu_grid works (issue 706)
        z: float
            Redshift at which to evaluate power spectrum.
        k: float
            Scale (wavenumber) at which to evaluate power spectrum.
        ms: list of int
            Order of the Legendre expansion.

        Returns
        -------
        integrand: numpy.ndarray of numpy.ndarray of float
            Integrand (over :math:`\mu_k'` between [-1,1])
            of multipole power spectrum expansion, for all m
        """
        if self.theory['Pgg_spectro'] is None:
            raise Exception('Pgg_spectro is not defined inside the cosmo dic. '
                            'Run update_cosmo_dic() method first.')

        galspec = \
            self.theory['Pgg_spectro'](z, self.get_k(k, mu_rsd, z),
                                       self.get_mu(mu_rsd, z))

        if np.array_equal(mu_rsd, self.mu_grid):
            return galspec * np.array([self.dict_m_legendrepol[m] for m in ms])
        else:
            return galspec * np.array([legendre(m)(mu_rsd) for m in ms])

    def multipole_spectra(self, z, k, ms=None):
        r"""Multipole Spectra

        Computation of multipole power spectra.
        Note: we name :math:`\ell = m` in the code.

        .. math::
            P_{{\rm obs},\ell}(k';z)=\frac{1}{[q_\perp(z)]^2 q_\parallel(z)} \
            \frac{2\ell+1}{2}\int^1_{-1} L_\ell(\mu_k') \
            P_{\rm gg}^{\rm spectro}\left[k(k',\mu_k'),\mu_k(\mu_k') \
            {;z}\right]\,{\rm d}\mu_k'\\

        Parameters
        ----------
        z: float
            Redshift at which to evaluate power spectrum.
        k: float
            Scale (wavenumber) at which to evaluate power spectrum.
        ms: list of int
            Orders of the Legendre expansion. Default is [0, 2, 4]

        Returns
        -------
        spectra: numpy.ndarray of float
            Multipole power spectra
        """

        if ms is None:
            ms = [0, 2, 4]

        constant = 1.0 / self.scaling_factor_parall(z) / \
            self.scaling_factor_perp(z) ** 2.0 / 2.0

        prefactors = np.array([constant * (2.0 * m + 1.0) for m in ms])

        integrals = \
            integrate.simps(self.multipole_spectra_integrand(self.mu_grid,
                                                             z, k, ms),
                            self.mu_grid)

        spectra = prefactors * integrals

        return np.asarray(spectra)
