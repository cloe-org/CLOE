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

# ATTENTION! JUST FOR NOW: test with
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path,
                    "../../data/ExternalBenchmark/matteo_bias.npy")

bias_dic = np.load(
    path,
    allow_pickle=True).item()

z_extended = np.linspace(2.6, 4, 100)


bias_interpolator = interpolate.InterpolatedUnivariateSpline(
    x=np.concatenate([bias_dic['z'], z_extended]),
    y=np.concatenate([bias_dic['bias'],
                      bias_dic['bias'][-1] * np.ones(len(z_extended))]),
    ext=2)


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
        r"""
        Computation of the perpendicular scaling factor

        .. math::
             q_{\perp} &= \frac{D_{\rm M}(z)}{D'_{\rm M}(z)}

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.

        Returns
        -------
        scaling_factor_perp: float
           Value of the scaling_factor_perp at given redshift

        """

        return self.theory['d_z_func'](z) / self.fiducial['d_z_func'](z)

    def scaling_factor_parall(self, z):
        r"""
        Computation of the parallel scaling factor

        .. math::
            q_{\parallel} &= \frac{H'(z)}{H(z)}

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.

        Returns
        -------
        scaling_factor_perp: float
           Value of the scaling_factor_perp at given redshift

        """
        return self.fiducial['H_z_func'](z) / self.theory['H_z_func'](z)

    def get_k(self, k_prime, mu_prime, z):
        r"""
        Computation of the wavenumber k

        .. math::
            k(k',\mu_k', ) &= k' \left[q_{\parallel}^{-2} \,
            \ (\mu_k')^2 + \
            \ q_{\perp}^{-2} \left( 1 - (\mu_k')^2 \right)\right]^{1/2}

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

        """
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - (mu_prime)**2))**(1. / 2)

    def get_mu(self, mu_prime, z):
        r"""
        Computation of the cosine of the angle

        .. math::
            \mu_k(\mu_k') &= \mu_k' \, q_{\parallel}^{-1}
            \left[ q_{\parallel}^{-2}\,
            (\mu_k')^2 + q_{\perp}^{-2}
            \left( 1 - (\mu_k')^2 \right) \right]^{-1/2}

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

        """

        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - (mu_prime)**2))**(-1. / 2)

    # SJ: Linear galaxy spectrum for now as a placeholder.
    # SJ: To be extended (i.e new function) by IST:NL.
    def pkgal_linear(self, mu, z, k):
        r"""
        Computation of P_gg appearing in the IST:L document
        (found below). Here, we express it using  linear theory.

        Parameters
        ----------
        mu: float
           Cosine of the angle between the wavenumber and LOS.
        z: float
           Redshift at which to evaluate power spectrum.
        k: float
           Scale (wavenumber) at which to evaluate power spectrum.

        Returns
        -------
        pkgal: float
           Linear galaxy power spectrum (including RSDs)

        Notes:
        ------
        https://www.overleaf.com/read/pvfkzvvkymbj

        .. math::
            P_{\rm gg}\left(k(k',\mu_k'),\mu_k(\mu_k');z\right) = \
            \left(b(z) + f(z)({\mu_k}')^2\right)^2 P_{\rm mm}(k';z)
        """
        # (GCH): For the moment, this does not work
        # Issue with k value and units

        growth = self.theory['fsigma8_z_func'](z) / \
            self.theory['sigma8_z_func'](z)

        #growth = self.theory['f_z_k']
        pkgal = self.theory['Pk_interpolator'].P(z, k) * \
            (bias_interpolator(z) + growth * mu**2.0)**2.0

        return pkgal

    def multipole_spectra_integrand(self, mu, z, k, m):
        r"""
        Computation of multipole power spectrum integrand
        appearing in the Euclid IST:L document, found below.
        Note we consider ell = m in the code.

        Parameters
        ----------
        mu: float
           Cosine of the angle between the wavenumber and LOS (AP-distorted).
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        k: float
           Scale (wavenumber) at which to evaluate the galaxy power
           spectrum (AP-distorted).
        m: float
           Order of the Legendre expansion.

        Returns
        -------
        integrand: float
           Integrand of multipole power spectrum

        Notes:
        ------
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
        Computation of multipole power spectra appearing in
        the Euclid IST:L document, found below.
        Note we consider ell = m in the code.

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        k: float
           Scale (wavenumber) at which to evaluate the galaxy power spectrum.
        m: float
           Order of the Legendre expansion.

        Returns
        -------
        integral: float
           Multipole power spectrum

        Notes:
        ------
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

    def multipole_spectra_noap(self, z, k):
        r"""
        Computation of Eqns 28 - 30 of Euclid IST:L documentation
        (corresponding to multipole power spectra pre geometric distortions),
        found below.

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum.
        k: float
           Scale (wavenumber) at which to evaluate the galaxy power spectrum.

        Notes:
        ------
        https://www.overleaf.com/read/pvfkzvvkymbj

        Eqns 28 - 30 shown in latex format below.

        .. math::
            P_0(k) &=\left(1+\frac{2}{3}\beta+\frac{1}{5}\beta^2\right)\,P(k)\\
            P_2(k) &=\left(\frac{4}{3}\beta+\frac{4}{7}\beta^2\right)\,P(k)\\
            P_4(k) &=\frac{8}{35}\beta^2\,P(k)

        """

        # SJ: Set up power spectrum [note weirdly Cobaya has it as P(z,k)
        # SJ: instead of the more common P(k,z)]
        beta = self.theory['fsigma8_z_func'](z) / \
            self.theory['sigma8_z_func'](z) / bias_interpolator(z)
        Pk_gal = bias_interpolator(z)**2.0 * \
            self.theory['Pk_interpolator'].P(z, k)
        P0k = (1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta**2.0) * Pk_gal
        P2k = (4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2.0) * Pk_gal
        P4k = (8.0 / 35.0 * beta**2.0) * Pk_gal

        return P0k, P2k, P4k
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
        return self.theory['Pk_interpolator'].P(0.5, 0.02)
