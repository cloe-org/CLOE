"""
SPECTRO

This module computes the spectroscopic quantities following the
v2.0 recipes of CLOE.
"""

# Global
import numpy as np
from scipy.special import legendre
from scipy import integrate
from cloe.fftlog.fftlog import fftlog


class Spectro:
    r"""
    Class for the spectroscopic observables.
    """

    def __init__(self, cosmo_dic, z_str):
        """Initialises the class.

        Constructor of the class :obj:`Spectro`.

        Parameters
        ----------
        cosmo_dic: dict
            Cosmology dictionary containing the current cosmology
        z_str: list of str
            List of the redshift bin centers
        """
        if cosmo_dic is not None:
            self.update(cosmo_dic)

        mu_min = -1.0
        mu_max = 1.0
        mu_samp = 2001
        self.mu_grid = np.linspace(mu_min, mu_max, mu_samp)
        leg_m_max = 10
        self.dict_m_legendrepol = \
            {m: legendre(m)(self.mu_grid) for m in range(leg_m_max)}
        self.z_arr = np.array(z_str).astype("float")

    def update(self, cosmo_dic):
        r"""Updates method.

        Method to update the :obj:`theory` class attribute to the
        dictionaries passed as input.

        Parameters
        ----------
        cosmo_dic: dict
            Cosmology dictionary containing the current cosmology
        """
        self.theory = cosmo_dic

    def scaling_factor_perp(self, z):
        r"""Perpendicular scaling factor.

        Computation of the perpendicular scaling factor.

        .. math::
            q_{\perp}(z) &= \frac{D_{\rm M}(z)}{D'_{\rm M}(z)}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum

        Returns
        -------
        Perpendicular scaling factor: float
           Value of the perpendicular scaling factor at given redshift
        """
        return self.theory['d_z_func'](z) / self.theory['fid_d_z_func'](z)

    def scaling_factor_parall(self, z):
        r"""Parallel scaling factor.

        Computation of the parallel scaling factor.

        .. math::
            q_{\parallel}(z) &= \frac{H'(z)}{H(z)}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum

        Returns
        -------
        Parallel scaling factor: float
           Value of the the parallel scaling factor at a given redshift
        """
        return self.theory['fid_H_z_func'](z) / self.theory['H_z_func'](z)

    def get_k(self, k_prime, mu_prime, z):
        r"""Gets wavenumber.

        Computation of the wavenumber k.

        .. math::
            k(k',\mu_k', z) &= k' \left[[q_{\parallel}(z)]^{-2} \,
            (\mu_k')^2 + \
            [q_{\perp}(z)]^{-2} \left( 1 - (\mu_k')^2 \right)\right]^{1/2}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum
        k_prime: float
           Fiducial scale (wavenumber) at which to evaluate the galaxy power
        mu_prime: float or numpy.ndarray of float
           Fiducial cosine of the angle between the wavenumber and
           line of sight (Alcock–Paczynski distorted)

        Returns
        -------
        Wavenumber: float or numpy.ndarray of float
           Value of the scalar wavenumber  at given redshift
           cosine of the angle and fiducial wavenumber
        """
        return k_prime * (self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
                          self.scaling_factor_perp(z)**(-2) *
                          (1 - mu_prime**2))**(1. / 2)

    def get_mu(self, mu_prime, z):
        r"""Gets cosine of the angle.

        Computation of the cosine of the angle.

        .. math::
            \mu_k(\mu_k') &= \mu_k' \, [q_{\parallel}(z)]^{-1}
            \left[ [q_{\parallel}(z)]^{-2}\,
            (\mu_k')^2 + [q_{\perp}(z)]^{-2}
            \left( 1 - (\mu_k')^2 \right) \right]^{-1/2}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate the galaxy power spectrum
        mu_prime: float or numpy.ndarray of float
           Fiducial cosine of the angle between the wavenumber
           and line of sight (Alcock–Paczynski distorted)

        Returns
        -------
        Cosine of the angle: float or numpy.ndarray of float
           Value of the cosine of the angle at given redshift
           and fiducial wavenumber
        """

        return mu_prime * self.scaling_factor_parall(z)**(-1) * (
            self.scaling_factor_parall(z)**(-2) * mu_prime**2 +
            self.scaling_factor_perp(z)**(-2) *
            (1 - mu_prime**2))**(-1. / 2)

    def gal_redshift_scatter(self, k, mu_rsd, z):
        r"""Unbiased scatter in the measured galaxy redshifts.

        Computes the correction factor to the galaxy power
        spectrum, when taking into account errors in the measured
        redshift of the galaxies.

        .. math::
            F_z(k(k',\mu_k'),\mu(\mu_k');z) = \
                e^{-k^2\mu_k^2[\frac{c}{H(z)}\sigma_{z_0}]^2}

        Parameters
        ----------
        k: float or numpy.ndarray of float
            Scale (wavenumber) at which to evaluate the correction
            factor
        mu_rsd: numpy.ndarray of float
           Cosines of the angles between the wavenumber and the
           line of sight (Alcock–Paczynski distorted)
           Warning: only mu_rsd = self.mu_grid works (issue 706)
        z: float or numpy.ndarray of float
            Redshift at which to evaluate the correction factor

        Returns
        -------
        Factor: float or numpy.ndarray of float
            Value of the correction factor :math:`F_z` due to errors
            in measured galaxy redshift, for given values of k, mu
            and redshift.
        """

        sigma_z = self.theory['nuisance_parameters']['sigma_z']
        sigma_r = self.theory['c'] * sigma_z / self.theory['fid_H_z_func'](z)

        return np.exp(-k**2 * mu_rsd**2 * sigma_r**2)

    def multipole_spectra_integrand(self, mu_rsd, z, k, ms):
        r"""Multipole power spectrum integrand.

        Computation of multipole power spectrum integrand
        over the whole :math:`\mu_k'` grid [-1, 1].
        Note: we name :math:`\ell = m` in the code

        .. math::
            L_\ell(\mu_k')P_{\rm gg}^{\rm spectro}\
            \left[k(k',\mu_k'),\mu_k(\mu_k');z\right]\\

        Parameters
        ----------
        mu_rsd: numpy.ndarray of float
           Cosines of the angles between the wavenumber and
           line of sight (Alcock–Paczynski distorted)
           Warning: only mu_rsd = self.mu_grid works (issue 706)
        z: float
            Redshift at which to evaluate power spectrum
        k: float
            Scale (wavenumber) at which to evaluate power spectrum
        ms: list of int
            Order of the Legendre expansion

        Returns
        -------
        Integrand: numpy.ndarray of numpy.ndarray of float
            Integrand (over :math:`\mu_k'` between [-1,1])
            of multipole power spectrum expansion, for all m.
        """
        if self.theory['Pgg_spectro'] is None:
            raise Exception('Pgg_spectro is not defined inside the cosmo dic. '
                            'Run update_cosmo_dic() method first.')

        galspec = \
            self.theory['Pgg_spectro'](z, self.get_k(k, mu_rsd, z),
                                       self.get_mu(mu_rsd, z))

        if self.theory['GCsp_z_err']:
            # Get redshift error correction term
            z_err = self.gal_redshift_scatter(self.get_k(k, mu_rsd, z),
                                              self.get_mu(mu_rsd, z), z)

            # Multiply by redshift error correction
            galspec *= z_err

        # Find the outlier fraction value in the nuisance_parameters dictionary
        if self.theory['f_out_z_dep']:
            if np.where(np.isclose(self.z_arr, z))[0].size == 1:
                iz = int(np.where(np.isclose(self.z_arr, z))[0])
            else:
                raise Exception('Problem matching redshift to bin center in'
                                'multipole_spectra')
            # redshift-dependent case
            f_out = self.theory['nuisance_parameters']['f_out_' + str(iz + 1)]
        else:
            # redshift-independent case
            f_out = self.theory['nuisance_parameters']['f_out']

        outlier_factor = (1.0 - f_out) ** 2.0

        galspec *= outlier_factor

        # Add shot-noise contributions (after the systematics!)
        noise = \
            self.theory['noise_Pgg_spectro'](z, self.get_k(k, self.mu_grid, z),
                                             self.get_mu(self.mu_grid, z))
        galspec += noise

        if np.array_equal(mu_rsd, self.mu_grid):
            return galspec * np.array([self.dict_m_legendrepol[m] for m in ms])
        else:
            return galspec * np.array([legendre(m)(mu_rsd) for m in ms])

    def multipole_spectra(self, z, k, ms=None):
        r"""Multipole power spectra.

        Computation of multipole power spectra.
        Note: we name :math:`\ell = m` in the code.

        .. math::
            P_{{\rm obs},\ell}(k';z)=(1-f_{out})^2\frac{1}{[q_\perp(z)]^2 \
            q_\parallel(z)} \frac{2\ell+1}{2}\int^1_{-1} L_\ell(\mu_k') \
            P_{\rm gg}^{\rm spectro}\left[k(k',\mu_k'),\mu_k(\mu_k') \
            {;z}\right]\,{\rm d}\mu_k'\\

        Parameters
        ----------
        z: float
            Redshift at which to evaluate power spectrum
        k: float
            Scale (wavenumber) at which to evaluate power spectrum
        ms: list of int
            Orders of the Legendre expansion. Default is [0, 2, 4]

        Returns
        -------
        Spectra: numpy.ndarray of float
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

    def multipole_correlation_function(self, s, z, ell,
                                       k_min=5e-5,
                                       k_max=50,
                                       k_num_points=2048):
        r"""Evaluates the multipole correlation function.

        Evaluates the multipole correlation function for the separation values
        contained in the :math:`s` array, at the requested values of redshift
        :math:`z` and multipole :math:`\ell`.

        Parameters
        ----------
        s: numpy.array of float
            Array of :math:`s` values in Mpc/h
        z: float
            Value of redshift, among {1.00, 1.20, 1.40, 1.65}
        ell: int or array of int
            Value or array of values of :math:`\ell` among {0, 2, 4}
        k_min: float
           Lower bound of the range of wavenumbers used for the evaluation of
           the multipole power spectra
        k_max: float
           Upper bound of the range of wavenumbers used for the evaluation of
           the multipole power spectra
        k_num_points: int
           Number of points used for the evaluation of the multipole
           power spectra

        Returns
        -------
        Multipole correlation function: numpy.ndarray of float
            Array of shape (len(ell), len(s)), containing the values of the
            multipole correlation function evaluated in correspondence of the
            :math:`\ell` and :math:`s` values provided as input
        """
        # vectorize input ell if it is a scalar
        if np.isscalar(ell):
            ell = [ell]

        # define the log-spaced k grid to evaluate the multipole spectra
        k_grid = np.logspace(np.log10(k_min), np.log10(k_max), k_num_points)

        # evaluate multipole spectra in correspondence of the points in k_grid.
        # This instruction produces one pk_array for each input ell value
        pk_arrays = np.transpose([self.multipole_spectra(z, k, ell)
                                 for k in k_grid])

        # perform the Hankel transform
        transformed_lin = np.empty((len(ell), len(s)))
        volume_factor = (k_grid**3) / (2 * (np.pi**2))
        for id in range(len(ell)):
            y_array = volume_factor * pk_arrays[id] * np.real(1j**ell[id])
            transformer = fftlog(k_grid, y_array)
            s_grid, transformed_log = transformer.fftlog(ell[id])
            # interpolate to get the linearly-spaced values of the multipole
            # correlation function from the log-spaced ones.
            transformed_lin[id] = np.interp(s, s_grid, transformed_log)

        return transformed_lin
