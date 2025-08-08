"""
SPECTRO

This module computes the spectroscopic quantities following the
theoretical recipe of CLOE.
"""

# Global
import numpy as np
from scipy.special import legendre, binom
from scipy import integrate
from scipy.interpolate import interp1d
from cloe.fftlog import fftlog, hankel


class Spectro:
    r"""
    Class for the spectroscopic observables.
    """

    def __init__(self, cosmo_dic, z_str, mixing_matrix_dict=None):
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

        self.mixing_matrix_dict = mixing_matrix_dict

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
           line of sight (Alcock-Paczynski distorted)
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

    def convolved_power_spectrum_multipoles(self, redshift):
        r"""Power spectrum multipoles convolved with the mixing matrix

        Returns the power spectrum multipoles convolved with the mixing matrix
        stored as class attribute.

        .. math::
            P_{\ell}^{\rm obs}(k) = \int {\rm d}k'\,{k'}^2\,\sum_{\ell'} \
            W_{\ell\ell'}(k,k')P_{\ell'}(k')

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the convolved power spectrum
            multipoles.

        Returns
        -------
        mps: list of numpy.ndarray
            List containing the convolved power spectrum multipoles of order
            (0,2,4). The list includes the three multipoles, in this order.
        """
        if self.mixing_matrix_dict is None:
            raise TypeError('Mixing matrix has not been initialised since no '
                            'argument "mixing_matrix_dict" was specified when '
                            'instantiating the Spectro class.')

        kin0 = self.mixing_matrix_dict['kin0']
        kin2 = self.mixing_matrix_dict['kin2']
        kin4 = self.mixing_matrix_dict['kin4']
        kout = self.mixing_matrix_dict['kout']

        multipoles_in = {}
        for ell in [0, 2, 4]:
            multipoles_in[f'ell{ell}'] = np.empty(
                self.mixing_matrix_dict[f'kin{ell}'].shape)
            if np.all(kin0 == kin2) and np.all(kin0 == kin4):
                for i, kk in enumerate(kin0):
                    multipoles_in[f'ell{ell}'][i] = \
                        self.multipole_spectra(redshift, kk, ms=[ell])
            else:
                for i, kk in enumerate(self.mixing_matrix_dict[f'kin{ell}']):
                    multipoles_in[f'ell{ell}'][i] = \
                        self.multipole_spectra(redshift, kk, ms=[ell])

        multipoles_out = {}
        for ell in [0, 2, 4]:
            multipoles_out[f'ell{ell}'] = np.zeros(kout.shape)
            for ell_prime in [0, 2, 4]:
                multipoles_out[f'ell{ell}'] += \
                    np.dot(self.mixing_matrix_dict[f'W{ell}{ell_prime}'],
                           multipoles_in[f'ell{ell_prime}'])

        return (multipoles_out['ell0'], multipoles_out['ell2'],
                multipoles_out['ell4'])

    def multipole_correlation_function_mag_mag(self, r_xi, z, ell):
        r"""Evaluates the magnification-magnification correlation function

        Evaluates the magnification-magnification bias contribution to the
        spectroscopic galaxy clustering correlation function for the
        separation values in the :math:`r_\xi` array, at the requested
        value of redshift :math:`z` and multipole :math:`\ell`.

        .. math::
            \xi_{\ell}^{\mu\mu}(r;z)=C_{\mu\mu}(\ell) \
            \frac{9\Omega^2_{m,0}{H^4_0}}{8\pi}[2-5s_{\rm bias}(z)]^2r^3(z) \
            \int^{1}_{0}f_{\ell}(x,r,z)dx\\

        Parameters
        ----------
        r_xi: numpy.ndarray of float
            Array of :math:`r_\xi` values in Mpc
        z: float
            A value of redshift, among {1.00, 1.20, 1.40, 1.65}
        ell: int
            A value of :math:`\ell` among {0, 2, 4}

        Returns
        -------
        xi_mu_mu: numpy.ndarray of float
            Array of shape len(r_xi), containing the values
            of the magnification-magnification correlation function
            evaluated at the :math:`\ell` and :math:`r_\xi` values provided.
        """
        # Retrieve magnification bias value corresponding to redshift bin
        spec_zs = [1.0, 1.2, 1.4, 1.65]
        nuisance_dict = self.theory['nuisance_parameters']
        try:
            spec_bin = spec_zs.index(z) + 1
            mag_bi = nuisance_dict[f'magnification_bias_spectro_bin{spec_bin}']
        except ValueError:
            raise ValueError('Spectroscopic magnification bias cannot '
                             'be obtained. Check that redshift is '
                             'inside the bin edges.')

        def K_ell(ell, dx, zx):
            r"""Evaluates the integral :math:`K_{\ell}(xr)`

            Evaluates :math:`K_{\ell}(xr)` given a value of :math:`ell`,
            an array of redshifts :math:`zx` and integration steps
            :math:`x` by performing the integration using fftlog.

            .. math::
                K_{\ell}(xr) = \int^{\infty}_{0}k^2P_{\delta}{\delta} \
                [k,z(x_{rad})]\frac{j_{\ell}(xkr)}{xkr} dk\\

            Parameters
            ----------
            ell: int
                A value of :math:`\ell` among {0, 2, 4}
            dx: numpy.ndarray of float
                Array of integration steps :math:`x`
            zx: numpy.ndarray of float
                Array of redshifts :math:`z(x_rad)` corresponding to the
                radial comoving distance :math:`x_{rad}`

            Returns
            -------
            K_ell_xr: numpy.ndarray of float
                Array of size len(dx) of the integral :math:`K_{\ell}(xr)`
            """
            nu = 1.01
            # Define the integrand
            integ = self.theory['Pk_delta'].P(zx, self.theory['k_win']) * \
                self.theory['k_win'] ** 2
            # Perform the integration for each step in dx
            K_ell_xr = np.array([])
            for i, xx in enumerate(dx):
                fftlog_integral = fftlog.fftlog(self.theory['k_win'],
                                                integ[i], nu=nu,
                                                N_extrap_begin=1000,
                                                N_extrap_end=1000,
                                                c_window_width=0.25,
                                                N_pad=1000)
                d, Int = fftlog_integral.fftlog(ell)
                # Interpolate for desired values of dx
                K_ell_xr = np.append(K_ell_xr, interp1d(d, Int)(xx))
            return K_ell_xr

        def f_ell(ell, dx, r):
            r"""Evaluates the integrand :math:`f_{\ell}(x,r,z)`

            Evaluates the intergand :math:`f_{\ell}(x,r,z)` given a value
            of :math:`ell`, :math:`r_\xi` and step :math:`x` to
            integrate over.

            .. math::
                f_{\ell}(x,r) = x^2(1-x^2)[1+z(x_{rad})]^2(xr) \
                \int^{\infty}_{0}k^2P_{\delta}{\delta}[k,z(x_{rad})] \
                \frac{j_{\ell}(xkr)}{xkr} dk\\

            Parameters
            ----------
            ell: int
                A value of :math:`\ell` among {0, 2, 4}
            dx: numpy.ndarray of float
                Array of integration steps :math:`x`
            r: float
                A value of :math:`r_\xi` in Mpc

            Returns
            -------
            f_ell: numpy.ndarray of float
                Array of size len(dx) corresponding to the values
                of the integrand :math:`f_ell`
            """
            # Interpolate the redshift z at the radial comoving distance xr
            interp_comov_dist = interp1d(self.theory['comov_dist'],
                                         self.theory['z_win'])
            comov_dist_z = interp_comov_dist(dx * self.theory['r_z_func'](z))
            # Calcuate f_ell
            f_ell = dx ** 2 * (1 - dx) ** 2 * \
                (1 + comov_dist_z) ** 2 * \
                K_ell(ell, dx * r, comov_dist_z)
            return f_ell

        def compute_xi_mu_mu(ell, r):
            r"""Computes a single value of :math:`xi_mu_mu`

            Evaluates the magnification-magnification bias contribution
            to the correlation function for a single :math:`r_\xi`, at
            the requested value of redshift :math:`z` and multipole
            :math:`\ell`.

            Parameters
            ----------
            ell: int
                A value of :math:`\ell` among {0, 2, 4}
            r: float
                A value of :math:`r_\xi` in Mpc

            Returns
            -------
            xi_mu_mu: float
                Value of magnification-magnification correlation function
                at given :math:`\ell` and :math:`r_\xi`
            """
            # Define the step size and x array for integration
            x_step = [0.004, 0.010, 0.016]
            try:
                dx = np.linspace(x_step[int(ell / 2)] / 40.0, 1.0,
                                 num=10, endpoint=True)
            except ValueError:
                raise ValueError('Magnification-magnification bias '
                                 'cannot be calculated for requested '
                                 'ell value = 'f'{ell}')
            # Calculate the integrand f_ell(x,s,z) as an array
            f_ells = f_ell(ell, dx, r)
            # Perform the integration
            integ_f_ell = integrate.simps(f_ells, dx)
            # Calculate the coefficient C_mu_mu(ell)
            C_ell = (2 * ell + 1) / 2 * np.math.factorial(ell) / \
                (2 ** (ell - 1) * np.math.factorial(int(ell / 2)) ** 2)
            # Calculate xi_mu_mu
            xi_mu_mu_r = C_ell * 9 * \
                (self.theory['Omc'] + self.theory['Omb']) ** 2 * \
                self.theory['H0_Mpc'] ** 4 / (8 * np.pi) * \
                (2 - 5 * mag_bi) ** 2 * \
                self.theory['r_z_func'](z) ** 3 * integ_f_ell
            return xi_mu_mu_r

        # Calculate xi_mu_mu for the array of s
        xi_mu_mu = np.array([compute_xi_mu_mu(ell, r) for r in r_xi])

        return xi_mu_mu

    def multipole_correlation_function_dens_mag(self, r_xi, z, ell):
        r"""Evaluate the density-magnification correlation function

        Evaluates the density-magnification bias contribution to the
        spectroscopic galaxy clustering correlation function for the
        separation values in the :math:`r_\xi` array, at the requested
        value of redshift :math:`z` and multipole :math:`\ell`.

        .. math::
            \xi_{\ell}^{g\mu}(r;z)=&-C_{g\mu}(\ell) \
            \frac{3\Omega_{m,0}{H^2_0}}{4\pi}b(z)s_{\rm bias}(z)^2 \
            [2-5s_{\rm bias}(z)](1+z)  \\ & \times \sum_{n=0}^{\ell/2} \
            \frac{(-1)^n}{2^n}{\ell \choose n}{2\ell-2n \choose \ell} \
            \left(\frac{\ell}{2}-n \right)!I^{\ell/2-n+3/2}_{\ell/2-n+1/2} \
            (r;z)\\

        Parameters
        ----------
        r_xi: numpy.ndarray of float
            Array of :math:`r_\xi` values in Mpc
        z: float
            A value of redshift, among {1.00, 1.20, 1.40, 1.65}
        ell: int
            A value of :math:`\ell` among {0, 2, 4}

        Returns
        -------
        xi_dens_mu: numpy.ndarray of float
            Array of shape len(r_xi), containing the values of the
            density-magnification bias correlation function evaluated at the
            given :math:`\ell` and :math:`r_\xi` values
        """
        # Retrieve galaxy and magnification bias corresponding to the redshift
        spec_zs = [1.0, 1.2, 1.4, 1.65]
        nuisance_dict = self.theory['nuisance_parameters']
        try:
            spec_bin = spec_zs.index(z) + 1
            mag_bi = nuisance_dict[f'magnification_bias_spectro_bin{spec_bin}']
            gal_bi = nuisance_dict[f'b1_spectro_bin{spec_bin}']
        except ValueError:
            raise ValueError('Spectroscopic galaxay and magnification bias '
                             'cannot be obtained. Check that redshift is '
                             'inside the bin edges. ')
        # Calculate the coefficient C_g_mu(ell)
        C_gmu = (2 * ell + 1) / 2 * np.pi ** (3 / 2) * \
            2 ** (3 / 2) / 2 ** (ell / 2)
        # Calculate the prefactor of xi_g_mu
        prefactor = -C_gmu * 3 * (self.theory['Omc'] + self.theory['Omb']) * \
            self.theory['H0_Mpc'] ** 2 / (4 * np.pi) * 2 * gal_bi * \
            (2 - 5 * mag_bi) * (1 + z) * r_xi ** 2

        def I_n_l(nx, r_xi):
            r"""Evaluate I_n_l(r,z)

            Performs the integration to return I_n_l(r,z) within the
            equation of :math:`\xi_\ell^{g\mu}`, for a given :math:`n`
            and array :math:`r_\xi`, using the Hankel transform method.

            .. math::
                I_n^{\ell}(r,z)=\frac{1}{2\pi^2}\int_0^{\infty \
                k^2P_{\delta\delta}(k,z)\frac{j_\ell(kr)}{(kr)^n}\\

            Parameters
            ----------
            nx: int
                A value of :math:`n` among {0, 1, 2}
            r_xi: numpy.ndarray of float
                Array of :math:`r_\xi` values in Mpc

            Returns
            -------
            fx: float
                A value of the summation
            """
            nu = 1.01
            # Define the integrand
            integ = self.theory['Pk_delta'].P(z, self.theory['k_win']) / \
                self.theory['k_win'] ** (nx + 1)
            # Perform the integration
            hankel_integral = hankel.hankel(self.theory['k_win'],
                                            integ, nu=nu,
                                            N_extrap_begin=1500,
                                            N_extrap_end=1500,
                                            c_window_width=0.25,
                                            N_pad=1000)
            d, Int = hankel_integral.hankel(nx + 1)
            # Interpolate for desired values of s
            interp_int = interp1d(d, Int)(r_xi)
            I_n_l = 1 / (2 * np.pi ** 2) * np.sqrt(np.pi / 2) * \
                1 / (r_xi ** (nx + 2)) * interp_int
            return I_n_l

        def sum_fact(ell, nx):
            r"""Evaluate the summation expression

            Evaluates the summation expression within the equation of
            :math:`xi_\ell^{g\mu}`, given a value of :math:`\ell` and
            :math:`r_\xi`.

            .. math::
                \sum_{n=0}^{\ell/2}\frac{(-1)^n}{2^n}{\ell \choose n} \
                {2\ell-2n \choose \ell}\left(\frac{\ell}{2}-n \right)!\\

            Parameters
            ----------
            ell: int
                A value of :math:`\ell` among {0, 2, 4}
            nx: int
                A value of :math:`n` among {0, 1, 2}

            Returns
            -------
            fx: float
                A value of the summation
            """
            fx = (-1) ** nx / (2 ** nx) * binom(ell, nx) * \
                binom(2 * ell - 2 * nx, ell) * \
                np.math.factorial(int(ell / 2 - nx))
            return fx

        # Calculate xi_g_mu according to ell value
        if ell == 0:
            xi_g_mu = prefactor * I_n_l(0, r_xi)
        elif ell == 2:
            xi_g_mu = prefactor * (sum_fact(ell, 0) * I_n_l(1, r_xi) +
                                   sum_fact(ell, 1) * I_n_l(0, r_xi))
        elif ell == 4:
            xi_g_mu = prefactor * (sum_fact(ell, 0) * I_n_l(2, r_xi) +
                                   sum_fact(ell, 1) * I_n_l(1, r_xi) +
                                   sum_fact(ell, 2) * I_n_l(0, r_xi))
        else:
            raise ValueError('Density-magnification bias '
                             'cannot be calculated for requested '
                             'ell value = ' f'{ell}')
        return xi_g_mu

    def multipole_correlation_function(self, r_xi, z, ell,
                                       k_min=5e-5,
                                       k_max=50,
                                       k_num_points=2048):
        r"""Evaluates the multipole correlation function.

        Evaluates the multipole correlation function for the separation values
        contained in the :math:`r_\xi` array, at the requested values of
        redshift :math:`z` and multipole :math:`\ell`.

        Parameters
        ----------
        r_xi: numpy.ndarray of float
          Array of :math:`r_\xi` values in Mpc
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
            Array of shape (len(ell), len(r_xi)), containing the values of the
            multipole correlation function evaluated in correspondence of the
            :math:`\ell` and :math:`r_\xi` values provided as input
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
        transformed_lin = np.empty((len(ell), len(r_xi)))
        volume_factor = (k_grid**3) / (2 * (np.pi**2))
        for id in range(len(ell)):
            y_array = volume_factor * pk_arrays[id] * np.real(1j**ell[id])
            transformer = fftlog.fftlog(k_grid, y_array)
            r_grid, transformed_log = transformer.fftlog(ell[id])
            # interpolate to get the linearly-spaced values of the multipole
            # correlation function from the log-spaced ones.
            transformed_lin[id] = np.interp(r_xi, r_grid, transformed_log)

        # add magnification bias contribution
        if self.theory['use_magnification_bias_spectro']:
            for id, ells in enumerate(ell):
                xi_dens_mag = \
                    self.multipole_correlation_function_dens_mag(r_xi, z, ells)
                xi_mag_mag = \
                    self.multipole_correlation_function_mag_mag(r_xi, z, ells)
                transformed_lin[id] += 2 * xi_dens_mag
                transformed_lin[id] += xi_mag_mag

        return transformed_lin
