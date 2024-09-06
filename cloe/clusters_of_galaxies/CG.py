"""
module: CG

This module computes the galaxy clusters theoretical observables quantities.
"""

import numpy as np
import os.path
from cloe.cosmo.cosmology import Cosmology
from scipy import integrate, interpolate
from scipy.special import gamma, eval_legendre, spherical_jn, erf, j0, j1
from scipy.integrate import simps, quad, quad_vec
from astropy import units
from astropy.constants import G


class CG:
    r"""
    Class for clusters of galaxies likelihood observable
    """

    def __init__(self, cosmo_dic):
        """Initialise

        Constructor of the class CG

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary from :obj:`cosmology` class
        """

        self.theory = cosmo_dic
        # self.fiducial = fiducial_dic
        self.profile = self.theory[
            'observables_specifications']['CG']['halo_profile']
        self.overdensity_type = self.theory[
            'observables_specifications']['CG']['overdensity_type']
        self.overdensity = self.theory[
            'observables_specifications']['CG']['overdensity']

        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

        self.neutrino_cdm = self.theory['observables_specifications']['CG'][
            'neutrino_cdm']

        # Effective area in deg^2
        self.area = self.theory[
            'observables_specifications']['CG']['effective_area']

        # Observed redshift and richness binning
        self.z_obs_edges_CC = self.theory[
            'observables_specifications']['CG']['z_obs_edges_CC']
        self.lambda_obs_edges_CC = self.theory[
            'observables_specifications']['CG']['Lambda_obs_edges_CC']

        if isinstance(self.z_obs_edges_CC, list):
            self.z_obs_edges_CC = np.array(self.z_obs_edges_CC)
        if isinstance(self.lambda_obs_edges_CC, list):
            self.lambda_obs_edges_CC = np.array(self.lambda_obs_edges_CC)

        self.z_obs_div_CC = len(self.z_obs_edges_CC) - 1
        self.lambda_obs_div_CC = len(self.lambda_obs_edges_CC) - 1

        # Selection function
        self.external_richness_selection_function = self.theory[
            'observables_specifications']['CG'][
                'external_richness_selection_function']
        self.selection_function = self.theory[
            'observables_specifications']['CG'][
                'file_richness_selection_function']

        if self.external_richness_selection_function == 'CG_ESF':

            self.interp_Plambda_obs = []
            lambda_grid = 10.**(np.linspace(np.log10(5.), np.log10(100.), 100))
            z_grid = np.linspace(0.1, 1.8, 18)

            for ltab in range(len(self.lambda_obs_edges_CC) - 1):

                self.interp_Plambda_obs.append(
                    interpolate.interp2d(
                        z_grid, lambda_grid,
                        self.selection_function[ltab, :, :],
                        kind='cubic', fill_value=None))

        # Parameters of the mass-richness scaling relation
        self.A_l = self.theory['observables_specifications']['CG']['A_l']
        self.B_l = self.theory['observables_specifications']['CG']['B_l']
        self.C_l = self.theory['observables_specifications']['CG']['C_l']

        self.sig_A_l = self.theory[
            'observables_specifications']['CG']['sig_A_l']
        self.sig_B_l = self.theory[
            'observables_specifications']['CG']['sig_B_l']
        self.sig_C_l = self.theory[
            'observables_specifications']['CG']['sig_C_l']

        self.M_pivot = float(self.theory[
            'observables_specifications']['CG']['M_pivot'])
        self.z_pivot = self.theory[
            'observables_specifications']['CG']['z_pivot']

        # Parameters defining the scatter of the observed-true richness PDF
        self.sig_lambda_norm = self.theory[
            'observables_specifications']['CG']['sig_lambda_norm']
        self.sig_lambda_z = self.theory[
            'observables_specifications']['CG']['sig_lambda_z']
        self.sig_lambda_exponent = self.theory[
            'observables_specifications']['CG']['sig_lambda_exponent']

        # Parameters defining the scatter of the observed-true redshift PDF
        self.sig_z_norm = self.theory[
            'observables_specifications']['CG']['sig_z_norm']
        self.sig_z_z = self.theory[
            'observables_specifications']['CG']['sig_z_z']
        self.sig_z_lambda = self.theory[
            'observables_specifications']['CG']['sig_z_lambda']

        # Hard-coded variables optimised for the integration routines
        k_div = 250
        k_min = 1.e-3
        k_max = 2.0
        self.k = np.logspace(np.log10(k_min), np.log10(k_max), k_div)

        M_min = 12.0  # in M_sol
        M_max = 16.0
        M_div = 50
        self.Mass = np.linspace(M_min, M_max, M_div + 1)

        z_min = 1.e-5
        z_max = 2.0
        z_div = 50
        self.z = np.linspace(z_min, z_max, z_div + 1)

        lambda_min = 1.0
        lambda_max = 250.0
        lambda_div = 50
        self.Lambda = np.linspace(
            lambda_min, lambda_max, lambda_div + 1)

        self.r_interp = np.logspace(-10, 2.5, 200)

    def Pk_def(self, z, k, nu_cdm):
        r"""
        Computes the power spectrum for the clusters probe.

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the power spectrum.
        k: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum.
            Units: Mpc:math:`^{-1}`

        Returns
        -------
            float or numpy.ndarray
            Power spectrum
            at a given redshift and k-mode for galaxy clusters
            Units: Mpc:math:`^3`
        """

        if nu_cdm == 1.0:
            return (self.theory['Pk_cb'].P(z, k))
        else:
            return (self.theory['Pk_delta'].P(z, k))

    def dVdzdO(self, z):
        r"""
        Volume element per redshit per solid angle

        ..math::
           \frac{\text{d} V}{\text{d} z \text{d}\Omega} =
           \frac{c (1+z)^2 D_{\rm A}^2(z)}{H(z)}

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate the volume element

        Returns
        -------
        dVdzdO: numpy.ndarray of the volume element
                dVdzdO[i] where i is the redshift axis
                Units: Mpc:math:`^3`
        """

        return self.theory['c'] * \
            (1. + z)**2.0 * (self.theory['d_z_func']
                             (z))**2.0 / self.theory['H_z_func'](z)

    def rho_crit_0(self):
        r"""
        Critical density of the Universe at redshift = 0

        Units: Mpc:math:`^{-3}` M:math:`\odot`:math:`^{-1}`
        """

        hubble_value = self.theory['H0'] / 3.085677581491367e+19
        G_unit = G.to(units.Mpc**3.0 / (units.Msun * units.s**2)).value

        return 3.0 * hubble_value**2.0 / (8.0 * np.pi * G_unit)

    def rho_mean_0(self):
        r"""
        Mean matter density of the Universe at redshift = 0

        Units: Mpc:math:`^{-3}` M:math:`\odot`:math:`^{-1}`
        """

        if self.neutrino_cdm == 'cb':
            return ((self.theory['Omm'] -
                     self.theory['Omnu']) *
                    self.rho_crit_0())
        else:
            return (self.theory['Omm'] * self.rho_crit_0())

    def radius_M(self, M):
        r"""
        Converts the requested mass in the associated radius

        .. math::
            R(M)= 3M/(4\pi\rho_c\Omega_{\rm m})^{1/3}

        Parameters
        ----------
        M: float or numpy.ndarray
              Mass at which the radius is
              to be estimated

        Returns
        -------
        radius_M: float or numpy.ndarray
                Radius in Mpc
        """

        return (M / self.rho_mean_0() * (3.0 / (4.0 * np.pi)))**(1.0 / 3.0)

    def Omm_z(self, z, nu_cdm):
        r"""
        Computes the evolution of the matter density parameter with reshift

        Parameters
        ----------
        z: float or numpy.ndarray
                  Redshifts at which the computation parameter is
                  to be estimated
        """

        if nu_cdm == 1:
            Ommz = (self.theory['Omm'] - self.theory['Omnu']) \
                    * (1.0 + z)**3.0 \
                    * (self.theory['H0'] / self.theory['H_z_func'](z))**2.0
            return Ommz
        else:
            Ommz = self.theory['Omm'] * (1.0 + z)**3.0 \
                    * (self.theory['H0'] / self.theory['H_z_func'](z))**2.0
            return Ommz

    def delta_c(self, z):
        r"""
        Critical linear collapse overdensity following Kitayama & Suto (1999)

        ..math::
           \delta_{c, {\rm lin}}(z) \eqsim  \frac{3}{20}
           (12\pi)^{2/3}[1+0.012299 \log \Omega_{\rm m}(z)]

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the critical overdensity

        Returns
        -------
        delta_c:  float or numpy.ndarray
            Value of the critical linear collapse overdensity
            at a given redshift
        """

        return (3.0 / 20.0 * (12.0 * np.pi)**(2.0 / 3.0) *
                (1.0 + 0.012299 * np.log10(self.Omm_z(z, nu_cdm=0))))

    def sigma_z_M(self, z, M):
        r"""
        Root mean square of matter perturbation density

        Computes the root mean square of
        the matter perturbation density :math:`\sigma(z,M)`
        at a given redshift and mass

        .. math::
            \sigma^2(R,z) = \frac{1}{2\pi^2} \int_0^\infty
            \text{d} k k^2 P(k,z)_{\rm m} W(kR)^2

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate :math:`\sigma(z,M)`
        M: float or numpy.ndarray
               Mass at which to evaluate :math:`\sigma(z,M)`

        Returns
        -------
        sigma_z_M: numpy.ndarray
                sigma_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        if self.neutrino_cdm == 'cb':
            return self.theory['sigmaR_z_func_cb'](
                z, self.radius_M(M), grid=True)
        else:
            return self.theory['sigmaR_z_func'](z, self.radius_M(M), grid=True)

    def nu_z_M(self, z, M):
        r"""
        Critical collapse overdensity over mass rms

        Critical collapse overdensity over mass rms
        at a given redshift and mass

        ..math::
           \nu(M,z)=\delta_c(z)/\sigma(R,z)

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate :math:`\nu(z,M)`
        M: float or numpy.ndarray
               Mass at which to evaluate :math:`\nu(z,M)`

        Returns
        -------
        nu_z_M:   numpy.ndarray
            nu_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        return self.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def window(self, k, R):
        r"""
        Computes the top-hat window function and its derivative

        ..math::
           W(kR) = \frac{3(\sin(kR)-kR\cos(kR))}{(kR)^3}

        Parameters
        ----------
        k: float or numpy.ndarray
               Wavenumber at which to evaluate :math:`W(k R)`
               Units: Mpc:math:`^{-1}`
        R: float or numpy.ndarray
               Radius at which to evaluate :math:`W(k R)`
               Units: Mpc

        Returns
        -------
        W:   numpy.ndarray
             W[i,j] where i is the wavenumber axis and
                j the radius axis
        dWdx: numpy.ndarray
              dWdx[i,j] where i is the wavenumber axis and
                j the radius axis
        """

        x = R[:, np.newaxis] * k
        W = 3. * (np.sin(x) - x * np.cos(x)) / x**3.
        dWdx = 3. * (np.sin(x) * (x**2. - 3.) + 3. * x * np.cos(x)) / x**4.

        return W, dWdx

    def dlnsdlnR(self, z, M):
        r"""
        Derivative of the log of the rms of matter

        Derivative of the log of the rms of matter with respect
        to the radius corresponding to a given mass

        .. math::
            \frac{{\text{d} \ln \sigma(R(M),z)^{-1}}{{\text{d} R} =
            \frac{1}{\pi^2} \int \text{d} k k^3 P(k,z)_{\rm m}
            W(kR) \frac{\text{d} W(kR)}{\text{d} kR}

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate :math:`dlns\dlnR(z,M)`
        M: float or numpy.ndarray
               Mass at which to evaluate :math:`dlns\dlnR(z,M)`
               Units: M:math:`\odot`

        Returns
        -------
        dlnsdlnR: numpy.ndarray
                dlnsdlnR[i,j] where i is the redshift axis and
                j the mass axis
        """

        k = self.k
        R = self.radius_M(M)
        W, dWdx = self.window(k, R)
        if self.neutrino_cdm == 'cb':
            nu_cdm = 1
        else:
            nu_cdm = 0
        dsigma2_dR = np.pi**-2 \
            * simps(k.reshape(1, 1, len(k))**3 *
                    self.Pk_def(z, k, nu_cdm).reshape(len(z), 1, len(k)) *
                    W.reshape(1, len(R), len(k)) *
                    dWdx.reshape(1, len(R), len(k)), k, axis=-1)

        return R / (2 * self.sigma_z_M(z, M)**2) * dsigma2_dR

    def f_sigma_nu(self, z, M):
        r"""
        Multiplicity of the halo mass function following Castro et al. (2023)

        ..math::
           f(\nu,z) = A(p,q) \sqrt{\frac{2a}{\pi}} e^{-a\nu^2/2}
           \left(1+ \frac{1}{(a\nu^2)^p} \right) (\nu\sqrt{a})^{q-1}

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate :math:`\mathcal{f}(\nu(z,M))`
        M: float or numpy.ndarray
               Mass at which to evaluate :math:`\mathcal{f}(\nu(z,M))`

        Returns
        -------
        f_sigma_nu: numpy.ndarray
                f_sigma_nu[i,j] where i is the redshift axis and
                j the mass axis
        """

        a1 = 0.7962
        a2 = 0.1449
        az = -0.0658
        p1 = -0.5612
        p2 = -0.4743
        q1 = 0.3688
        q2 = -0.2804
        qz = 0.0251

        if self.neutrino_cdm == 'cb':
            nu_cdm = 1
        else:
            nu_cdm = 0
        dlnsigmadlnR = self.dlnsdlnR(z, M)
        Ommz = self.Omm_z(z, nu_cdm)[:, np.newaxis]
        nu = self.nu_z_M(z, M)

        aR = a1 + a2 * (dlnsigmadlnR + 0.6125)**2.0
        a = aR * Ommz**az
        p = p1 + p2 * (dlnsigmadlnR + 0.5)
        qR = q1 + q2 * (dlnsigmadlnR + 0.5)
        q = qR * Ommz**qz
        A = (1.0 / (2.0**(-0.5 - p + q / 2.0) / np.sqrt(np.pi) *
                    (2.0**p * gamma(q / 2.0) + gamma(-p + q / 2.0))))
        B = np.sqrt(2.0 * a / (np.pi)) * np.exp(-a * nu**2.0 / 2.0) \
            * (1.0 + 1.0 / (a * nu**2.0)**p) * (nu * np.sqrt(a))**(q - 1.0)
        return A * B * nu

    def dndm(self, z, M):
        r"""
        Derivative of the number density at a given redshift and mass

        ..math::
           \frac {\text{d} n(M,z)}{\text{d} M} =
           \frac{3}{4\pi R(M)^3} \frac{\text{d}
           \ln \sigma(R(M),z)^{-1}}{\text{d} M}
           \nu f(\nu(M,z),z)

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate :math:`\frac{d n(z,M)}{d m}`
        M: float or numpy.ndarray
               Mass at which to evaluate :math:`\frac{d n(z,M)}{d m}`
               Units: M:math:`\odot`

        Returns
        -------
        dndm: numpy.ndarray
                dndm[i,j] where i is the redshift axis and
                j the mass axis
        """

        dlnsigmadlnR = self.dlnsdlnR(z, M)

        dndm = self.rho_mean_0() / M**2.0 \
            * self.f_sigma_nu(z, M) * dlnsigmadlnR / (-3)
        return dndm

    def lnlambda(self, z, M):
        r"""
        Theoretical richness mass redshift scaling relation

        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the theoretical richness
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness
               Units: M:math:`\odot`

        Returns
        -------
        lnlambda : numpy.ndarray
                lnlambda[i,j] where i is the true redhshift axis and
                                    j the mass axis
        """

        lnlambda = np.log(self.A_l) \
            + self.B_l \
            * np.log(M / self.M_pivot) \
            + self.C_l \
            * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_pivot))
        return lnlambda

    def scatter_lnlambda(self, z, M):
        r"""
        Scatter of the theoretical richness probability distribution

        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the theoretical richness
                   scatter
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness scatter
               Units: M:math:`\odot`

        Returns
        -------
        scatter_lnlambda : float or numpy.ndarray
                scatter_lnlambda[i,j] where i is the true redhshift axis and
                                       j the mass axis
        """

        return self.sig_A_l + self.sig_B_l * np.log(
            M / self.M_pivot) + self.sig_C_l * np.log(
            (1.0 + z[:, np.newaxis]) / (1.0 + self.z_pivot))

    def Pscaling_relation(self, z, M, Lambda):
        r"""
        Probability distribution defining the richness-mass scaling relation

        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the theoretical-observed
                   richness scatter
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical-observed
               richness scatter
               Units: M:math:`\odot`
        Lambda: float or numpy.ndarray
               True richness at which to evaluate the observed
               richness

        Returns
        -------
        Pscaling_relation: float or numpy.ndarray
                Pscaling_relation[i,j,k] where i is the redshift,
                                     j is the mass,
                                     k is the observed richness
        """

        lnlambda1 = self.lnlambda(z, M)[:, :, np.newaxis]
        sigmalnl = self.scatter_lnlambda(z, M)[:, :, np.newaxis]
        Lambda = Lambda[np.newaxis, np.newaxis, :]

        return 1.0 / (Lambda * np.sqrt(2.0 * np.pi * sigmalnl**2.0)) * \
            np.exp(-(np.log(Lambda) - lnlambda1)**2.0 / (2.0 * sigmalnl**2.0))

    def scatter_Plambda_obs(self, z, Lambda):
        r"""
        Scatter of the observed-true richness probability distribution

        ..math::
           \sigma_{\lambda_{\rm ob}}=(\sigma_{\lambda_{\rm ob},0}+
           \sigma_{\lambda_{\rm ob},z}z) \lambda^{\eta}

        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the scatter of the
                   theoretical-observed richness probability distribution
        Lambda: float or numpy.ndarray
               Theoretical richness at which to evaluate the scatter of the
               theoretical-observed richness probability distribution

        Returns
        -------
        scatter_Plambda_obs: float or numpy.ndarray
                scatter_Plambda_obs[i,j] where i is the redshift axis
                                              j is the theoretical richness
                                              axis
        """

        return (self.sig_lambda_norm + self.sig_lambda_z * z[:, np.newaxis]) \
            * Lambda**self.sig_lambda_exponent

    def Plambda_obs(self, z, Lambda, Lambda_obs):
        r"""
        Observed-theoretical richness probability distribution

        Parameters
        ----------
        z: float or numpy.ndarray
               True redshift at which to evaluate the observed richness
               probability distribution
        Lambda: float or numpy.ndarray
                   Theoretical richness at which to evaluate the
                   theoretical-observed richness probability distribution
        Lambda_obs: float or numpy.ndarray
               Observed richness at which to evaluate the
               theoretical observed richness probability distribution

        Returns
        -------
        Plambda_obs: float or numpy.ndarray
                Plambda_obs[i,j,k] where i is the redshift axis,
                j is the the theoretical richness axis,
                k is the observed richness
        """

        scatter = self.scatter_Plambda_obs(z, Lambda)[:, :, np.newaxis]

        Plambda_obs = 1.0 / (np.sqrt(2.0 * np.pi * scatter**2.0)) \
            * np.exp(-(Lambda_obs[np.newaxis, np.newaxis, :] -
                       Lambda[np.newaxis, :, np.newaxis])**2.0 /
                     (2.0 * scatter**2.0))
        return Plambda_obs

    def scatter_z_obs(self, Lambda_obs, z):
        r"""
        Scatter of the observed-true redshift probability distribution

        ..math::
           \sigma_{z_{\rm ob}} = \sigma_{z_{\rm ob},0} +
           \sigma_{z_{\rm ob},z} z +
           \sigma_{z_{\rm ob},\lambda} \lambda_{\rm ob}

        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the true-observed scatter
        Lambda_obs: float or numpy.ndarray
               Observed richness at which to evaluate the true-observed scatter

        Returns
        -------
        scatter_z_obs: float or numpy.ndarray
                scatter_z_obs[i,j] where i is the true redshift axis and
                j the observed richness axis
        """

        return self.sig_z_norm + self.sig_z_z * z + \
            self.sig_z_lambda * Lambda_obs

    def Pz_obs(self, z_obs, Lambda_obs, z):
        r"""
        True-observed redshift probability distribution

        Parameters
        ----------
        z_obs: float or numpy.ndarray
               Observed redshift at which to evaluate the true-observed
               redshift probability distribution
        Lambda_obs: float or numpy.ndarray
               Observed richness at which to evaluate the true-observed
                   redshift probability distribution
        z: float or numpy.ndarray
               True redshift at which to evaluate the true-observed
               redshift probability distribution

        Returns
        -------
        Pz_obs: float or numpy.ndarray
                Pz_obs[i,j,k] where i is the observed redshift axis
                                      j is the observed richness axis
                                      k is the true redshift axis
        """

        scatter = self.scatter_z_obs(Lambda_obs, z)

        return 1.0 / (np.sqrt(2.0 * np.pi * scatter**2.0)) * \
            np.exp(-(z_obs[:, np.newaxis] - z) **
                   2.0 / (2.0 * scatter**2.0))

    def Delta(self, overdensity_type, z, nu_cdm, overdensity=200):
        r"""
        Overdensity factor.

        Parameters
        ----------
        overdensity_type: str
            The overdensity type. Possibilities are: "crit", "mean", "vir".
        z: float
            Redshift.
        overdensity: int
            The overdensity value. If a virial density is assumed, this input
            variable is not used.

        Returns
        -------
        Delta: float
            The critical overdensity factor which needs to be multiplied
            by the matter density in order to obtain
            the background overdensity.

        Notes
        -----
        The function is returned for :math:`\rm \rho` in a density definition
        at a given redshift. The function returns :math:`\rm \Delta_c` for the
        critical density of the universe, :math:`\rm \Delta_{bk}` for
        the mean matter density of the universe, :math:`\rm \Delta_{vir}`
        determined following Bryan & Norman (1998)
        """
        if overdensity_type == 'crit':
            Delta = overdensity

        elif overdensity_type == 'mean':
            Delta = overdensity * self.Omm_z(z, nu_cdm)

        elif overdensity_type == 'vir':
            x = self.Omm_z(z, nu_cdm) - 1.0
            Delta = 18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2

        else:
            raise ValueError(
                'Invalid overdensity definition, %s.' %
                overdensity_type)

        return Delta

    def halo_bias(self, z, M, Delta):
        r"""
        Computes the halo bias at a given redshift and mass

        following Tinker et al. (2010)

        ..math::
           b(M,z) = 1 + \frac{a \nu(M,z)^2 - q}{\delta_c(z)}
           + \frac{2 p/\delta_c(z)}{1+(a\nu^2)^p}

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate :math:`b_\nu(z,M)`
        M: float or numpy.ndarray
               Mass at which to evaluate :math:`b_\nu(z,M)`
        Returns
        -------
        halo_bias:   numpy.ndarray
                halo_bias[i,j] where i is the redshift axis and
                j the mass axis
        """

        p = [1.0, 0.24, 0.44, 0.88, 0.183, 1.5, 0.019, 0.107, 0.19, 2.4]
        y = np.log10(Delta)
        A_par = p[0] + p[1] * y * np.e**(-(4. / y)**4)
        a_par = p[2] * y - p[3]
        B_par = p[4]
        b_par = p[5]
        C_par = p[6] + p[7] * y + p[8] * np.e**(-(4. / y)**4)
        c_par = p[9]

        b_nu = self.nu_z_M(z, M).T
        return (1. - A_par * b_nu**a_par / (b_nu**a_par + self.delta_c(z) **
                a_par) + B_par * b_nu**b_par + C_par * b_nu**c_par).T

    def cov_window(self, zbin, ztab, kh, L):
        r"""
        Computes the window function at a given redshift and wavenumber

        ..math::
           W_i({k, z}) = \left( \left. \frac{dV}{d\Omega}
           \right|_{\substack{\Delta z\ob_i}} \right)^{-1}
           \int_{\Delta z\ob_i} dz  \frac{dV}{dz d\Omega}
           4 \pi \sum_{l=0}^{\infty} \sum_{m=-l}^{l} (i)^l j_l
           (k \chi(z)) Y_{l,m}(k) K_{l}

        Parameters
        ----------
        ztab: float or numpy.ndarray
                Redshift at which to evaluate :math:`Cov_w`
        kh: float or numpy.ndarray
               Wavenumber used to evaluate the power spectrum
               Units: Mpc:math:`^{-1}`
        L: float or numpy.ndarray
                Spherical harmonic expansion coefficients
                at which to evaluate :math:`Cov_w`
        Returns
        -------
        cluster count covariance window:   numpy.ndarray
            cov_window[i,j] where i is the redshift axis and
                j the mass axis
        """

        rvec = self.theory['d_z_func'](ztab) * (1 + ztab)    # Mpc
        Vz = (rvec[-1]**3 - rvec[0]**3) / 3   # Mpc^3
        kr = self.k[:, np.newaxis] * rvec
        self.rint[zbin] = 1 / Vz \
            * simps(rvec**2. *
                    np.array(
                             [spherical_jn(ell, kr, derivative=False)
                              for ell in range(L + 1)]), rvec, axis=-1).T
        cov_window = (4 * np.pi) * np.sum(self.rint[zbin, :, :] *
                                          self.rint[:(zbin + 1), :, :] *
                                          self.KL[:]**2, axis=-1)
        return cov_window

    def N_zbin_Lbin_Rbin(self):
        r"""
        Galaxy cluster observables theoretical outputs

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate
                   Cluster counts, shear, and clustering
        l: float or numpy.ndarray
               Richness at which to evaluate
               Cluster counts, shear, and clustering
        R: float or numpy.ndarray
               Radius at which to evaluate shear, and clustering
        Returns
        -------
        Cluster counts, shear, and clustering
        """

        CG_like_selection = self.theory[
            'observables_specifications']['CG']['CG_probe']
        CG_xi2_cov_selection = self.theory[
            'observables_specifications']['CG']['CG_xi2_cov_selection']
        external_richness_selection_function = self.theory[
            'observables_specifications']['CG'][
                'external_richness_selection_function']

        # array initialization for number counts NC and mass observable
        # scaling relations
        N_zbin_Lbin = np.zeros((self.z_obs_div_CC, self.lambda_obs_div_CC))
        cov_zbin_Lbin = np.zeros(
            (self.z_obs_div_CC,
             self.z_obs_div_CC,
             self.lambda_obs_div_CC,
             self.lambda_obs_div_CC))
        n_lbdobs_z = np.empty(self.lambda_obs_div_CC, dtype=list)
        Plob_M_z = np.empty(self.lambda_obs_div_CC, dtype=list)
        b_n_lbdobs_z = np.empty(self.lambda_obs_div_CC, dtype=list)
        dzobs_dz = np.empty(
            (self.z_obs_div_CC, self.lambda_obs_div_CC), dtype=list)

        if CG_like_selection in ['CC', 'CC_CWL', 'CC_Cxi2', 'CC_CWL_Cxi2']:

            # Starting of galaxy cluster number counts computation
            z_obs_mid = 0.5 * \
                (self.z_obs_edges_CC[1:] + self.z_obs_edges_CC[:-1])
            z_mid = 0.5 * (self.z[1:] + self.z[:-1])
            lambda_obs_mid = 0.5 * \
                (self.lambda_obs_edges_CC[1:] + self.lambda_obs_edges_CC[:-1])
            lambda_mid = 0.5 * (self.Lambda[1:] + self.Lambda[:-1])

            # calculation of the volume element for the redshift bins
            dvdzdomega_z1z2 = self.dVdzdO(z_mid)
            Delta_bkg = self.Delta(
                self.overdensity_type, z_mid, 0, self.overdensity) \
                / self.Omm_z(z_mid, nu_cdm=0)
            # calculation of the mass function for the mass and redshift bins
            dndm_z = self.dndm(z_mid, 10.0**self.Mass)
            # calculation of the halo bias for the mass and redshift bins
            bias_z = self.halo_bias(z_mid, 10.0**self.Mass, Delta_bkg)

            if CG_xi2_cov_selection in ['covCC', 'covCC_covCxi2']:

                # calculation of the spherical harmonic expansion
                # coefficients for the NC covariance matrix
                L = 20
                ell = np.linspace(0.0, L, L + 1)
                theta = np.arccos(
                    1 - (self.area * (np.pi / 180.0)**2.0) / (2 * np.pi))
                self.KL = np.sqrt(np.pi / (2. * ell + 1.)) * (
                    eval_legendre(ell - 1, np.cos(theta)) -
                    eval_legendre(ell + 1, np.cos(theta))) \
                    / (2. * np.pi * (1 - np.cos(theta)))
                self.KL[0] = 1 / (2. * np.sqrt(np.pi))
                self.rint = np.zeros((self.z_obs_div_CC, len(self.k), L + 1))

                # array initialization for the NC covariance matrix diagonal
                # and off diagonal terms
                pk = self.Pk_def(z_obs_mid, self.k, nu_cdm=0)
                sab1 = np.zeros((self.z_obs_div_CC, self.z_obs_div_CC))
                SN = np.zeros(
                    (self.z_obs_div_CC,
                     self.z_obs_div_CC,
                     self.lambda_obs_div_CC,
                     self.lambda_obs_div_CC))
                Nb_zbin_Lbin = np.zeros(
                    (self.z_obs_div_CC, self.lambda_obs_div_CC))

                for z_bin in range(len(self.z_obs_edges_CC) - 1):

                    z_tab = np.linspace(
                        self.z_obs_edges_CC[z_bin],
                        self.z_obs_edges_CC[z_bin + 1],
                        self.z_tab_sig)

                    sab1[z_bin, :(z_bin + 1)] = 1 / (2 * np.pi**2) \
                        * simps((self.k**2 * np.sqrt(
                            pk[z_bin] * pk[:(z_bin + 1)])) *
                        self.cov_window(z_bin, z_tab, self.k, L),
                        self.k, axis=-1)

            for lambda_bin in range(len(self.lambda_obs_edges_CC) - 1):

                l_tab = np.geomspace(
                    self.lambda_obs_edges_CC[lambda_bin],
                    self.lambda_obs_edges_CC[lambda_bin + 1],
                    self.l_m_tab_sig[lambda_bin])

                # calculation of the Probability distribution
                # relating richness observed to true richness
                Plob_l_z = simps(
                    self.Plambda_obs(
                        z_mid,
                        lambda_mid,
                        l_tab),
                    l_tab,
                    axis=-
                    1)

                if external_richness_selection_function == 'CG_ESF':
                    Plob_l_z = self.interp_Plambda_obs[lambda_bin](
                        z_mid, lambda_mid).T

                # calculation of the Probability distribution relating
                # richness to mass and redshift
                Pl_M_z = self.Pscaling_relation(
                    z_mid,
                    10.0**self.Mass,
                    lambda_mid)
                Plob_M_z[lambda_bin] = simps(
                    Pl_M_z[:, :, :] * Plob_l_z[:, np.newaxis, :],
                    lambda_mid, axis=-1)

                # galaxy cluster number density from convolving
                # halo mass function and total selection function
                # probabilty density
                n_lbdobs_z[lambda_bin] = simps(
                    Plob_M_z[lambda_bin] * dndm_z,
                    10.0**self.Mass,
                    axis=1)

                # galaxy cluster number density from convolving
                # halo mass function and total selection function
                # probabilty density times the halo bias as an
                # intermediate quantity for samplning variance
                # needed for the covariance matrix
                if CG_xi2_cov_selection in ['covCC', 'covCC_covCxi2']:
                    b_n_lbdobs_z[lambda_bin] = simps(
                        Plob_M_z[lambda_bin] * dndm_z * bias_z,
                        10.0**self.Mass,
                        axis=1)

                for z_bin in range(len(self.z_obs_edges_CC) - 1):

                    z_tab = np.linspace(
                        self.z_obs_edges_CC[z_bin],
                        self.z_obs_edges_CC[z_bin + 1],
                        self.z_tab_sig)

                    # calculation of the Probability distribution
                    # relating redshift observed to true redshift
                    dzobs_dz[z_bin, lambda_bin] = simps(self.Pz_obs(
                            z_tab, self.lambda_obs_edges_CC[lambda_bin],
                            z_mid),
                        z_tab, axis=0)

                    # calculation of the volume in redshift bins
                    dVz = dvdzdomega_z1z2 * \
                        dzobs_dz[z_bin, lambda_bin] * \
                        (self.area) * (np.pi**2.0 / 180.0**2.0)

                    # cluster number counts from number density * volume
                    N_zbin_Lbin[z_bin, lambda_bin] = simps(
                        n_lbdobs_z[lambda_bin] * dVz, z_mid, axis=0)

                    if CG_xi2_cov_selection in ['covCC', 'covCC_covCxi2']:

                        # sample variance terms in the covariance matrix
                        Nb_zbin_Lbin[z_bin, lambda_bin] = simps(
                            b_n_lbdobs_z[lambda_bin] * dVz, z_mid)

                        # shot-noise for the diagonal terms in
                        # the covariance matrix
                        SN[z_bin, z_bin] = np.diag(N_zbin_Lbin[z_bin])

            if CG_xi2_cov_selection in ['covCC', 'covCC_covCxi2']:
                sab = sab1 + sab1.T - np.diag(np.diag(sab1))

                # total final covariance = shot-noise terms +
                #                          sample variance terms
                cov_zbin_Lbin = SN \
                    + (Nb_zbin_Lbin.reshape
                        (1, self.z_obs_div_CC, 1, self.lambda_obs_div_CC) *
                        Nb_zbin_Lbin.reshape(self.z_obs_div_CC, 1,
                                             self.lambda_obs_div_CC, 1) *
                        sab.reshape(
                            self.z_obs_div_CC, self.z_obs_div_CC, 1, 1))

        g_zbin_Lbin_Rbin = np.zeros(
            ((self.z_obs_div_CC, self.lambda_obs_div_CC, 1)))

        if CG_like_selection in ['CC_CWL', 'CC_CWL_Cxi2']:

            raise ValueError("The galaxy cluster weak-lensing likelihood is \
                            not implemented yet!")

        xi2_zbin_Lbin_Rbin = np.zeros((((1, 1, 1))))
        cov_Cxi2 = np.ones((1, 1, 1, 1, 1, 1))

        # Starting of the galaxy cluster lensing shear profile computation

        if CG_like_selection in ['CC_Cxi2', 'CC_CWL_Cxi2']:

            raise ValueError("The galaxy cluster 2pt correlation function \
                            likelihood is not implemented yet!")

        # Starting of the galaxy clusters clustering computation

        if CG_like_selection not in ['CC', 'CC_CWL', 'CC_Cxi2', 'CC_CWL_Cxi2']:

            raise ValueError(f"Wrong declaration (\"{CG_like_selection}\") of\
                              the galaxy cluster probe")
        # this will break the code

        return N_zbin_Lbin, g_zbin_Lbin_Rbin, xi2_zbin_Lbin_Rbin, \
            cov_zbin_Lbin, cov_Cxi2
