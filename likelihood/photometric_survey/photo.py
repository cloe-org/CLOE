"""PHOTOMETRIC MODULE
"""

# General imports
import numpy as np
from scipy import integrate
from likelihood.photometric_survey.redshift_distribution \
    import RedshiftDistribution

# General error class


class PhotoError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class Photo:
    """
    Class for photometric observables
    """

    def __init__(self, cosmo_dic, nz_dic_WL, nz_dic_GC):
        """Initialize

        Constructor of the class Photo

        Parameters
        ----------
        cosmo_dic: dict
           cosmological dictionary from cosmo
        nz_dic_WL: dict
            Dictionary containing n(z)s for WL probe.
        nz_dic_GC: dict
            Dictionary containing n(z)s for GC-phot probe.
        """
        self.theory = cosmo_dic
        nuisance_dict = self.theory['nuisance_parameters']
        self.nz_GC = RedshiftDistribution('GCphot', nz_dic_GC, nuisance_dict)
        self.nz_WL = RedshiftDistribution('WL', nz_dic_WL, nuisance_dict)
        self.multbias = [nuisance_dict[f'multiplicative_bias_{i}'] for i in
                         sorted(
                         [int(k.replace('multiplicative_bias_', '')) for k in
                          nuisance_dict.keys()
                          if k.startswith('multiplicative_bias_')])]
        if self.theory['f_K_z_func'] is None:
            raise KeyError('No interpolated function for transverse comoving '
                           'distance exists in cosmo_dic.')
        # temporary fix, see #767
        if self.theory['CAMBdata'] is None:
            raise KeyError('CAMBdata is not available in cosmo_dic.')
        self.cl_int_z_min = 0.001
        self.cl_int_z_max = self.theory['z_win'][-1]
        # The size of z_winterp sufficient for now, could be tuned later
        z_wlogmin = -2
        z_wmin1 = 1e-5
        z_wmin2 = 1e-4
        z_wmin3 = 1e-3
        z_wmax = 4.0
        z_wsamp = 1000
        self.z_trapz_sampling = 500
        self.z_winterp = np.logspace(z_wlogmin, np.log10(z_wmax), z_wsamp)
        self.z_winterp[0] = z_wmin1
        self.z_winterp[1] = z_wmin2
        self.z_winterp[2] = z_wmin3
        # Number of bins should be generalized, hard-coded for now

        # z_wtom is the number of tomographic bins + 1
        z_wtom_wl = 1 + self.nz_WL.get_num_tomographic_bins()
        z_wtom_gc = 1 + self.nz_GC.get_num_tomographic_bins()
        self.interpwin = np.zeros(shape=(z_wsamp, z_wtom_wl))
        self.interpwingal = np.zeros(shape=(z_wsamp, z_wtom_gc))
        self.interpwinia = np.zeros(shape=(z_wsamp, z_wtom_wl))
        self.interpwin[:, 0] = self.z_winterp
        self.interpwingal[:, 0] = self.z_winterp
        self.interpwinia[:, 0] = self.z_winterp

        for tomi in range(1, z_wtom_wl):
            self.interpwin[:, tomi] = self.WL_window(tomi)
            self.interpwinia[:, tomi] = self.IA_window(self.z_winterp, tomi)
        for tomi in range(1, z_wtom_gc):
            self.interpwingal[:, tomi] = self.GC_window(self.z_winterp, tomi)

    def GC_window(self, z, bin_i):
        r"""GC Window

        Implements the galaxy clustering photometric window function.

        .. math::
            W_i^G(z) &= \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\

        Parameters
        ----------
        z: numpy.ndarray of float or float
           Redshift at which to evaluate distribution.
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        W_i_G: float
           Window function for galaxy clustering photometric
        """

        n_z_normalized = self.nz_GC.evaluates_n_i_z(bin_i, z)
        W_i_G = n_z_normalized * self.theory['H_z_func_Mpc'](z)

        return W_i_G

    def WL_window_integrand(self, zprime, z, nz):
        r"""WL Window Integrand

        Calculates the Weak-lensing (WL) kernel integrand as

        .. math::
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_{K}\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}
            }

        Parameters
        ----------
        zprime: float
            redshift parameter that will be integrated over.
        z: float
            Redshift at which kernel is being evaluated.
        nz: InterpolatedUnivariateSpline, or function
            Galaxy distribution function for the tomographic bin for
            which the kernel is currently being evaluated.

        Returns
        -------
        wint: float
           Weak-lensing kernel integrand
        """
        # temporary fix, see #767
        wint = (
            nz(zprime) *
            self.theory['CAMBdata'].angular_diameter_distance2(zprime, z) /
            self.theory['CAMBdata'].angular_diameter_distance2(zprime, 0)
        )
        return wint

    def WL_window(self, bin_i, k=0.0001):
        r"""WL Window

        Calculates the weak lensing shear kernel for a given tomographic bin.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        integrate.trapz on the array along one axis.

        .. math::
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z) \Sigma(z, k)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\

        Parameters
        ----------
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.
        k: float
            k-mode at which to evaluate the Modified Gravity
            :math:`\Sigma(z,k)` function

        Returns
        -------
        W_val: numpy.ndarray
           1-D Numpy array of shear kernel values for specified bin
           at specified scale for the redshifts defined in self.z_winterp
        """
        zint_mat = np.linspace(self.z_winterp, self.z_winterp[-1],
                               self.z_trapz_sampling)
        zint_mat = zint_mat.T
        diffz = np.diff(zint_mat)
        H0_Mpc = self.theory['H0_Mpc']
        O_m = self.theory['Omm']

        n_z_normalized = self.nz_WL.interpolates_n_i(bin_i, self.z_winterp)

        intg_mat = np.array([self.WL_window_integrand(zint_mat[zii],
                                                      zint_mat[zii, 0],
                                                      n_z_normalized)
                             for zii in range(len(zint_mat))])

        integral_arr = integrate.trapz(intg_mat, dx=diffz, axis=1)

        W_val = (1.5 * H0_Mpc * O_m * (1.0 + self.z_winterp) *
                 self.theory['MG_sigma'](self.z_winterp, k) *
                 (self.theory['f_K_z_func'](self.z_winterp) /
                  (1 / H0_Mpc)) * integral_arr)

        return W_val

    def WL_window_slow(self, z, bin_i, k=0.0001):
        r"""WL Window Slow

        Calculates the weak lensing shear kernel for a given tomographic bin.

        .. math::
            W_{i}^{\gamma}(z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z) \Sigma(z, k)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated.
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.
        k: float
            k-mode at which to evaluate the Modified Gravity
            :math:`\Sigma(z,k)` function

        Returns
        -------
        W_val: float
           Value of shear kernel for specified bin at specified redshift
           and scale.
        """
        H0_Mpc = self.theory['H0_Mpc']
        O_m = self.theory['Omm']

        n_z_normalized = self.nz_WL.interpolates_n_i(bin_i, self.z_winterp)

        W_val = ((1.5 * H0_Mpc * O_m * (1.0 + z) *
                  self.theory['MG_sigma'](z, k) * (
                  self.theory['f_K_z_func'](z) /
                  (1 / H0_Mpc)) * integrate.quad(self.WL_window_integrand,
                                                 a=z, b=self.cl_int_z_max,
                                                 args=(z, n_z_normalized))[0]))
        return W_val

    def IA_window(self, z, bin_i):
        r"""IA Window

        Calculates the intrinsic alignment (IA) weight function for a
        given tomographic bin

        .. math::
            W_{i}^{\rm IA}(z) = \frac{n_i^{\rm L}(z)}{\bar{n}_i^{\rm L}}\
            \frac{H(z)}{c}\\

        Parameters
        ----------
        z: numpy.ndarray of float or float
            Redshift at which weight is evaluated.
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        W_IA: float
           Value of IA kernel for specified bin at specified redshift.
        """

        n_z_normalized = self.nz_WL.evaluates_n_i_z(bin_i, z)

        W_IA = n_z_normalized * self.theory['H_z_func_Mpc'](z)

        return W_IA

    def Cl_generic_integrand(self, z, PandW_i_j_z_k):
        r"""Cl Generic Integrand

        Calculates the angular power spectrum integrand
        for any two probes and tomographic bins for which
        the bins are supplied. The power
        spectrum is either that of
        :math:`\delta \delta`, gg, or :math:`{\rm g}\delta`,
        with {A, B} in {G, L}

        .. math::
            \frac{W_{i}^{\rm A}(z)W_{j}^{\rm B}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}\
            P_{\rm AB}\left(k_{\ell}=\
            \frac{{\rm\ell} + 1/2}
            {f_K\left[\tilde{r}(z)\right]}, z\right)\\

        Parameters
        ----------
        z: numpy.ndarray
            List of redshifts at which integrand is being evaluated.
        PandW_i_j_z_k: numpy.ndarray
           Values of the product of kernel for bin i, kernel for bin j,
           and the power spectrum at redshift z and scale k.

        Returns
        -------
        kern_mult_power: numpy.ndarray
           Values of the angular power spectrum integrand at
           the given redshifts and multipole :math:`\ell`.
        """
        kern_mult_power = (PandW_i_j_z_k /
                           (self.theory['H_z_func'](z) *
                            (self.theory['f_K_z_func'](z)) ** 2.0))

        if np.isnan(PandW_i_j_z_k).any():
            raise Exception('Requested k, z values are outside of power'
                            ' spectrum interpolation range.')
        return kern_mult_power

    def Cl_WL_noprefac(self, ell, bin_i, bin_j, int_step=0.01):
        r"""Cl WL

        Calculates angular power spectrum for weak lensing,
        for the supplied bins. Includes intrinsic alignments.

        .. math::
            C_{ij}^{\rm LL, no prefac}(\ell)= c \int \frac{dz}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}\
            \bigg\lbrace W_{i}^{\rm \gamma}\left[ k_{\ell}(z), z \right]\
            W_{j}^{\rm \gamma}\left[ k_{\ell}(z), z \right ]\
            P_{\rm \delta \delta}\left[ k_{\ell}(z), z \right] +\\
            \left[ W_{i}^{\rm IA}(z)W_{j}^{\rm \gamma}\
            \left[ k_{\ell}(z), z \right ]+W_{i}^{\rm \gamma}\
            \left[ k_{\ell}(z), z \right]W_{j}^{\rm IA}(z)\right]\
            P_{\rm \delta I}\left[ k_{\ell}(z), z \right]+\\
            W_{i}^{\rm IA}(z)W_{j}^{\rm IA}(z)\
            P_{\rm II}\left[k_{\ell}(z), z\right] \bigg\rbrace \\

        Parameters
        ----------
        ell: float
            :math:`\ell`-mode at which :math:`C_{\ell}` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           Index of second tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        c_final: float
           Value of the angular shear power spectrum.
        """

        zs_arr = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        P_dd = self.theory['Pmm_phot']
        P_ii = self.theory['Pii']
        P_di = self.theory['Pdeltai']

        ks_arr = (ell + 0.5) / self.theory['f_K_z_func'](zs_arr)
        pow_dd = P_dd(zs_arr, ks_arr, grid=False)
        pow_ii = P_ii(zs_arr, ks_arr, grid=False)
        pow_di = P_di(zs_arr, ks_arr, grid=False)

        kern_i = np.interp(zs_arr, self.interpwin[:, 0],
                           self.interpwin[:, bin_i])
        kern_j = np.interp(zs_arr, self.interpwin[:, 0],
                           self.interpwin[:, bin_j])
        kernia_i = np.interp(zs_arr, self.interpwinia[:, 0],
                             self.interpwinia[:, bin_i])
        kernia_j = np.interp(zs_arr, self.interpwinia[:, 0],
                             self.interpwinia[:, bin_j])

        pandw_dd = kern_i * kern_j * pow_dd
        pandw_ii = kernia_i * kernia_j * pow_ii
        pandw_di = (kern_i * kernia_j + kernia_i * kern_j) * pow_di
        pandwijk = pandw_dd + pandw_ii + pandw_di

        c_int_arr = self.Cl_generic_integrand(zs_arr, pandwijk)
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, zs_arr)
        c_final = c_final * (1 + self.multbias[bin_i - 1]) * \
            (1 + self.multbias[bin_j - 1])
        return c_final

    def Cl_GC_phot(self, ell, bin_i, bin_j, int_step=0.05):
        r"""Cl GC Phot

        Calculates angular power spectrum for photometric galaxy clustering,
        for the supplied bins.

        .. math::
            C_{ij}^{\rm GG}(\ell) = c \int {\rm d}z
            \frac{W_{i}^{\rm G}(z)W_{j}^{\rm G}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}\
            P^{\rm{photo}}_{\rm gg}\
            \left[ k_{\ell}(z), z \right]\\

        Parameters
        ----------
        ell: float
            :math:`\ell`-mode at which :math:`C_{\ell}` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           Index of second tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        c_final: float
           Value of angular power spectrum for
           galaxy clustering photometric.
        """

        zs_arr = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        P_gg = self.theory['Pgg_phot']

        ks_arr = (ell + 0.5) / self.theory['f_K_z_func'](zs_arr)
        power = P_gg(zs_arr, ks_arr, grid=False)

        kern_i = np.interp(zs_arr, self.interpwingal[:, 0],
                           self.interpwingal[:, bin_i])
        kern_j = np.interp(zs_arr, self.interpwingal[:, 0],
                           self.interpwingal[:, bin_j])
        pandwijk = kern_i * kern_j * power

        c_int_arr = self.Cl_generic_integrand(zs_arr, pandwijk)
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, zs_arr)

        return c_final

    def Cl_cross_noprefac(self, ell, bin_i, bin_j, int_step=0.02):
        r"""Cl Cross

        Calculates angular power spectrum for cross-correlation
        between weak lensing and galaxy clustering, for the supplied bins.
        Includes intrinsic alignments.

        .. math::
            C_{ij}^{\rm LG, no prefac}(\ell) = c \int \frac{dz}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}\
            \bigg\lbrace W_{i}^{\gamma}\left[ k_{\ell}(z), z \right ]\
            W_{j}^{\rm{G}}(z)P^{\rm{photo}}_{\rm g\delta}\
            \left[ k_{\ell}(z), z \right]\\
            +\,W_{i}^{\rm{IA}}(z)W_{j}^{\rm{G}}(z)P^{\rm{photo}}_{\rm g\rm{I}}\
            \left[ k_{\ell}(z), z \right] \bigg\rbrace \\

        Parameters
        ----------
        ell: float
            :math:`\ell`-mode at which :math:`C_{\ell}` is evaluated.
        bin_i: int
           Index of source tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           Index of lens tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        c_final: float
           Value of cross-correlation between weak lensing and
           galaxy clustering photometric angular power spectrum.
        """

        zs_arr = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        P_gd = self.theory['Pgdelta_phot']
        P_gi = self.theory['Pgi_phot']

        ks_arr = (ell + 0.5) / self.theory['f_K_z_func'](zs_arr)
        pow_gd = P_gd(zs_arr, ks_arr, grid=False)
        pow_gi = P_gi(zs_arr, ks_arr, grid=False)

        kern_i = np.interp(zs_arr, self.interpwin[:, 0],
                           self.interpwin[:, bin_i])
        kernia_i = np.interp(zs_arr, self.interpwinia[:, 0],
                             self.interpwinia[:, bin_i])
        kern_j = np.interp(zs_arr, self.interpwingal[:, 0],
                           self.interpwingal[:, bin_j])
        pandw_gd = kern_i * kern_j * pow_gd
        pandw_gi = kernia_i * kern_j * pow_gi
        pandwijk = pandw_gd + pandw_gi

        c_int_arr = self.Cl_generic_integrand(zs_arr, pandwijk)
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, zs_arr)
        c_final = c_final * (1 + self.multbias[bin_i - 1])

        return c_final

    def Cl_WL(self, ell, bin_i, bin_j, int_step=0.01):
        r"""Cl WL

        Calculates angular power spectrum for weak lensing,
        for the supplied bins. Includes intrinsic alignments.
        Includes prefactor for extended Limber approximation and curved sky.

        .. math::
            C_{ij}^{\rm LL}(\ell)= (\ell+2)(\ell+1)\ell(\ell-1)\
            (\ell+1/2)^{-4} C_{ij}^{\rm LL, no prefac}(\ell) \\

        Parameters
        ----------
        ell: float
            :math:`\ell`-mode at which :math:`C_{\ell}` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           Index of second tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        c_final: float
           Value of the angular shear power spectrum.
        """

        c_final = self.Cl_WL_noprefac(ell, bin_i, bin_j, int_step)
        c_final *= self.prefactor(ell)**2

        return c_final

    def Cl_cross(self, ell, bin_i, bin_j, int_step=0.02):
        r"""Cl Cross

        Calculates angular power spectrum for cross-correlation
        between weak lensing and galaxy clustering, for the supplied bins.
        Includes intrinsic alignments. Includes prefactor for extended
        Limber approximation and curved sky.

        .. math::
            C_{ij}^{\rm LG}(\ell) = \sqrt{(\ell+2)(\ell+1)\ell(\ell-1)}\
            (\ell+1/2)^{-2}C_{ij}^{\rm LG, no prefac}(\ell) \\

        Parameters
        ----------
        ell: float
            :math:`\ell`-mode at which :math:`C_{\ell}` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           Index of second tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        c_final: float
           Value of cross-correlation between weak lensing and
           galaxy clustering photometric angular power spectrum.
        """

        c_final = self.Cl_cross_noprefac(ell, bin_i, bin_j, int_step)
        c_final *= self.prefactor(ell)

        return c_final

    def prefactor(self, ell):
        r"""Prefactor for photometric probes

        Calculates the prefactors for extended Limber and curved sky.

        .. math::
            \sqrt{(\ell+2)(\ell+1)\ell(\ell-1)}(\ell+1/2)^{-2} \\

        Parameters
        ----------
        ell: float
            :math:`\ell`-mode at which the prefactor is evaluated.

        Returns
        -------
        prefactor: float
           Value of the prefactor at the given :math:`\ell`.
        """
        prefactor = np.sqrt((ell + 2.) * (ell + 1.) * ell * (ell - 1.)) / \
            (ell + 0.5)**2
        return prefactor
