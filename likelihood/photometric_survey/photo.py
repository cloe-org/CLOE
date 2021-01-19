# General imports
import numpy as np
import likelihood.cosmo
from scipy import integrate
from scipy import interpolate
import os.path

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
        """
        Constructor of the class Photo

        Parameters
        ----------
        cosmo_dic: dictionary
           cosmological dictionary from cosmo
        nz_dic_WL: dictionary
            Dictionary containing n(z)s for WL probe.
        nz_dic_GC: dictionary
            Dictionary containing n(z)s for GC-phot probe.
        """
        self.theory = cosmo_dic
        self.nz_dic_WL = nz_dic_WL
        self.nz_dic_GC = nz_dic_GC
        if self.theory['r_z_func'] is None:
            raise Exception('No interpolated function for comoving distance '
                            'exists in cosmo_dic.')
        self.cl_int_z_min = 0.001
        self.cl_int_z_max = self.theory['z_win'][-1]

    def GC_window(self, z, bin_i):
        r"""
        Implements the galaxy clustering photometric window function.

        .. math::
            W_i^G(z) &= \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        W_i_G: float
           Window function for galaxy clustering photometric
        """

        n_z_normalized = self.nz_dic_GC[''.join(['n', str(bin_i)])]

        W_i_G = (n_z_normalized(z) * self.theory['H_z_func_Mpc'](z))

        return W_i_G

    def WL_window_integrand(self, zprime, z, nz):
        r"""
        Calculates the Weak-lensing (WL) kernel integrand as

        .. math::
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \left [ 1 - \frac{\tilde{r}(z)}{\tilde{r}(z^{\prime})} \right ]}

        Parameters
        ----------
        zprime: float
            redshift parameter that will be integrated over.
        z: float
            Redshift at which kernel is being evaluated.
        nz: function
            Galaxy distribution function for the tomographic bin for
            which the kernel is currently being evaluated.

        Returns
        -------
        wint: float
           Weak-lensing kernel integrand
        """
        wint = nz(zprime) * (1.0 - (self.theory['r_z_func'](z) /
                                    self.theory['r_z_func'](zprime)))
        return wint

    def WL_window(self, z, bin_i, k=0.0001):
        r"""
        Calculates the weak lensing shear kernel for a given tomographic bin.

        .. math::
            W_{i}^{\gamma}(z, k) = \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z) \Sigma(z, k) r(z)
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \left [ 1 -\frac{\tilde{r}(z)}{\tilde{r}(z^{\prime})} \right ]}\\

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

        n_z_normalized = self.nz_dic_WL[''.join(['n', str(bin_i)])]

        W_val = ((1.5 * H0_Mpc * O_m * (1.0 + z) *
                  self.theory['MG_sigma'](z, k) * (
                  self.theory['r_z_func'](z) /
                  (1 / H0_Mpc)) * integrate.quad(self.WL_window_integrand,
                                                 a=z, b=self.cl_int_z_max,
                                                 args=(z, n_z_normalized))[0]))
        return W_val

    def IA_window(self, z, bin_i):
        r"""
        Calculates the intrinsic alignment (IA) weight function for a
        given tomographic bin

        .. math::
            W_{i}^{\rm IA}(z) = \frac{n_i^{\rm L}(z)}{\bar{n}_i^{\rm L}}\
            \frac{H(z)}{c}\\

        Parameters
        ----------
        z: float
            Redshift at which weight is evaluated.
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        W_IA: float
           Value of IA kernel for specified bin at specified redshift.
        """

        n_z_normalized = self.nz_dic_WL[''.join(['n', str(bin_i)])]

        W_IA = (n_z_normalized(z) * self.theory['H_z_func_Mpc'](z))

        return W_IA

    def Cl_generic_integrand(self, z, PandW_i_j_z_k):
        r"""
        Calculates the angular power spectrum integrand
        for any two probes and tomographic bins for which
        the bins are supplied. The power
        spectrum is either that of
        :math:`\delta \delta`, gg, or :math:`{\rm g}\delta`,
        with {A, B} in {G, L}

        .. math::
            \frac{W_{i}^{\rm A}(z)W_{j}^{\rm B}(z)}{H(z)r^2(z)}\
            P_{\rm AB}\left(k_{\ell}=\frac{{\rm\ell} + 1/2}{r(z)}, z\right)\\

        Parameters
        ----------
        z: float
            Redshift at which integrand is being evaluated.
        PandW_i_j_z_k: float
           Value of the product of kernel for bin i, kernel for bin j,
           and the power spectrum at redshift z and scale k.

        Returns
        -------
        kern_mult_power: float
           Value of the angular power spectrum integrand at
           a given redshift and multipole :math:`\ell`.
        """
        kern_mult_power = (PandW_i_j_z_k /
                           (self.theory['H_z_func'](z) *
                            (self.theory['r_z_func'](z)) ** 2.0))

        if np.isnan(PandW_i_j_z_k):
            raise Exception('Requested k, z values are outside of power'
                            ' spectrum interpolation range.')
        return kern_mult_power

    def Cl_WL(self, ell, bin_i, bin_j, int_step=0.1):
        r"""
        Calculates angular power spectrum for weak lensing,
        for the supplied bins. Includes intrinsic alignments.

        .. math::
            C_{ij}^{\rm LL}(\ell)= c \int \frac{dz}{H(z)r^2(z)}\
            \left\lbrace W_{i}^{\rm \gamma}\left[ k_{\ell}(z), z \right]\
            W_{j}^{\rm \gamma}\left[ k_{\ell}(z), z \right ]\
            P_{\rm \delta \delta}\left[ k_{\ell}(z), z \right] +\\
            \left[ W_{i}^{\rm IA}(z)W_{j}^{\rm \gamma}\
            \left[ k_{\ell}(z), z \right ]+W_{i}^{\rm \gamma}\
            \left[ k_{\ell}(z), z \right]W_{j}^{\rm IA}(z)\right]\
            P_{\rm \delta I}\left[ k_{\ell}(z), z \right]+\\
            W_{i}^{\rm IA}(z)W_{j}^{\rm IA}(z)\
            P_{\rm II}\left[k_{\ell}(z), z\right] \right\rbrace \\

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

        int_zs = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        c_int_arr = np.empty(len(int_zs))
        P_dd = self.theory['Pk_interpolator'].P
        P_ii = self.theory['Pii']
        P_di = self.theory['Pdeltai']
        for ii, rshift in enumerate(int_zs):
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshift)
            pow_dd = np.atleast_1d(P_dd(rshift, current_k))[0]
            pow_ii = np.atleast_1d(P_ii(rshift, current_k))[0]
            pow_di = np.atleast_1d(P_di(rshift, current_k))[0]
            kern_i = self.WL_window(rshift, bin_i, current_k)
            kern_j = self.WL_window(rshift, bin_j, current_k)
            kernia_i = self.IA_window(rshift, bin_i)
            kernia_j = self.IA_window(rshift, bin_j)
            pandw_dd = kern_i * kern_j * pow_dd
            pandw_ii = kernia_i * kernia_j * pow_ii
            pandw_di = (kern_i * kernia_j + kernia_i * kern_j) * pow_di
            pandwijk = pandw_dd + pandw_ii + pandw_di
            c_int_arr[ii] = self.Cl_generic_integrand(rshift, pandwijk)
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_GC_phot(self, ell, bin_i, bin_j, int_step=0.1):
        r"""
        Calculates angular power spectrum for photometric galaxy clustering,
        for the supplied bins.

        .. math::
            C_{ij}^{\rm GG}(\ell) = c \int {\rm d}z
            \frac{W_{i}^{\rm G}(z)W_{j}^{\rm G}(z)}{H(z)r^2(z)}\
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

        int_zs = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        c_int_arr = np.empty(len(int_zs))
        P_int = self.theory['Pgg_phot']
        for ii, rshift in enumerate(int_zs):
            # (ACD): Although k is specified here for the GC window function,
            # note that the implementation currently uses a scale-independent
            # bias.
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshift)
            power = np.atleast_1d(P_int(rshift, current_k))[0]
            kern_i = self.GC_window(rshift, bin_i)
            kern_j = self.GC_window(rshift, bin_j)
            pandwijk = kern_i * kern_j * power
            c_int_arr[ii] = self.Cl_generic_integrand(rshift, pandwijk)
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_cross(self, ell, bin_i, bin_j, int_step=0.1):
        r"""
        Calculates angular power spectrum for cross-correlation
        between weak lensing and galaxy clustering, for the supplied bins.
        Includes intrinsic alignments.

        .. math::
            C_{ij}^{\rm LG}(\ell) = c \int \frac{dz}{H(z)r^2(z)}\
            \left\lbrace W_{i}^{\gamma}\left[ k_{\ell}(z), z \right ]\
            W_{j}^{\rm{G}}(z)P^{\rm{photo}}_{\rm g\delta}\
            \left[ k_{\ell}(z), z \right]\\
            +\,W_{i}^{\rm{IA}}(z)W_{j}^{\rm{G}}(z)P^{\rm{photo}}_{\rm g\rm{I}}\
            \left[ k_{\ell}(z), z \right] \right\rbrace \\

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

        int_zs = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        c_int_arr = np.empty(len(int_zs))
        P_gd = self.theory['Pgdelta_phot']
        P_gi = self.theory['Pgi_phot']
        for ii, rshift in enumerate(int_zs):
            # (ACD): Although k is specified here for the GC window function,
            # note that the implementation currently uses a scale-independent
            # bias.
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshift)
            pow_gd = np.atleast_1d(P_gd(rshift, current_k))[0]
            pow_gi = np.atleast_1d(P_gi(rshift, current_k))[0]
            kern_i = self.WL_window(rshift, bin_i, current_k)
            kernia_i = self.IA_window(rshift, bin_i)
            kern_j = self.GC_window(rshift, bin_j)
            pandw_gd = kern_i * kern_j * pow_gd
            pandw_gi = kernia_i * kern_j * pow_gi
            pandwijk = pandw_gd + pandw_gi
            c_int_arr[ii] = self.Cl_generic_integrand(rshift, pandwijk)
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final
