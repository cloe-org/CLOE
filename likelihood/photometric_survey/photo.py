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
        Implements GC window

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
           GCph window function

        Notes
        -----
        .. math::
            W_i^G(z) &= n_i(z)/\bar{n_i}H(z)/c\\
        """

        n_z_normalized = self.nz_dic_GC[''.join(['n', str(bin_i)])]

        W_i_G = (n_z_normalized(z) * self.theory['H_z_func'](z) /
                 self.theory['c'])
        return W_i_G

    def WL_window_integrand(self, zprime, z, nz):
        r"""
        Calculates the WL kernel integrand.

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
        Integrand value

        Notes
        -----
        .. math::
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \left [ 1 - \frac{\tilde{r}(z)}{\tilde{r}(z^{\prime}} \right ]}
        """
        wint = nz(zprime) * (1.0 - (self.theory['r_z_func'](z) /
                                    self.theory['r_z_func'](zprime)))
        return wint

    def WL_window(self, z, bin_i):
        r"""
        Calculates the weak lensing shear kernel for a given tomographic bin.

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated.
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices start from 1.
        Returns
        -------
        Value of lensing kernel for specified bin at specified redshift.

        Notes
        -----
        .. math::
            W_{i}^{\gamma}(z) = \frac{3 H_0}{2 c}\Omega_{{\rm m},0} (1 + z)\
            Sigma(z,k) \tilde{r}(z)\int_{z}^{z_{\rm max}}{{\rm d}z^{\prime}\
            n_{i}^{\rm WL}(z^{\prime})\
            left [ 1 -\frac{\tilde{r}(z)}{\tilde{r}(z^{\prime}} \right ]}\\
        """
        H0 = self.theory['H0']
        c = self.theory['c']
        O_m = ((self.theory['omch2'] / (H0 / 100.0)**2.0) +
               (self.theory['ombh2'] / (H0 / 100.0)**2.0) +
               (self.theory['omnuh2'] / (H0 / 100.0)**2.0))

        n_z_normalized = self.nz_dic_WL[''.join(['n', str(bin_i)])]

        # (ACD): Note that impact of MG is currently neglected (\Sigma=1).
        W_val = (1.5 * (H0 / c) * O_m * (1.0 + z) * (
            self.theory['r_z_func'](z) /
                (c / H0)) * integrate.quad(self.WL_window_integrand,
                                           a=z, b=self.cl_int_z_max,
                                           args=(z, n_z_normalized))[0])
        return W_val

    def Cl_generic_integrand(self, z, W_i_z, W_j_z, k, P_int):
        r"""
        Calculates the angular power spectrum integrand
        for any two probes and bins for which
        the bins are supplied.

        Parameters
        ----------
        z: float
            Redshift at which integrand is being evaluated.
        W_i_z: float
           Value of kernel for bin i, at redshift z.
        W_j_z: float
           Value of kernel for bin j, at redshift z.
        k: float
           Scale at which the current C_\ell is being evaluated at.
        P_int: obj
            Choice of power spectrum interpolator. Either matter power spectrum
            GG power spectrum, or G-delta power spectrum.
        Returns
        -------
        Value of angular power spectrum integrand at redshift z.

        Notes
        -----
        .. math::
            \frac{W_{i}^{\rm A}(z)W_{j}^{\rm B}(z)}{H(z)r^2(z)}\
            P_{\rm AB}(k=({\rm\ell} + 0.5)/r(z), z)\\
            \text{A, B in {G, L}}
        """
        kern_mult = ((W_i_z * W_j_z) /
                     (self.theory['H_z_func'](z) *
                      (self.theory['r_z_func'](z)) ** 2.0))
        power = P_int(z, k)
        return kern_mult * power

    def Cl_WL(self, ell, bin_i, bin_j, int_step=0.1):
        r"""
        Calculates angular power spectrum for weak lensing,
        for the supplied bins.

        Parameters
        ----------
        ell: float
            \ell-mode at which C_\ell is evaluated.
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
        Value of angular power spectrum.


        Notes
        -----
        .. math::
            c \int {\rm d}z \frac{W_{i}^{\rm WL}(z)W_{j}^{\rm WL}(z)}\
            {H(z)r^2(z)}P_{\delta\delta}(k=({\rm\ell} + 0.5)/r(z), z)\\
        """

        int_zs = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        c_int_arr = []
        for rshft in int_zs:
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshft)
            kern_i = self.WL_window(rshft, bin_i)
            kern_j = self.WL_window(rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       current_k,
                                                       self.theory[
                                                           'Pk_interpolator'].P
                                                       ))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_GC_phot(self, ell, bin_i, bin_j, int_step=0.1):
        r"""
        Calculates angular power spectrum for photometric galaxy clustering,
        for the supplied bins.

        Parameters
        ----------
        ell: float
            \ell-mode at which C_\ell is evaluated.
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
        Value of angular power spectrum for photometric galaxy clustering.


        Notes
        -----
        .. math::
            c \int {\rm d}z \frac{W_{i}^{\rm GC}(k, z)W_{j}^{\rm GC}(k, z)}\
            {H(z)r^2(z)}P_{\rm GG}(k=(\ell + 0.5)/r(z), z)\\
        """

        int_zs = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        c_int_arr = []
        for rshft in int_zs:
            # (ACD): Although k is specified here for the GC window function,
            # note that the implementation currently uses a scale-independent
            # bias.
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshft)
            kern_i = self.GC_window(rshft, bin_i)
            kern_j = self.GC_window(rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       current_k, self.theory[
                                                           'Pgg_phot']))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_cross(self, ell, bin_i, bin_j, int_step=0.1):
        r"""
        Calculates angular power spectrum for cross-correlation
        between weak lensing and galaxy clustering, for the supplied bins.

        Parameters
        ----------
        ell: float
            \ell-mode at which C_\ell is evaluated.
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
        Value of cross correlation angular power spectrum.


        Notes
        -----
        .. math::
            c \int {\rm d}z \frac{W_{i}^{\rm WL}(z)W_{j}^{\rm GC}(k, z)}\
            {H(z)r^2(z)}P_{\rm G\delta}(k=({\rm \ell} + 0.5)/r(z), z)\\
        """

        int_zs = np.arange(self.cl_int_z_min, self.cl_int_z_max, int_step)

        c_int_arr = []
        for rshft in int_zs:
            # (ACD): Although k is specified here for the GC window function,
            # note that the implementation currently uses a scale-independent
            # bias.
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshft)
            kern_i = self.WL_window(rshft, bin_i)
            kern_j = self.GC_window(rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       current_k, self.theory[
                                                           'Pgdelta_phot']))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final
