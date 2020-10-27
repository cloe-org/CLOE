# General imports
import numpy as np
import likelihood.cosmo
from scipy import integrate
from scipy import interpolate
import os.path

# General error class


class ShearError(Exception):
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
            Dictionary contain
        """
        self.theory = cosmo_dic
        self.nz_dic_WL = nz_dic_WL
        self.nz_dic_GC = nz_dic_GC
        if self.theory['r_z_func'] is None:
            raise Exception('No interpolated function for comoving distance '
                            'exists in cosmo_dic.')
        self.cl_int_min = 0.001
        self.cl_int_max = self.theory['z_win'][-1]

    def GC_window(self, k, z, bin_i):
        """
        Implements GC window

        Parameters
        ----------
        k: float
           Scale at which to evaluate the bias
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
            W_i^G(k, z) &=b(k, z)n_i(z)/\bar{n_i}H(z)\\
        """

        n_z_normalized = self.nz_dic_GC[''.join(['n', str(bin_i)])]

        W_i_G = (n_z_normalized(z) * self.theory['H_z_func'](z) /
                 self.theory['c'])
        return W_i_G

    def WL_window_integrand(self, zprime, z, nz):
        """
        Calculates the WL kernel integrand.

        .. math::
        \int_{z}^{z_{\rm max}}
        {{\rm d}z^{\prime} n_{i}^{\rm WL}(z^{\prime}) \left [ 1 -
        \frac{\tilde{r}(z)}{\tilde{r}(z^{\prime}} \right ]}

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
        """
        wint = nz(zprime) * (1.0 - (self.theory['r_z_func'](z) /
                                    self.theory['r_z_func'](zprime)))
        return wint

    def WL_window(self, z, bin_i):
        """
        Calculates the W^{\gamma} lensing kernel for a given tomographic bin.

        .. math::
        W_{i}^{\gamma}(z) = \frac{3 H_0}{2 c}
        \Omega_{{\rm m},0} (1 + z) \Sigma(z,k) \tilde{r}(z)
        \int_{z}^{z_{\rm max}}
        {{\rm d}z^{\prime} n_{i}^{\rm WL}(z^{\prime}) \left [ 1 -
        \frac{\tilde{r}(z)}{\tilde{r}(z^{\prime}} \right ]}

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
                                           a=z, b=4.0,
                                           args=(z, n_z_normalized))[0])
        return W_val

    def Cl_generic_integrand(self, z, W_i_z, W_j_z, k, P_int):
        """
        Calculates the C_\ell integrand for any two probes and bins for which
        the bins are supplied.

        .. math::
        \frac{W_{i}^{A}(z)W_{j}^{B}(z)}{H(z)r^2(z)}P_{\delta\delta}(k=
        (\ell + 0.5)/r(z), z).

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
        Value of C_\ell integrand at redshift z.
        """
        kern_mult = ((W_i_z * W_j_z) /
                     (self.theory['H_z_func'](z) *
                      (self.theory['r_z_func'](z)) ** 2.0))
        power = P_int(z, k)
        return kern_mult * power

    def Cl_WL(self, ell, bin_i, bin_j, int_step=0.1):
        """
        Calculates C_\ell for weak lensing, for the supplied bins.

        .. math::
        c \int {\rm d}z \frac{W_{i}^{WL}(z)W_{j}^{WL}(z)}{H(z)r^2(z)}
        P_{\delta\delta}(k=(\ell + 0.5)/r(z), z).

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
        Value of C_\ell.
        """

        int_zs = np.arange(self.cl_int_min, self.cl_int_max, int_step)

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
        """
        Calculates C_\ell for photometric galaxy clustering, for the
        supplied bins.

        .. math::
        c \int {\rm d}z \frac{W_{i}^{GC}(k, z)W_{j}^{GC}(k, z)}{H(z)r^2(z)}
        P_{\delta\delta}(k=(\ell + 0.5)/r(z), z).

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
        Value of C_\ell.
        """

        int_zs = np.arange(self.cl_int_min, self.cl_int_max, int_step)

        c_int_arr = []
        for rshft in int_zs:
            # (ACD): Although k is specified here for the GC window function,
            # note that the implementation currently uses a scale-independent
            # bias.
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshft)
            kern_i = self.GC_window(current_k, rshft, bin_i)
            kern_j = self.GC_window(current_k, rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       current_k, self.theory[
                                                           'Pgg_phot']))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_cross(self, ell, bin_i, bin_j, int_step=0.1):
        """
        Calculates C_\ell for cross-correlation between weak lensing and
        galaxy clustering, for the supplied bins.

        .. math::
        c \int {\rm d}z \frac{W_{i}^{WL}(z)W_{j}^{GC}(k, z)}{H(z)r^2(z)}
        P_{\delta\delta}(k=(\ell + 0.5)/r(z), z).

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
        Value of C_\ell.
        """

        int_zs = np.arange(self.cl_int_min, self.cl_int_max, int_step)

        c_int_arr = []
        for rshft in int_zs:
            # (ACD): Although k is specified here for the GC window function,
            # note that the implementation currently uses a scale-independent
            # bias.
            current_k = (ell + 0.5) / self.theory['r_z_func'](rshft)
            kern_i = self.WL_window(rshft, bin_i)
            kern_j = self.GC_window(current_k, rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       current_k, self.theory[
                                                           'Pgdelta_phot']))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final
