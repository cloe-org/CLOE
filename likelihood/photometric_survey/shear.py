# General imports
import numpy as np
import likelihood.cosmo
from scipy import integrate

# Import auxilary classes
from ..general_specs.estimates import Galdist

# General error class


class ShearError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Shear:
    """
    Class for Shear observable
    """

    def __init__(self, cosmo_dic):
        """
        Constructor of the class Shear

        Parameters
        ----------
        cosmo_dic: dictionary
           cosmological dictionary from cosmo
        """
        self.theory = cosmo_dic
        if self.theory['r_z_func'] is None:
            raise Exception('No interpolated function for comoving distance '
                            'exists in cosmo_dic.')

    # SJ: k-indep bias for now
    # def bias(self, k, z):
    # Use theory approach for bias for now
    # def phot_galbias(self, bin_i):
    def phot_galbias(self, bin_z_min, bin_z_max):
        """
        Returns the photometric galaxy bias.
        For now use Eqn 133 in arXiv:1910.09273

        Parameters
        ----------
        # SJ: k-indep bias for now
        # k: float
        #    Scale at which to evaluate the bias
        # z: float
        #    Redshift at which to evaluate distribution.
        # SJ: use theory approach for bias for now
        # bin_i: float
        #        Bin index
        bin_z_max: float
                   Upper limit of bin
        bin_z_min: float
                   Lower limit of bin

        Returns
        -------
        b: float
           galaxy bias

        Notes
        -----
        .. math::
            b(z) &= \sqrt{1+\bar{z}}\\
        """

        # SJ: We could eventually have a bias parameter for each
        # SJ: tomographic bin (also see yaml file)
        # phot_galbias = [params_values.get(p, None) for p in \
        #     ['phot_b1', 'phot_b2', 'phot_b3', 'phot_b4', 'phot_b5', \
        #     'phot_b6', 'phot_b7', 'phot_b8', 'phot_b9', 'phot_b10']]

        b = np.sqrt(1.0 + (bin_z_min + bin_z_max) / 2.0)
        # SJ: Yet another option, not used
        # b = np.sqrt(1 + z)
        return b

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

        # (GCH): create instance from Galdist class
        try:
            galdist = Galdist(bin_i)
        except ShearError:
            print('Error in initializing the class Galdist')
        # (GCH): call n_z_normalized from Galdist
        n_z_normalized = galdist.n_i
        # SJ: k-indep bias, let us follow the IST:F approach for now
        # W_i_G = self.phot_galbias(k, z) * n_z_normalized(z) * \
        #     self.theory['H'](z)
        W_i_G = self.phot_galbias(n_z_normalized.get_knots()[0],
                                  n_z_normalized.get_knots()[-1]) * \
            n_z_normalized(z) * self.theory['H_z_func'](z)
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
               (self.theory['ombh2'] / (H0 / 100.0)**2.0))

        # create instance from Galdist class
        try:
            galdist = Galdist(bin_i)
        except ShearError:
            print('Error in initializing the class Galdist')
        # call n_z_normalized from Galdist
        n_z_normalized = galdist.n_i

        # (ACD): Note that impact of MG is currently neglected (\Sigma=1).
        W_val = (1.5 * (H0 / c) * O_m * (1.0 + z) * (
            self.theory['r_z_func'](z) /
                (c / H0)) * integrate.quad(self.WL_window_integrand,
                                           a=z, b=galdist.survey_max -
                                           galdist.int_step,
                                           args=(z, n_z_normalized))[0])
        return W_val

    def Cl_generic_integrand(self, z, W_i_z, W_j_z, ell):
        """
        Calculates the C_\ell integrand for any two probes and bins for which
        the bins are supplied.

        .. math::
        \frac{W_{i}^{A}(z)W_{j}^{B}(z)}{H(z)r^2(z)}P_{\delta\delta}(k=
        \ell + 0.5/r(z), z).

        Parameters
        ----------
        z: float
            Redshift at which integrand is being evaluated.
        W_i_z: float
           Value of kernel for bin i, at redshift z.
        W_j_z: float
           Value of kernel for bin j, at redshift z.
        ell: float
           \ell-mode at which the current C_\ell is being evaluated at.
        Returns
        -------
        Value of C_\ell integrand at redshift z.
        """
        kern_mult = ((W_i_z * W_j_z) /
                     (self.theory['H_z_func'](z) *
                      (self.theory['r_z_func'](z)) ** 2.0))
        k = (ell + 0.5) / self.theory['r_z_func'](z)
        power = self.theory['Pk_interpolator'].P(z, k)
        return kern_mult * power

    def Cl_WL(self, ell, bin_i, bin_j):
        """
        Calculates C_\ell for weak lensing, for the supplied bins.

        .. math::
        c \int {\rm d}z \frac{W_{i}^{WL}(z)W_{j}^{WL}(z)}{H(z)r^2(z)}
        P_{\delta\delta}(k=\ell + 0.5/r(z), z).

        Parameters
        ----------
        ell: float
            \ell-mode at which C_\ell is evaluated.
        bin_i: int
           index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           index of second tomographic bin. Tomographic bin
           indices start from 1.
        Returns
        -------
        Value of C_\ell.
        """

        int_zs = np.arange(0.001, self.theory['z_win'][-1], 0.1)

        c_int_arr = []
        for rshft in int_zs:
            kern_i = self.WL_window(rshft, bin_i)
            kern_j = self.WL_window(rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       ell))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_GC_phot(self, ell, bin_i, bin_j):
        """
        Calculates C_\ell for photometric galaxy clustering, for the
        supplied bins.

        .. math::
        c \int {\rm d}z \frac{W_{i}^{GC}(z)W_{j}^{GC}(z)}{H(z)r^2(z)}
        P_{\delta\delta}(k=\ell + 0.5/r(z), z).

        Parameters
        ----------
        ell: float
            \ell-mode at which C_\ell is evaluated.
        bin_i: int
           index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           index of second tomographic bin. Tomographic bin
           indices start from 1.
        Returns
        -------
        Value of C_\ell.
        """

        int_zs = np.arange(0.001, self.theory['z_win'][-1], 0.1)

        c_int_arr = []
        for rshft in int_zs:
            kern_i = self.GC_window(0.1, rshft, bin_i)
            kern_j = self.GC_window(0.1, rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       ell))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def Cl_cross(self, ell, bin_i, bin_j):
        """
        Calculates C_\ell for cross-correlation between weak lensing and
        galaxy clustering, for the supplied bins.

        .. math::
        c \int {\rm d}z \frac{W_{i}^{WL}(z)W_{j}^{GC}(z)}{H(z)r^2(z)}
        P_{\delta\delta}(k=\ell + 0.5/r(z), z).

        Parameters
        ----------
        ell: float
            \ell-mode at which C_\ell is evaluated.
        bin_i: int
           index of first tomographic bin. Tomographic bin
           indices start from 1.
        bin_j: int
           index of second tomographic bin. Tomographic bin
           indices start from 1.
        Returns
        -------
        Value of C_\ell.
        """

        int_zs = np.arange(0.001, self.theory['z_win'][-1], 0.1)

        c_int_arr = []
        for rshft in int_zs:
            kern_i = self.WL_window(rshft, bin_i)
            kern_j = self.GC_window(0.1, rshft, bin_j)
            c_int_arr.append(self.Cl_generic_integrand(rshft, kern_i, kern_j,
                                                       ell))
        c_final = self.theory['c'] * integrate.trapz(c_int_arr, int_zs)

        return c_final

    def loglike(self):
        """
        Returns loglike for Shear observable
        """

        # likelihood value is currently only a place holder!
        loglike = 0.0
        return loglike
