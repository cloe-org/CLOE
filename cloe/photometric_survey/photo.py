"""PHOTO

This module computes the photometric quantities following the v2.0 recipes
of CLOE.
"""

# General imports
import numpy as np
from scipy import integrate, interpolate
from scipy.special import jv
from cloe.photometric_survey.redshift_distribution \
    import RedshiftDistribution
from cloe.auxiliary.redshift_bins import linear_interpolator
import warnings

# General error class


class PhotoError(Exception):
    r"""
    Class to define Exception Error.
    """

    pass


class Photo:
    """
    Class for photometric observables.
    """

    def __init__(self, cosmo_dic, nz_dic_WL, nz_dic_GC, add_RSD=False):
        """Initialise.

        Constructor of the class Photo.

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary from :obj:`cosmology` class
        nz_dic_WL: dict
            Dictionary containing redshift distributions for WL probe
        nz_dic_GC: dict
            Dictionary containing redshift distributions for GCphot probe
        add_RSD: bool
            Flag to determine whether RSD have to be included in the
            calculations for GCphot or not
        """

        self.nz_dic_WL = nz_dic_WL
        self.nz_dic_GC = nz_dic_GC

        self.cl_int_z_min = 0.001

        # The size of z_winterp sufficient for now, could be tuned later
        z_wlogmin = -2.0
        z_wmin1 = 1e-5
        z_wmin2 = 1e-4
        z_wmin3 = 1e-3
        z_wmax = 4.0
        self.z_wsamp = 1000
        self.z_trapz_sampling = 500
        self.z_winterp = np.logspace(z_wlogmin,
                                     np.log10(z_wmax),
                                     self.z_wsamp)
        self.z_winterp[0] = z_wmin1
        self.z_winterp[1] = z_wmin2
        self.z_winterp[2] = z_wmin3
        # Number of bins should be generalized, hard-coded for now

        # Cl integral z step (TODO: take from the accuracy module)
        int_step = 0.01
        self.z_grid_for_cl = np.arange(self.cl_int_z_min, z_wmax, int_step)

        # ell grid integrated over to obtain the 3x2pt correlation functions
        self.ell_max = int(1e5)
        self.nint = 128
        self.ells_int = np.append(
            np.linspace(2.0, 9.0, 8),
            np.logspace(1.0, np.log10(self.ell_max + 1), self.nint - 8))
        self.ells_dense = \
            np.linspace(2, self.ell_max, self.ell_max - 1).astype(int)

        self.bessel_dict = {}
        self._prefactor_dict = {}

        self.ells_WL = []
        self.ells_GC_phot = []
        self.ells_XC = []

        self.add_RSD = add_RSD
        self._ells_GC_or_XC = None

        self.multiply_bias_cl = False

        # The class might be initialized with no cosmo dictionary, as it is
        # currently done when instantiating Photo from Euclike, for running
        # the likelihood of CLOE
        if cosmo_dic is not None:
            self.update(cosmo_dic)

    def calc_nz_distributions(self, cosmo_dic):
        r"""calc_nz_distributions method

        Method to compute the n(z) distributions, based on the
        input cosmology and nuisance parameters of the cosmo_dic

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary from Cosmology class.
        """
        nuisance_dict = cosmo_dic['nuisance_parameters']
        self.nz_GC = RedshiftDistribution('GCphot', self.nz_dic_GC,
                                          nuisance_dict)
        self.nz_WL = RedshiftDistribution('WL', self.nz_dic_WL,
                                          nuisance_dict)
        return None

    def update(self, cosmo_dic):
        r"""Updates method.

        Method to update the theory class attribute to the passed cosmo
        dictionary, and recompute all cosmology-dependent quantities.

        The model for magnification galaxy bias is determined with the value of
        the `magbias_model` key in the `cosmo_dic` input parameter.

        The implemented models are:

            #. Linear interpolation between input bin values
            #. Constant value per tomographic bin
            #. Third-order polynomial bias function

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary from :obj:`cosmology` class
        """
        self.theory = cosmo_dic
        nuisance_dict = self.theory['nuisance_parameters']
        obs_sel = self.theory['obs_selection']

        # Commenting this part out as we are temporarily not using the
        # angular diameter distance obtained from CAMB, but we are coding
        # it up ourselves
        # self.vadd2 = np.vectorize(
        #         self.theory['CAMBdata'].angular_diameter_distance2)

        if self.theory['bias_model'] == 2:
            self.multiply_bias_cl = True

        if self.theory['f_K_z_func'] is None:
            raise KeyError('No interpolated function for transverse comoving '
                           'distance exists in cosmo_dic.')

        self.f_K_z_grid = self.theory['f_K_z_func'](self.z_grid_for_cl)
        self.H_z_grid = self.theory['H_z_func'](self.z_grid_for_cl)

        z_wmax = self.theory['z_win'][-1]

        # holders for power spectra in Limber approximation
        self.power_dd_WL = []
        self.power_ii_WL = []
        self.power_di_WL = []
        self.power_gg_GC = []
        self.power_dd_GC = []
        self.power_gd_GC = []
        self.power_gd_XC = []
        self.power_gi_XC = []
        self.power_dd_XC = []
        self.power_di_XC = []
        self._stored_WL = False
        self._stored_GC = False
        self._stored_XC = False

        if any(obs_sel['GCphot'].values()):
            self.nz_GC = RedshiftDistribution('GCphot', self.nz_dic_GC,
                                              nuisance_dict)
            self.photobias = [nuisance_dict[f'b{i}_photo']
                              for i in self.nz_GC.get_tomographic_bins()]

            if self.theory['magbias_model'] <= 2:
                self.magbias = [nuisance_dict[f'magnification_bias_{i}']
                                for i in self.nz_GC.get_tomographic_bins()]
                if self.theory['magbias_model'] == 1:
                    self.magbias = \
                        linear_interpolator(
                            self.theory['redshift_bins_means_phot'],
                            self.magbias)
            elif self.theory['magbias_model'] == 3:
                self.magbias = self.poly_mag_bias
            else:
                raise ValueError('magbias_model flag can only be selected '
                                 'from the values (1,2,3)')

            # z_wtom is the number of tomographic bins + 1
            z_wtom_gc = 1 + self.nz_GC.get_num_tomographic_bins()

            self.interpwingal = np.zeros(shape=(self.z_wsamp, z_wtom_gc))
            self.interpwinmag = np.zeros(shape=(self.z_wsamp, z_wtom_gc))
            self.interpwingal[:, 0] = self.z_winterp
            self.interpwinmag[:, 0] = self.z_winterp
            for tom_i in range(1, z_wtom_gc):
                self.interpwingal[:, tom_i] = self.GC_window(self.z_winterp,
                                                             tom_i)
                self.interpwinmag[:, tom_i] = self.magnification_window(
                    self.z_winterp, tom_i)
            if self.add_RSD and self._ells_GC_or_XC is not None:
                self.interpwinrsd = np.zeros(
                    shape=(z_wtom_gc, 3, len(self._ells_GC_or_XC),
                           self.z_wsamp))
                self.interpwinrsd[0, :, :, :] = self.z_winterp
                for tom_i in range(1, z_wtom_gc):
                    self.interpwinrsd[tom_i, :, :, :] = \
                        self.GC_window_RSD(self.z_winterp,
                                           self._ells_GC_or_XC, tom_i)
        if any(obs_sel['WL'].values()):
            self.nz_WL = RedshiftDistribution('WL', self.nz_dic_WL,
                                              nuisance_dict)
            self.multbias = [nuisance_dict[f'multiplicative_bias_{i}']
                             for i in self.nz_WL.get_tomographic_bins()]
            # z_wtom is the number of tomographic bins + 1
            z_wtom_wl = 1 + self.nz_WL.get_num_tomographic_bins()

            self.wl_int_z_max = {i: z_wmax
                                 for i in self.nz_WL.get_tomographic_bins()}
            self.interpwin = np.zeros(shape=(self.z_wsamp, z_wtom_wl))
            self.interpwinia = np.zeros(shape=(self.z_wsamp, z_wtom_wl))
            self.interpwin[:, 0] = self.z_winterp
            self.interpwinia[:, 0] = self.z_winterp
            for tom_i in range(1, z_wtom_wl):
                self.interpwin[:, tom_i] = self.WL_window(self.z_winterp,
                                                          tom_i)
                self.interpwinia[:, tom_i] = self.IA_window(self.z_winterp,
                                                            tom_i)

    def _set_bessel_tables(self, theta_rad):
        r"""Sets tables for Bessel functions.

        Method to set the dictionary containing the precomputed tables of the
        Bessel functions of order 0,2,4.

        Parameters
        ----------
        theta_rad: list, numpy.ndarray
            Values of the angular separation (in radians) at which the Bessel
            functions have to be evaluated
        """
        bessel0_grid = np.array([jv(0, self.ells_dense * th)
                                 for th in theta_rad])
        bessel2_grid = np.array([jv(2, self.ells_dense * th)
                                 for th in theta_rad])
        bessel4_grid = np.array([jv(4, self.ells_dense * th)
                                 for th in theta_rad])
        self.bessel_dict[0] = bessel0_grid
        self.bessel_dict[2] = bessel2_grid
        self.bessel_dict[4] = bessel4_grid
        warnings.warn('Bessel tables have been set with the specified angular '
                      'separations. Computing 3x2pt correlation functions at '
                      'different angles will lead to unexpected outputs.')

    def set_prefactor(self, ells_WL=None, ells_XC=None, ells_GC_phot=None):
        r"""Computes the prefactors for the WL XC and GCphot :math:`C(\ell)`'s.

        The arrays of :math:`\ell` values provided as input are stored
        in the class instance, to be used in other private methods.
        The following prefactors are evaluated.

        * :obj:`shearIA_WL` (from the ells_WL input array): \
          :math:`\left[(\ell+2)!/(\ell-2)!\right]/(\ell+1/2)^4`
        * :obj:`shearIA_XC` (from the ells_XC input array): \
          :math:`\sqrt{(\ell+2)!/(\ell-2)!}/(\ell+1/2)^2`
        * :obj:`mag_XC` (from the ells_XC input array): \
          :math:`\ell(\ell+1)/(\ell+1/2)^2`
        * :obj:`mag_GCphot` (from the ells_GCphot input array): \
          :math:`\ell(\ell+1)/(\ell+1/2)^2`
        * :obj:`L0_GCphot` (from the ells_GCphot input array): \
          :math:`\frac{2 \ell^2 + 2 \ell -1}{(2 \ell - 1)(2 \ell + 3)}`
        * :obj:`Lplus1_GCphot` (from the ells_GCphot input array):

          .. math::
              -\frac{(\ell + 1)(\ell + 2)}
              {(2\ell + 3)\sqrt{(2\ell + 1)(2 \ell + 5)}}

        * :obj:`Lminus1_GCphot` (from the ells_GCphot input array):

          .. math::
              -\frac{\ell(\ell - 1)}
              {(2 \ell -1)\sqrt{(2 \ell - 3)(2 \ell + 1)}}\\

        The prefactors are evaluated using the functions
        :meth:`_eval_prefactor_mag()`, :meth:`_eval_prefactor_shearia()`,
        :meth:`_eval_prefactor_l_0()`, :meth:`_eval_prefactor_l_plus1()`,
        and :meth:`_eval_prefactor_l_minus1()`.

        Parameters
        ----------
        ells_WL: numpy.ndarray
           Array of :math:`\ell` values for the WL probe
        ells_XC: numpy.ndarray
           Array of :math:`\ell` values for the XCphot probe
        ells_GC_phot: numpy.ndarray
           Array of :math:`\ell` values for the GCphot probe
        """

        # WL
        if ells_WL is not None:
            self._prefactor_dict['shearIA_WL'] = \
                (self._eval_prefactor_shearia(ells_WL))**2
            self.ells_WL = ells_WL

        # XC
        if ells_XC is not None:
            self._prefactor_dict['shearIA_XC'] = \
                self._eval_prefactor_shearia(ells_XC)
            self._prefactor_dict['mag_XC'] = \
                self._eval_prefactor_mag(ells_XC)
            self.ells_XC = ells_XC

        # GCphot
        if ells_GC_phot is not None:
            self._prefactor_dict['mag_GCphot'] = \
                self._eval_prefactor_mag(ells_GC_phot)
            self._prefactor_dict['L0_GCphot'] = \
                self._eval_prefactor_l_0(ells_GC_phot)
            self._prefactor_dict['Lplus1_GCphot'] = \
                self._eval_prefactor_l_plus1(ells_GC_phot)
            self._prefactor_dict['Lminus1_GCphot'] = \
                self._eval_prefactor_l_minus1(ells_GC_phot)
            self.ells_GC_phot = ells_GC_phot

        if self.add_RSD:
            if ells_GC_phot is not None and ells_XC is not None:
                self._ells_GC_or_XC = \
                        np.union1d(self.ells_GC_phot, self.ells_XC)
                self._prefactor_dict['L0_GC_or_XC'] = \
                    self._eval_prefactor_l_0(self._ells_GC_or_XC)
                self._prefactor_dict['Lplus1_GC_or_XC'] = \
                    self._eval_prefactor_l_plus1(self._ells_GC_or_XC)
                self._prefactor_dict['Lminus1_GC_or_XC'] = \
                    self._eval_prefactor_l_minus1(self._ells_GC_or_XC)

    def GC_window(self, z, bin_i):
        r"""GC window.

        Implements the galaxy clustering photometric window function.

        .. math::
            W_i^G(z) &= \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\

        Parameters
        ----------
        z: numpy.ndarray of float or float
           Redshift at which to evaluate distribution
        bin_i: int
           Index of desired tomographic bin. Tomographic bin
           indices start from 1

        Returns
        -------
        GCphot window function: float
           Window function for photometric galaxy clustering
        """

        n_z_normalized = self.nz_GC.evaluates_n_i_z(bin_i, z)
        window_GC = n_z_normalized * self.theory['H_z_func_Mpc'](z)

        return window_GC

    def GC_window_RSD(self, z, ell, bin_i):
        r"""GC window RSD.

        Implements the RSD correction to the galaxy clustering photometric
        window function in an array-like format, modulo the Limber and
        full sky prefactor,

        .. math::
            W_i^{\rm{G,RSD}}(z,\ell) =
            \frac{1}{c \,b_{\mathrm{g},i}^\mathrm{photo}} \
            \left[H(z_{\rm m})f(z_{\rm m})\
            \frac{n_i(z)}{\bar{n_i}}\right]_{\rm m}

        where :math:`m` assumes the values (-1,0,+1).

        Parameters
        ----------
        z: numpy.ndarray or float
            Redshift at which to evaluate the window function
        ell: numpy.ndarray or float
            Multipole at which to evaluate the window function
        bin_i: int
            Index of desired tomographic bin. Tomographic bin
            indices start from 1

        Returns
        -------
        RSD GCphot window function: numpy.ndarray
            Window function for RSD component of photometric galaxy clustering.
        """
        if isinstance(ell, (int, float)):
            ell = [ell]
        if isinstance(z, (int, float)):
            z = [z]

        tdist = self.theory['f_K_z_func'](z)
        zm_arr = np.array([[self.z_minus1(ll, tdist) for ll in ell],
                           np.full((len(ell), len(z)), z),
                           [self.z_plus1(ll, tdist) for ll in ell]])

        Hzm_arr = self.theory['H_z_func_Mpc'](zm_arr)
        fzm_arr = self.theory['f_z'](zm_arr)
        nzm_arr = self.nz_GC.evaluates_n_i_z(bin_i, zm_arr)

        if self.theory['bias_model'] == 2:
            bias = self.photobias[bin_i - 1]
        elif self.theory['bias_model'] in [1, 3]:
            bias = self.theory['b_inter'](z)

        return Hzm_arr * fzm_arr * nzm_arr / bias

    def _unpack_RSD_kernel(self, ell, *args):
        r"""Obtains the RSD kernel for GCphot or XCphot.

        Checks whether the RSD kernel has been already initialized, and
        depending on that unpacks the kernel of the corresponding photometric
        bin or computes it directly.

        Parameters
        ----------
        ell: float
            Multipole at which the RSD kernel has to be computed
        *args: list
            Extra arguments of variable length, corresponding to the indices
            of the photometric bins for which the RSD kernel has to be
            computed (2 for GCphot and 1 for XCphot)

        Returns
        -------
        RSD kernel: numpy.ndarray
            RSD kernel for the specified list of photometric bins
        """
        if self._ells_GC_or_XC is not None:
            idx = np.where(self._ells_GC_or_XC == ell)[0][0]
            prefactor_Lminus1 = self._prefactor_dict['Lminus1_GC_or_XC'][idx]
            prefactor_L0 = self._prefactor_dict['L0_GC_or_XC'][idx]
            prefactor_Lplus1 = self._prefactor_dict['Lplus1_GC_or_XC'][idx]
            kerngalrsd = \
                [prefactor_Lminus1 * self.interpwinrsd[bin, 0, idx, :] +
                 prefactor_L0 * self.interpwinrsd[bin, 1, idx, :] +
                 prefactor_Lplus1 * self.interpwinrsd[bin, 2, idx, :]
                 for bin in args]
        else:
            prefactor_Lminus1 = self._eval_prefactor_l_minus1(ell)
            prefactor_L0 = self._eval_prefactor_l_0(ell)
            prefactor_Lplus1 = self._eval_prefactor_l_plus1(ell)
            kernrsd = [self.GC_window_RSD(self.z_winterp, ell, bin)
                       for bin in args]
            kerngalrsd = [prefactor_Lminus1 * kern[0, 0, :] +
                          prefactor_L0 * kern[1, 0, :] +
                          prefactor_Lplus1 * kern[2, 0, :]
                          for kern in kernrsd]

        return np.array(kerngalrsd)

    def window_integrand(self, zprime, z, nz):
        r"""Window integrand.

        Calculates the Weak-lensing (WL) kernel
        or the Magnification bias kernel integrands
        with A = {L, G}, as

        .. math::
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm A}(z^{\prime})
            \frac{f_{K}\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}
            }

        Parameters
        ----------
        zprime: float or numpy.ndarray
            Redshift parameter that will be integrated over
        z: float
            Redshift at which kernel is being evaluated
        nz: Interpolator or function
            Galaxy distribution function for the tomographic bin for
            which the kernel is currently being evaluated

        Returns
        -------
        Kernel integrand: float
            Weak-lensing (WL) kernel or the magnification bias kernel integrand
        """
        # Commenting out this piece of code, as we are not using the
        # angular diameter distance function obtained from CAMB for now
        # wint = (
        #     nz(zprime) *
        #     self.vadd2(
        #         z,
        #         zprime) /
        #     self.theory['d_z_func'](zprime))
        curv = self.theory['Omk']

        if curv == 0.0:
            wint = nz(zprime) * (1.0 - (self.theory['r_z_func'](z) /
                                 self.theory['r_z_func'](zprime)))
        else:
            wint = (nz(zprime) * self.theory['f_K_z12_func'](
                z, zprime) / (1.0 + zprime) / self.theory['d_z_func'](zprime))

        return wint

    def WL_window(self, z, bin_i, k=0.0001):
        r"""WL window.

        Calculates the weak lensing shear kernel for a given tomographic bin.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        :obj:`integrate.trapz` on the array along one axis

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
        z: numpy.ndarray of float
            Redshift at which weight is evaluated.
        bin_i: int
           Index of desired tomographic bin. Tomographic bin
           indices start from 1.
        k: float
            Wavenumber at which to evaluate the Modified Gravity
            :math:`\Sigma(z,k)` function

        Returns
        -------
        Shear kernel: numpy.ndarray
           1-D Numpy array of shear kernel values for specified bin
           at specified scale for the redshifts defined in z
        """
        dz_i = self.theory['nuisance_parameters'][f'dz_{bin_i}_WL']
        zint_mat = np.linspace(z, z[-1],
                               self.z_trapz_sampling)
        zint_mat = zint_mat.T
        diffz = np.diff(zint_mat)
        H0_Mpc = self.theory['H0_Mpc']
        O_m = self.theory['Omm']

        n_z_normalized = self.nz_WL.interpolates_n_i(bin_i, z)

        intg_mat = np.array([self.window_integrand(zint_mat[zii],
                                                   zint_mat[zii, 0],
                                                   n_z_normalized)
                             for zii in range(len(zint_mat))])

        integral_arr = integrate.trapz(intg_mat, dx=diffz, axis=1)

        W_val = (1.5 * H0_Mpc * O_m * (1.0 + z) *
                 self.theory['MG_sigma'](z, k) *
                 (self.theory['f_K_z_func'](z) /
                  (1 / H0_Mpc)) * integral_arr)

        return W_val

    def poly_mag_bias(self, z):
        r"""Polynomial magnification bias.

        Computes bias using a 3rd order polynomial function of redshift.

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshift(s) at which to calculate bias

        Returns
        -------
        Magnification bias: float or numpy.ndarray
            Value(s) of magnification bias at input redshift(s)
        """
        nuisance = self.theory['nuisance_parameters']
        bias = (
            nuisance['b0_mag_poly'] +
            nuisance['b1_mag_poly'] * z +
            nuisance['b2_mag_poly'] * np.power(z, 2) +
            nuisance['b3_mag_poly'] * np.power(z, 3))
        return bias

    def magnification_window(self, z, bin_i, k=0.0001):
        r"""Magnification bias window.

        Calculates the magnification bias kernel for a given tomographic bin.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        integrate.trapz on the array along one axis.

        .. math::
            W_{i}^{\mu}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} b_{\rm mag, i}(1 + z) \Sigma(z, k)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm G}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\

        Parameters
        ----------
        z: numpy.ndarray of float
            Redshift at which weight is evaluated.
        bin_i: int
           Index of desired tomographic bin. Tomographic bin
           indices start from 1
        k: float
            Wavenumber at which to evaluate the Modified Gravity
            :math:`\Sigma(z,k)` function

        Returns
        -------
        Magnification window: numpy.ndarray
           1-D Numpy array of magnification bias kernel
           values for specified bin
           at specified scale for the redshifts defined in z
        """
        dz_i = self.theory['nuisance_parameters'][f'dz_{bin_i}_GCphot']
        zint_mat = np.linspace(z, z[-1],
                               self.z_trapz_sampling)
        zint_mat = zint_mat.T
        diffz = np.diff(zint_mat)
        H0_Mpc = self.theory['H0_Mpc']
        O_m = self.theory['Omm']

        n_z_normalized = self.nz_GC.interpolates_n_i(bin_i, z)

        intg_mat = np.array([self.window_integrand(zint_mat[zii],
                                                   zint_mat[zii, 0],
                                                   n_z_normalized)
                             for zii in range(len(zint_mat))])
        if self.theory['magbias_model'] != 2:
            intg_mat *= self.magbias(zint_mat)

        integral_arr = integrate.trapz(intg_mat, dx=diffz, axis=1)

        W_val = (1.5 * H0_Mpc * O_m * (1.0 + z) *
                 self.theory['MG_sigma'](z, k) *
                 (self.theory['f_K_z_func'](z) /
                  (1 / H0_Mpc)) * integral_arr)

        if self.theory['magbias_model'] == 2:
            # magbias is a list of constants
            W_val *= self.magbias[bin_i - 1]

        return W_val

    def WL_window_slow(self, z, bin_i, k=0.0001):
        r"""WL window slow.

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
           Index of desired tomographic bin. Tomographic bin
           indices start from 1.
        k: float
            Wavenumber at which to evaluate the Modified Gravity
            :math:`\Sigma(z,k)` function

        Returns
        -------
        Shear kernel: float
           Value of shear kernel for specified bin at specified redshift
           and scale
        """
        H0_Mpc = self.theory['H0_Mpc']
        O_m = self.theory['Omm']

        n_z_normalized = self.nz_WL.interpolates_n_i(bin_i, self.z_winterp)

        W_val = ((1.5 * H0_Mpc * O_m * (1.0 + z) *
                  self.theory['MG_sigma'](z, k) * (
                  self.theory['f_K_z_func'](z) /
                  (1 / H0_Mpc)) * integrate.quad(self.window_integrand,
                                                 a=z,
                                                 b=self.wl_int_z_max[bin_i],
                                                 args=(z, n_z_normalized))[0]))
        return W_val

    def IA_window(self, z, bin_i):
        r"""IA window.

        Calculates the intrinsic alignment (IA) weight function for a
        given tomographic bin

        .. math::
            W_{i}^{\rm IA}(z) = \frac{n_i^{\rm L}(z)}{\bar{n}_i^{\rm L}}\
            \frac{H(z)}{c}\\

        Parameters
        ----------
        z: numpy.ndarray of float or float
            Redshift at which weight is evaluated
        bin_i: int
           Index of desired tomographic bin. Tomographic bin
           indices start from 1

        Returns
        -------
        IA kernel: float
           Value of IA kernel for specified bin at specified redshift
        """

        n_z_normalized = self.nz_WL.evaluates_n_i_z(bin_i, z)

        W_IA = n_z_normalized * self.theory['H_z_func_Mpc'](z)

        return W_IA

    def Cl_generic_integrand(self, PandW_i_j_z_k):
        r"""Cl generic integrand.

        Calculates the integrand of the angular power spectra for
        the different probes given by the different combinations
        of window functions and power spectra.


        Parameters
        ----------
        PandW_i_j_z_k: numpy.ndarray
           Values of the product of kernel for bin i, kernel for bin j,
           and the power spectrum at redshift z and scale k.

        Returns
        -------
        Angular power spectrum integrand: numpy.ndarray
           Values of the angular power spectrum integrand at
           the given redshifts and multipoles :math:`\ell`.
        """
        kern_mult_power = \
            PandW_i_j_z_k / (self.H_z_grid * (self.f_K_z_grid ** 2))

        if np.isnan(PandW_i_j_z_k).any():
            raise Exception('Requested k, z values are outside of power'
                            ' spectrum interpolation range.')
        return kern_mult_power

    def Cl_WL(self, ells, bin_i, bin_j):
        r"""Cl WL

        Calculates angular power spectrum for weak lensing,
        for the supplied bins:

        .. math::
            C_{ij}^{\rm LL} = C_{ij}^{\gamma \gamma}(\ell) +
            C_{ij}^{\rm I\gamma}(\ell) + C_{ij}^{\rm II}(\ell)

        where :math:`\gamma` stands for gravitational shear and I for
        intrinsic shear.

        .. math::
            C_{ij}^{\rm \gamma \gamma}(\ell) =
            \frac{(\ell+2)!}{(\ell-2)!}
            \left(\frac{2}{2\ell+1}\right)^4
            \int {\rm d}z
            \frac{W_{i}^{\rm \gamma}(z)W_{j}^{\rm \gamma}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\rm \delta \delta}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm I\gamma}(\ell) =
            \frac{(\ell+2)!}{(\ell-2)!}
            \left(\frac{2}{2\ell+1}\right)^4
            \int {\rm d}z
            \frac{W_{i}^{\rm \gamma}(z)W_{j}^{\rm I}(z) +
                  W_{i}^{\rm I}(z)W_{j}^{\rm \gamma}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\rm \delta I}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm II}(\ell) =
            \frac{(\ell+2)!}{(\ell-2)!}
            \left(\frac{2}{2\ell+1}\right)^4
            \int {\rm d}z
            \frac{W_{i}^{\rm I}(z)W_{j}^{\rm I}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\rm II}
            \left[ k_{\ell}(z), z \right]

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1
        bin_j: int
           Index of second tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        Angular shear power spectrum: numpy.ndarray of float
           Values of the angular shear power spectrum
        """

        same_ells = np.array_equal(ells, self.ells_WL)
        precomputed_shearIA = 'shearIA_WL' in self._prefactor_dict.keys()
        if not precomputed_shearIA or not same_ells:
            self.set_prefactor(ells_WL=ells)

        prefactor_wl = self._prefactor_dict['shearIA_WL']

        kern_i = np.interp(self.z_grid_for_cl, self.interpwin[:, 0],
                           self.interpwin[:, bin_i])
        kern_j = np.interp(self.z_grid_for_cl, self.interpwin[:, 0],
                           self.interpwin[:, bin_j])
        kernia_i = np.interp(self.z_grid_for_cl, self.interpwinia[:, 0],
                             self.interpwinia[:, bin_i])
        kernia_j = np.interp(self.z_grid_for_cl, self.interpwinia[:, 0],
                             self.interpwinia[:, bin_j])

        self._evaluate_power_WL(force_recompute=(not same_ells))

        pandw_dd = kern_i * kern_j * self.power_dd_WL
        pandw_ii = kernia_i * kernia_j * self.power_ii_WL
        pandw_di = \
            (kern_i * kernia_j + kernia_i * kern_j) * self.power_di_WL
        pandwijk = pandw_dd + pandw_ii + pandw_di

        c_int_arr = self.Cl_generic_integrand(pandwijk)

        c_final = prefactor_wl * self.theory['c'] * \
            integrate.trapz(c_int_arr, self.z_grid_for_cl)
        c_final = c_final * (1 + self.multbias[bin_i - 1]) * \
            (1 + self.multbias[bin_j - 1])

        return c_final

    def _evaluate_power_WL(self, force_recompute=False):
        r"""Evaluate Power WL

        Evaluates and store the matter power spectra for weak lensing.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_WL or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_WL).T
            k_grid = (ell_col + 0.5) / self.f_K_z_grid

            P_dd = self.theory['Pmm_phot']
            P_ii = self.theory['Pii']
            P_di = self.theory['Pdeltai']
            self.power_dd_WL = P_dd(self.z_grid_for_cl, k_grid, grid=False)
            self.power_ii_WL = P_ii(self.z_grid_for_cl, k_grid, grid=False)
            self.power_di_WL = P_di(self.z_grid_for_cl, k_grid, grid=False)
            self._stored_WL = True

    def Cl_GC_phot(self, ells, bin_i, bin_j):
        r"""Cl GC Phot

        Calculates angular power spectrum for photometric galaxy clustering,
        for the supplied bins.

        .. math::
            C_{ij}^{\rm GG}(\ell) = C_{ij}^{\rm gg}(\ell)
            + C_{ij}^{\rm g\mu}(\ell) + C_{ij}^{\mu\mu}(\ell)

        where g stands for intrinsic number density fluctuations and \mu
        stands for lensing magnification. So

        .. math::
            C_{ij}^{\rm gg}(\ell) =
            \int {\rm d}z
            \frac{W_{i}^{\rm g}(z)W_{j}^{\rm g}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\rm gg}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm g\mu}(\ell) =
            \frac{\ell(\ell+1)}{(\ell+1/2)^2}
            \int {\rm d}z
            \frac{W_{i}^{\rm \mu}(z)W_{j}^{\rm g}(z) +
                  W_{i}^{\rm g}(z)W_{j}^{\rm \mu}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\rm g\delta}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm \mu\mu}(\ell) =
            \left(\frac{\ell(\ell+1)}{(\ell+1/2)^2} \right)^2
            \int {\rm d}z
            \frac{W_{i}^{\rm \mu}(z)W_{j}^{\rm \mu}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\delta \delta}
            \left[ k_{\ell}(z), z \right]

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1
        bin_j: int
           Index of second tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        Angular GCphot power spectrum: float
           Values of angular power spectrum for
           galaxy clustering photometric
        """

        same_ells = np.array_equal(ells, self.ells_GC_phot)
        precomputed_mag = 'mag_GCphot' in self._prefactor_dict.keys()

        if not precomputed_mag or not same_ells:
            self.set_prefactor(ells_GC_phot=ells)

        prefactor_mag = self._prefactor_dict['mag_GCphot']
        prefactor_mag_col = np.atleast_2d(prefactor_mag).T

        if self.add_RSD:
            kerngalrsd = \
                np.array([self._unpack_RSD_kernel(ell, bin_i, bin_j)
                          for ell in ells])
            kerngalrsd_i = kerngalrsd[:, 0, :]
            kerngalrsd_j = kerngalrsd[:, 1, :]
        else:
            kerngalrsd_i = np.zeros(len(ells))
            kerngalrsd_j = np.zeros(len(ells))

        kerngal_i = \
            np.array(
                [np.interp(self.z_grid_for_cl, self.interpwingal[:, 0],
                           self.interpwingal[:, bin_i] + kerngalrsd_ell_i)
                 for kerngalrsd_ell_i in kerngalrsd_i])
        kerngal_j = \
            np.array(
                [np.interp(self.z_grid_for_cl, self.interpwingal[:, 0],
                           self.interpwingal[:, bin_j] + kerngalrsd_ell_j)
                 for kerngalrsd_ell_j in kerngalrsd_j])

        kernmag_i = prefactor_mag_col * \
            np.interp(self.z_grid_for_cl, self.interpwinmag[:, 0],
                      self.interpwinmag[:, bin_i])
        kernmag_j = prefactor_mag_col * \
            np.interp(self.z_grid_for_cl, self.interpwinmag[:, 0],
                      self.interpwinmag[:, bin_j])

        if self.multiply_bias_cl:
            bi = self.photobias[bin_i - 1]
            bj = self.photobias[bin_j - 1]
            kerngal_i = bi * kerngal_i
            kerngal_j = bj * kerngal_j

        self._evaluate_power_GC_phot(force_recompute=(not same_ells))

        pandw_galgal = kerngal_i * kerngal_j * self.power_gg_GC
        pandw_magmag = kernmag_i * kernmag_j * self.power_dd_GC
        pandw_galmag = (kernmag_i * kerngal_j + kernmag_j * kerngal_i) * \
            self.power_gd_GC

        pandwijk = pandw_magmag + pandw_galgal + pandw_galmag

        c_int_arr = self.Cl_generic_integrand(pandwijk)
        c_final = self.theory['c'] * \
            integrate.trapz(c_int_arr, self.z_grid_for_cl)

        return c_final

    def _evaluate_power_GC_phot(self, force_recompute=False):
        r"""Evaluate Power GC photometric

        Evaluates and store the matter power spectra for GC photometric.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_GC or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_GC_phot).T
            k_grid = (ell_col + 0.5) / self.f_K_z_grid

            P_gg = self.theory['Pgg_phot']
            P_dd = self.theory['Pmm_phot']
            P_gd = self.theory['Pgdelta_phot']
            self.power_gg_GC = P_gg(self.z_grid_for_cl, k_grid, grid=False)
            self.power_dd_GC = P_dd(self.z_grid_for_cl, k_grid, grid=False)
            self.power_gd_GC = P_gd(self.z_grid_for_cl, k_grid, grid=False)
            self._stored_GC = True

    def Cl_cross(self, ells, bin_i, bin_j):
        r"""Cl Cross

        Calculates angular power spectrum for cross-correlation
        between weak lensing and galaxy clustering, for the supplied bins:

        .. math::
            C_{ij}^{\rm LG} = C_{ij}^{\rm \gamma g}(\ell)
            + C_{ij}^{\rm Ig}(\ell) + C_{ij}^{\gamma \mu}(\ell)
            + C_{ij}^{\rm I\mu}(\ell)

        where :math:`\gamma` stands for gravitational shear, g for intrinsic
        number density fluctuations, :math:`\mu` for lensing magnification and
        I for intrinsic shear.
        The notation LG is related to the order of the contributions at the
        window functions level respectively in the i-th and j-th bins.

        .. math::
            C_{ij}^{\rm \gamma g}(\ell) =
            \left(\frac{(\ell+2)!}{(\ell-2)!}\right)^{1/2}
            \left(\frac{2}{2\ell+1}\right)^2
            \int {\rm d}z
            \frac{W_{i}^{\rm \gamma}(z)W_{j}^{\rm g}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm photo}_{g \delta}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm Ig}(\ell) =
            \left(\frac{(\ell+2)!}{(\ell-2)!}\right)^{1/2}
            \left(\frac{2}{2\ell+1}\right)^2
            \int {\rm d}z
            \frac{W_{i}^{\rm I}(z)W_{j}^{\rm g}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm{photo}}_{\rm gI}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm \gamma \mu}(\ell) =
            \left(\frac{(\ell+2)!}{(\ell-2)!}\right)^{1/2}
            \left(\frac{2}{2\ell+1}\right)^2
            \frac{\ell(\ell+1)}{(\ell + 1/2)^2}
            \int {\rm d}z
            \frac{W_{i}^{\rm \gamma}(z)W_{j}^{\rm \mu}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm photo}_{\delta \delta}
            \left[ k_{\ell}(z), z \right]

        .. math::
            C_{ij}^{\rm I\mu}(\ell) =
            \left(\frac{(\ell+2)!}{(\ell-2)!}\right)^{1/2}
            \left(\frac{2}{2\ell+1}\right)^2
            \frac{\ell(\ell+1)}{(\ell + 1/2)^2}
            \int {\rm d}z
            \frac{W_{i}^{\rm I}(z)W_{j}^{\rm \mu}(z)}
            {H(z)f_K^2\left[\tilde{r}(z)\right]}
            P^{\rm photo}_{\delta I}
            \left[ k_{\ell}(z), z \right]

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        bin_i: int
           Index of source tomographic bin. Tomographic bin
           indices start from 1
        bin_j: int
           Index of lens tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        Angular XCphot power spectrum: float
           Values of cross-correlation between weak lensing and
           galaxy clustering photometric angular power spectrum
        """

        same_ells = np.array_equal(ells, self.ells_XC)
        precomputed_shearIA = 'shearIA_XC' in self._prefactor_dict.keys()
        precomputed_mag = 'mag_XC' in self._prefactor_dict.keys()
        if not precomputed_shearIA or not precomputed_mag or not same_ells:
            self.set_prefactor(ells_XC=ells)

        prefactor_mag = self._prefactor_dict['mag_XC']
        prefactor_shearia = self._prefactor_dict['shearIA_XC']

        prefactor_mag_col = np.atleast_2d(prefactor_mag).T
        prefactor_shearia_col = np.atleast_2d(prefactor_shearia).T

        if self.add_RSD:
            kerngalrsd_j = [self._unpack_RSD_kernel(ell, bin_j)[0]
                            for ell in ells]
        else:
            kerngalrsd_j = np.zeros(len(ells))

        kern_i = prefactor_shearia_col * np.interp(
            self.z_grid_for_cl, self.interpwin[:, 0], self.interpwin[:, bin_i])
        kernia_i = prefactor_shearia_col * \
            np.interp(self.z_grid_for_cl, self.interpwinia[:, 0],
                      self.interpwinia[:, bin_i])
        kerngal_j = \
            np.array(
                [np.interp(self.z_grid_for_cl, self.interpwingal[:, 0],
                           self.interpwingal[:, bin_j] + kerngalrsd_ell_j)
                 for kerngalrsd_ell_j in kerngalrsd_j])
        kernmag_j = prefactor_mag_col * \
            np.interp(self.z_grid_for_cl, self.interpwinmag[:, 0],
                      self.interpwinmag[:, bin_j])

        if self.multiply_bias_cl:
            bj = self.photobias[bin_j - 1]
            kerngal_j = bj * kerngal_j

        self._evaluate_power_XC(force_recompute=(not same_ells))

        pandw_gd = kern_i * kerngal_j * self.power_gd_XC
        pandw_gi = kernia_i * kerngal_j * self.power_gi_XC
        pandw_dmag = kern_i * kernmag_j * self.power_dd_XC
        pandw_imag = kernia_i * kernmag_j * self.power_di_XC

        pandwijk = pandw_gd + pandw_gi + pandw_dmag + pandw_imag

        c_int_arr = self.Cl_generic_integrand(pandwijk)
        c_final = self.theory['c'] * \
            integrate.trapz(c_int_arr, self.z_grid_for_cl)
        c_final = c_final * (1 + self.multbias[bin_i - 1])

        return c_final

    def _evaluate_power_XC(self, force_recompute=False):
        r"""Evaluate Power XC

        Evaluates and store the matter power spectra for XC.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_XC or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_XC).T
            k_grid = (ell_col + 0.5) / self.f_K_z_grid

            P_gd = self.theory['Pgdelta_phot']
            P_gi = self.theory['Pgi_phot']
            P_dd = self.theory['Pmm_phot']
            P_di = self.theory['Pdeltai']
            self.power_gd_XC = P_gd(self.z_grid_for_cl, k_grid, grid=False)
            self.power_gi_XC = P_gi(self.z_grid_for_cl, k_grid, grid=False)
            self.power_dd_XC = P_dd(self.z_grid_for_cl, k_grid, grid=False)
            self.power_di_XC = P_di(self.z_grid_for_cl, k_grid, grid=False)
            self._stored_XC = True

    @staticmethod
    def _eval_prefactor_mag(ell):
        r"""Computes the magnification prefactor in Limber approximation.

        The prefactor is computed as follows.

        .. math::
             \ell (\ell + 1) / (\ell + 1/2)^2

        Parameters
        ----------
        ell: float or numpy.ndarray of float
           :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        Pre-factor: float or numpy.ndarray of float
           Value(s) of the prefactor at the given :math:`\ell`
        """
        return ell * (ell + 1) / (ell + 0.5)**2

    @staticmethod
    def _eval_prefactor_shearia(ell):
        r"""Computes the shearIA prefactor in Limber approximation.

        The prefactor is computed as follows.

        .. math::
            \sqrt{(\ell + 2)!/(\ell - 2)!} / (\ell + 1/2)^2

        Parameters
        ----------
        ell: float or numpy.ndarray of float
           :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        Pre-factor: float or numpy.ndarray of float
           Value(s) of the prefactor at the given :math:`\ell`
        """
        prefactor = np.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) / \
            (ell + 0.5)**2
        return prefactor

    @staticmethod
    def _eval_prefactor_l_0(ell):
        r"""RSD prefactor :math:`L_{0}` in Limber approximation.

        Calculates the :math:`L_{0}` RSD prefactor in the
        harmonic-space power spectrum of galaxy clustering.

        .. math::
            L_{0} = \frac{2 \ell^2 + 2 \ell -1}
            {(2 \ell - 1)(2 \ell + 3)}\\

        Parameters
        ----------
        ell: float or numpy.ndarray of float
            :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        L_0: float or numpy.ndarray of float
           Value(s) of the :math:`L_{0}` prefactor at the given :math:`\ell`
        """
        l_0 = (2 * ell ** 2 + 2 * ell - 1) / ((2 * ell - 1) * (2 * ell + 3))
        return l_0

    @staticmethod
    def _eval_prefactor_l_minus1(ell):
        r"""RSD prefactor :math:`L_{-1}` in Limber approximation

        Calculates the :math:`L_{-1}` RSD prefactor in the
        harmonic-space power spectrum of galaxy clustering.

        .. math::
            L_{-1} = -\frac{\ell(\ell - 1)}
            {(2 \ell -1)\sqrt{(2 \ell - 3)(2 \ell + 1)}}\\

        Parameters
        ----------
        ell: float or numpy.ndarray of float
            :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        L_minus1: float or numpy.ndarray of float
           Value(s) of the :math:`L_{-1}` prefactor at the given :math:`\ell`
        """
        l_minus1 = -ell * (ell - 1) / \
            ((2 * ell - 1) * np.sqrt((2 * ell - 3) * (2 * ell + 1)))
        return l_minus1

    @staticmethod
    def _eval_prefactor_l_plus1(ell):
        r"""RSD prefactor :math:`L_{+1}` in Limber approximation

        Calculates the :math:`L_{+1}` RSD prefactor in the
        harmonic-space power spectrum of galaxy clustering.

        .. math::
            L_{+1} = -\frac{(\ell + 1)(\ell + 2)}
            {(2\ell + 3)\sqrt{(2\ell + 1)(2 \ell + 5)}}\\

        Parameters
        ----------
        ell: float or numpy.ndarray of float
            :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        L_plus1: float or numpy.ndarray of float
           Value(s) of the :math:`L_{+1}` prefactor at the given :math:`\ell`
        """
        l_plus1 = -(ell + 1) * (ell + 2) / \
            ((2 * ell + 3) * np.sqrt((2 * ell + 1) * (2 * ell + 5)))
        return l_plus1

    def z_minus1(self, ell, r):
        r""":math:`z_{-1}(r)`

        Calculates the :math:`z_{-1}(r)` function needed for RSD.

        .. math::
            z_{-1}(r) = z \left[\frac{2 \ell -3}{ 2 \ell + 1}
            r \right] \\

        Parameters
        ----------
        ell: int
            :math:`\ell`-mode at which the function is evaluated
        r: float
            Comoving distance at which the redshift is evaluated

        Returns
        -------
        Redshift: float
            The redshift corresponding to the transverse comoving distance
        """
        z_r_interp = self.theory['z_r_func']
        ell_factor = (2 * ell - 3) / (2 * ell + 1)
        return z_r_interp(ell_factor * r)

    def z_plus1(self, ell, r):
        r""":math:`z_{+1}(r)`

        Calculates the :math:`z_{+1}(r)` function needed for RSD.

        .. math::
            z_{+1}(r) = z \left[\frac{2 \ell +5}{ 2 \ell + 1}
            r \right] \\

        Parameters
        ----------
        ell: int
            :math:`\ell`-mode at which the function is evaluated
        r: float
            Comoving distance at which the redshift is evaluated

        Returns
        -------
        Redshift: float
            The redshift corresponding to the transverse comoving distance
        """
        z_r_interp = self.theory['z_r_func']
        ell_factor = (2 * ell + 5) / (2 * ell + 1)
        return z_r_interp(ell_factor * r)

    def corr_func_3x2pt(self, obs, theta_deg, bin_i, bin_j):
        r"""Generic 3x2pt correlation function.

        Computes the specified 3x2pt configuration space correlation function,
        for the specified list of angular separations, and for the specified
        combination of tomographic bins.

        .. math::
            \xi_{ij}^\mathrm{AB}(\theta) = \frac{1}{2\pi} \
            \sum_{\ell=0}^{+\infty} \ell C_{ij}^\mathrm{AB}(\ell) \
            J_n(\ell\theta) \\

        where :math:`\mathrm{AB}` is to be selected from the list [LL, LG, GG]
        (the LL option has two different cases, :math:`\xi^\mathrm{LL,+}` and
        :math:`\xi^\mathrm{LL,-}`), and :math:`i,j` correspond to the indexes
        of the selected photometric bins.

        Parameters
        ----------
        obs: str
            Type of correlation function. It must be selected from the list
            ["Shear-Shear_plus", "Shear-Shear_minus", "Shear-Position",
            "Position-Position"]. The match is case-insensitive
        theta_deg: float or numpy.ndarray of float
            :math:`\theta` values at which the correlation function
            is computed. To be specified in degrees
        bin_i: int
            Index of first tomographic bin
        bin_j: int
            Index of second tomographic bin

        Returns
        -------
        Correlation function: numpy.ndarray of float
           Correlation function for the specified observable,
           angular separations, and bin combination
        """
        obs = obs.casefold()
        if obs == 'shear-shear_plus':
            cells_func = self.Cl_WL
            bessel_order = 0
        elif obs == 'shear-shear_minus':
            cells_func = self.Cl_WL
            bessel_order = 4
        elif obs == 'shear-position':
            cells_func = self.Cl_cross
            bessel_order = 2
        elif obs == 'position-position':
            cells_func = self.Cl_GC_phot
            bessel_order = 0
        else:
            raise ValueError('obs parameter must be selected from the '
                             'following list: ["Shear-Shear_plus", '
                             '"Shear-Shear_minus", "Shear-Position", '
                             '"Position-Position"]')

        if isinstance(theta_deg, np.ndarray):
            pass
        elif isinstance(theta_deg, list):
            theta_deg = np.array(theta_deg)
        elif isinstance(theta_deg, (float, int)):
            theta_deg = np.array([theta_deg])
        else:
            raise TypeError('theta_deg argument must be a int, float, list or '
                            'numpy.ndarray')

        theta_rad = np.deg2rad(theta_deg)
        xi_arr = np.empty(theta_deg.size)

        cells_int = cells_func(self.ells_int, bin_i, bin_j).flatten()
        cells_interp = interpolate.interp1d(self.ells_int, cells_int,
                                            kind='cubic')

        cells_dense = cells_interp(self.ells_dense)

        if not self.bessel_dict:
            bessel = np.array([jv(bessel_order, self.ells_dense * th)
                               for th in theta_rad])
        else:
            bessel = self.bessel_dict[bessel_order]

        for i in range(len(xi_arr)):
            xi_arr[i] = np.sum(self.ells_dense * cells_dense * bessel[i])

        xi_arr /= (2.0 * np.pi)

        return xi_arr
