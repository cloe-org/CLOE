"""CMBX module"""

# General imports
import numpy as np
from scipy import integrate


class CMBX:
    """Class for CMBX observable and XCorr with Euclid photometric observables

    (GC phot and WL)
    """

    def __init__(self, photo_ins):
        """Initialize

        Constructor of the class CMBX

        Parameters
        ----------
        cosmo_dic: dict
           cosmological dictionary from cosmo
        photo_ins: object
            initialized instance of the class Photo
        """
        self.photo_ins = photo_ins

        # Allows to select the spectra from CAMB directly (faster ?)
        self.use_camb_clkcmb = False

        self.ells_kCMB = []
        self.ells_kCMB_X_WL = []
        self.ells_kCMB_X_GC_phot = []
        self.ells_ISW_X_GC = []

    def cmbx_update(self, photo_ins):
        r"""Updates method.

        Method to update the theory class attribute to the passed photo_ins
        object.
        It will update the cosmology-dependent quantities based on the updates
        already done in the theory dictionnary of the photo_ins object.

        It also updates the redshift window to z_win_max, following the
        update done in cobaya_interface.py

        Parameters
        ----------
        photo_ins: object
            initialized instance of the class Photo
        """
        self.photo_ins = photo_ins
        # Taking the cosmology dictionary
        # from photo.py instance
        self.theory = self.photo_ins.theory

        # Get the comoving radial distance to CMB
        self.chicmb = self.theory['chistar']

        # Get the max comoving distance
        self.chi_zmax_phot = self.theory["r_z_func"](
            self.photo_ins.z_grid_for_cl[-1])

        # Defining the upscaled redshift sampling here instead
        # if init function.
        self.zmax = self.theory["z_win_max"][-1]
        z_wsamp_up = 100
        z_winterp_up = np.logspace(
            np.log10(photo_ins.z_winterp[-1]), np.log10(self.zmax), z_wsamp_up
        )

        self.z_winterp_large = np.unique(
            np.append(photo_ins.z_winterp, z_winterp_up))

        # Window for integrating CMB lensing XCorr with GC and WL
        self.interpwinkcmb_small = self.kCMB_window(self.photo_ins.z_winterp)

        # Window for auto powerspectrum of CMB lensing
        self.interpwinkcmb_large = self.kCMB_window(self.z_winterp_large)

        self.interpwinisw = np.zeros(shape=(self.photo_ins.z_wsamp, 2))
        self.interpwinisw[:, 0] = photo_ins.z_winterp
        self.interpwinisw[:, 1] = self.ISW_window(photo_ins.z_winterp)

        # holders for power spectra in Limber approximation
        self.power_dd_kCMB = []
        self.power_gd_kCMB_X_GC = []
        self.power_dd_kCMB_X_GC = []
        self.power_dd_kCMB_X_WL = []
        self.power_di_kCMB_X_WL = []
        self.power_gd_ISW_X_GC = []
        self.power_dd_ISW_X_GC = []
        self._stored_kCMB = False
        self._stored_kCMB_X_GC = False
        self._stored_kCMB_X_WL = False
        self._stored_ISW_X_GC = False

    def cmbx_set_prefactor(
        self, ells_kCMB_X_WL=None, ells_kCMB_X_GC_phot=None, ells_ISW_X_GC=None
    ):
        r"""Computes the prefactors for the kCMB_X_WL, kCMB_X_GC_phot

        and ISWxGC.

        The arrays of :math:`\ell` values provided as input are stored
        in the class instance, to be used in other private methods.
        The following prefactors are evaluated.

        * :obj:
          `shearIA_kCMB_X_WL` (from the ells_kCMB_X_WL input array): \
          :math:`\sqrt{(\ell+2)!/(\ell-2)!}/(\ell+1/2)^2`
        * :obj:
          `mag_kCMB_X_GC_phot` (from the ells_kCMB_X_GC_phot input array): \
          :math:`\ell(\ell+1)/(\ell+1/2)^2`

        The prefactors are evaluated using the functions
        :meth:`_eval_prefactor_mag()`, :meth:`_eval_prefactor_shearia()`,

        Parameters
        ----------
        ells_kCMB_X_WL: numpy.ndarray
           Array of :math:`\ell` values for the kCMB x WL
        ells_kCMB_X_GC_phot: numpy.ndarray
           Array of :math:`\ell` values for the kCMB x GC_phot
        ells_ISW_X_GC: numpy.ndarray
           Array of :math:`\ell` values for the ISW x GC_phot
        """

        # kCMB_X_WL
        if ells_kCMB_X_WL is not None:
            self.photo_ins._prefactor_dict["shearIA_kCMB_X_WL"] = (
                self.photo_ins._eval_prefactor_shearia(ells_kCMB_X_WL)
            )
            self.ells_kCMB_X_WL = ells_kCMB_X_WL

        # kCMB_X_GC_phot
        if ells_kCMB_X_GC_phot is not None:
            self.photo_ins._prefactor_dict["mag_kCMB_X_GC_phot"] = (
                self.photo_ins._eval_prefactor_mag(ells_kCMB_X_GC_phot)
            )
            self.ells_kCMB_X_GC_phot = ells_kCMB_X_GC_phot

        # ISW_X_GC
        if ells_ISW_X_GC is not None:
            self.photo_ins._prefactor_dict["ISW_X_GC"] = (
                self._eval_prefactor_ISW(ells_ISW_X_GC)
            )
            self.photo_ins._prefactor_dict["mag_ISW_X_GC"] = (
                self.photo_ins._eval_prefactor_mag(ells_ISW_X_GC)
            )
            self.ells_ISW_X_GC = ells_ISW_X_GC

    def kCMB_window(self, zinterp):
        r"""Window for kCMB

        Returns the window function of the CMB lensing convergence kappa.
        Comment: not implementing the modified gravity Sigma(k,z)
            function used in the Euclid WL window

        .. math::
            W^{\kappa \mathrm{CMB}}(z) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z) r(z)
            \left( 1 -\frac{r(z)}{r(z_{\mathrm{CMB}})} \right)

        Parameters
        ----------
        zinterp: 1-D numpy array defining the z values at which we
            evaluate the window function

        Returns
        -------
        W_val: numpy.ndarray
           1-D Numpy array of CMB lensing kernel values
           at specified scale for the redshifts defined in zinterp

        """

        H0_Mpc = self.theory["H0_Mpc"]
        O_m = self.theory["Omm"]
        W_kcmb = 1.5 * H0_Mpc * O_m * (1.0 + zinterp) * \
            (self.theory["r_z_func"](zinterp) / (1 / H0_Mpc)) * \
            (1.0 - (self.theory["r_z_func"](zinterp) / self.chicmb))

        return W_kcmb

    def ISW_window(self, z, k=0.005):
        r"""
        Calculates the ISW window function for the Cl_ISWxGC

        .. math::
            W^T(z) &= \frac{d}{dz} \left(\frac{D}{a}\right)(z) H(z)\\

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.
        D: Growth rate at a particular z and k
        Returns
        -------
        window_ISW: float
           Window function for galaxy clustering photometric
        """
        # SASept2022: Fixing the value of k for now so that
        # it can be filled in line 151 where we don't have the Limber
        # k values yet. At the moment of writing this, the growth
        # factor is scale independent

        if self.theory["use_gamma_MG"]:
            window_ISW = 3 * self.theory["H0"] ** 2 * \
                self.theory["Omm"] / self.theory["c"] ** 3 * \
                (-self.theory['f_z'](z) + 1)
        else:
            window_ISW = 3 * self.theory["H0"] ** 2 * \
                self.theory["Omm"] / self.theory["c"] ** 3 * \
                (-self.theory['f_z'](z) + 1)

        return window_ISW

    def Cl_kCMB(self, ells, nsteps=100):
        r"""Cl kCMB

        Calculates angular power spectrum of the CMB lensing convergence field.
        The integral is done over two redshift bins.

        - In the first half, from z=0 up to the zmax of the galaxy photo
        observables, we use the same integration as the one used for the
        galaxies, with same matter power spectrum

        - In the second half, from the zmax to the zcmb, the integral is
        done using the Weyl potential

        .. math::
            C^{\kappa_{CMB}}(\ell) = c \int \frac{dz}{H(z)r^2(z)}\
            W^{\kappa_\mathrm{CMB}}(z)^2 P^{\rm{photo}}_{\rm g\delta}\
            \left[ k_{\ell}(z), z \right]

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        nsteps: int
            Number of sampled comoving distance to integrate from z_max to
            z_cmb

        Returns
        -------
        c_final: float
           Value of CMB lensing angular power spectrum at ell.
        """

        same_ells = np.array_equal(ells, self.ells_kCMB)
        if not same_ells:
            self.ells_kCMB = ells

        if self.use_camb_clkcmb:
            return self.theory["Cl"]["pp"][int(ells)] * ells**2 * \
                (ells + 1.0) ** 2 / 4
        else:
            if self.zmax <= 200:
                raise ValueError("To get accurate CMB lensing autopower"
                                 " spectrum, zmax should be higher than 200,"
                                 f"but it's {self.zmax}.")

            # First part: integration from chi_zmax_phot to chicmb
            # Get the window function of CMB lensing
            chis = np.linspace(self.chi_zmax_phot, self.chicmb, nsteps)
            win = ((self.chicmb - chis) / (chis**2 * self.chicmb)) ** 2

            zs_arr = self.theory["z_r_func"](chis)
            ell_col = np.atleast_2d(self.ells_kCMB).T  # Multipole range

            # Limber approximation for calling the P(k)
            ks_arr = (ell_col + 0.5) / chis

            # Set to zero k values out of range of interpolation
            k_min = min(self.theory['k_win'])
            k_max = max(self.theory['k_win'])
            w = np.where((ks_arr < k_min) | (ks_arr > k_max), 0.0, 1.0)

            if self.theory["NL_flag_phot_matter"] == 0:
                P_dd = self.theory["Pk_weyl"]
            else:
                P_dd = self.theory["Pk_weyl_NL"]

            # Cl integration
            c_final = integrate.trapz(
                w * P_dd.P(zs_arr, ks_arr, grid=False) * win / ks_arr**4, chis
            )
            c_final *= (ells * (ells + 1)) ** 2

            # Second step: integrate from 0 to chi_zmax_phot
            kern_kcmb = np.interp(
                self.photo_ins.z_grid_for_cl,
                self.z_winterp_large,
                self.interpwinkcmb_large,
            )

            self._evaluate_power_kCMB(force_recompute=(not same_ells))

            pandw = kern_kcmb**2 * self.power_dd_kCMB

            c_int_arr = self.photo_ins.Cl_generic_integrand(pandw)
            c_final += self.theory["c"] * integrate.trapz(
                c_int_arr, self.photo_ins.z_grid_for_cl
            )

            return c_final

    def _evaluate_power_kCMB(self, force_recompute=False):
        r"""Evaluate Power kCMB

        Evaluates and store the matter power spectra for kCMB.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_kCMB or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_kCMB).T
            k_grid = (ell_col + 0.5) / self.photo_ins.f_K_z_grid

            P_dd = self.theory["Pmm_phot"]
            self.power_dd_kCMB = P_dd(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False)
            self._stored_kCMB = True

    def Cl_kCMB_X_GC_phot(self, ells, bin_i):
        r"""Cl kCMB X GC phot

        Calculates angular power spectrum for the cross correlation,
        between kCMB and photometric galaxy for the supplied bin.

        .. math::
            C_{i}^{\rm L \kappa_{CMB}}(\ell) = c \int \frac{dz}{H(z)r^2(z)}\
            W^{\kappa_\mathrm{CMB}}(z)\
            W_{j}^{\rm{G}}(z)P^{\rm{photo}}_{\rm g\delta}\
            \left[ k_{\ell}(z), z \right]

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        Angular cross-correlation between CMB lensing and
           galaxy clustering photometric: numpy.ndarray of float
        """

        same_ells = np.array_equal(ells, self.ells_kCMB_X_GC_phot)
        precomputed_mag = \
            "mag_kCMB_X_GC_phot" in self.photo_ins._prefactor_dict.keys()

        if not precomputed_mag or not same_ells:
            self.cmbx_set_prefactor(ells_kCMB_X_GC_phot=ells)

        prefactor_mag = self.photo_ins._prefactor_dict["mag_kCMB_X_GC_phot"]
        prefactor_mag_col = np.atleast_2d(prefactor_mag).T

        kern_kcmb = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.z_winterp,
            self.interpwinkcmb_small,
        )
        kern_i = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.interpwingal[:, 0],
            self.photo_ins.interpwingal[:, bin_i],
        )

        kernmag_i = prefactor_mag_col * np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.interpwinmag[:, 0],
            self.photo_ins.interpwinmag[:, bin_i],
        )

        if self.photo_ins.multiply_bias_cl:
            bi = self.photo_ins.photobias[bin_i - 1]
            kern_i = bi * kern_i

        self._evaluate_power_kCMB_X_GC_phot(force_recompute=(not same_ells))

        pandw_ik = kern_i * kern_kcmb * self.power_gd_kCMB_X_GC
        pandw_magk = kernmag_i * kern_kcmb * self.power_dd_kCMB_X_GC
        pandwijk = pandw_ik + pandw_magk

        c_int_arr = self.photo_ins.Cl_generic_integrand(pandwijk)
        c_final = self.theory["c"] * integrate.trapz(
            c_int_arr, self.photo_ins.z_grid_for_cl
        )

        return c_final

    def _evaluate_power_kCMB_X_GC_phot(self, force_recompute=False):
        r"""Evaluate Power kCMB X GC photometric

        Evaluates and store the matter power spectra for kCMB X GC photometric.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_kCMB_X_GC or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_kCMB_X_GC_phot).T
            k_grid = (ell_col + 0.5) / self.photo_ins.f_K_z_grid

            P_dd = self.theory["Pmm_phot"]
            P_gd = self.theory["Pgdelta_phot"]
            self.power_gd_kCMB_X_GC = P_gd(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False
            )
            self.power_dd_kCMB_X_GC = P_dd(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False
            )
            self._stored_kCMB_X_GC = True

    def Cl_kCMB_X_WL(self, ells, bin_i):
        r"""Cl kCMB X WL

        Calculates angular power spectrum for cross-correlation
        between weak lensing and CMB lensing, for the supplied bin.
        Includes intrinsic alignments.

        .. math::
            C_{i}^{\rm L \kappa_{CMB}}(\ell) = c \int \frac{dz}{H(z)r^2(z)}\
            \left\lbrace W_{i}^{\gamma}\left[ k_{\ell}(z), z \right ]\
            W^{\kappa_{CMB}}(z)P_{\rm \delta\delta}\
            \left[ k_{\ell}(z), z \right]\\
            +\,W_{i}^{\rm{IA}}(z)
            W^{\kappa_{CMB}}(z)P^{\rm{photo}}_{\rm \delta\rm{I}}\
            \left[ k_{\ell}(z), z \right] \right\rbrace \\

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        bin_i: int
           Index of tomographic bin. Tomographic bin
           indices start from 1.
        int_step: float
            Size of step for numerical integral over redshift.

        Returns
        -------
        Angular cross-correlation between CMB lensing and
           weak lensing: numpy.ndarray of float
        """

        same_ells = np.array_equal(ells, self.ells_kCMB_X_WL)
        precomputed_shearIA = (
            "shearIA_kCMB_X_WL" in self.photo_ins._prefactor_dict.keys()
        )

        if not precomputed_shearIA or not same_ells:
            self.cmbx_set_prefactor(ells_kCMB_X_WL=ells)

        prefactor_wl = self.photo_ins._prefactor_dict["shearIA_kCMB_X_WL"]

        kern_i = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.interpwin[:, 0],
            self.photo_ins.interpwin[:, bin_i],
        )
        kernia_i = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.interpwinia[:, 0],
            self.photo_ins.interpwinia[:, bin_i],
        )
        kern_kcmb = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.z_winterp,
            self.interpwinkcmb_small,
        )
        self._evaluate_power_kCMB_X_WL(force_recompute=(not same_ells))

        pandw_dk = kern_i * kern_kcmb * self.power_dd_kCMB_X_WL
        pandw_ik = kernia_i * kern_kcmb * self.power_di_kCMB_X_WL
        pandwijk = pandw_dk + pandw_ik

        c_int_arr = self.photo_ins.Cl_generic_integrand(pandwijk)
        c_final = prefactor_wl * self.theory["c"] * \
            integrate.trapz(c_int_arr, self.photo_ins.z_grid_for_cl)
        c_final *= 1 + self.photo_ins.multbias[bin_i - 1]

        return c_final

    def _evaluate_power_kCMB_X_WL(self, force_recompute=False):
        r"""Evaluate Power kCMB X WL

        Evaluates and store the matter power spectra for kCMB X WL.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_kCMB_X_WL or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_kCMB_X_WL).T
            k_grid = (ell_col + 0.5) / self.photo_ins.f_K_z_grid

            P_dd = self.theory["Pmm_phot"]
            P_di = self.theory["Pdeltai"]
            self.power_dd_kCMB_X_WL = P_dd(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False
            )
            self.power_di_kCMB_X_WL = P_di(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False
            )
            self._stored_kCMB_X_WL = True

    def Cl_ISWxGC(self, ells, bin_i):
        r"""Cl ISWXGC Probe

        Calculates angular power spectrum for cross-correlation of ISW and
        the galaxy clustering, for the supplied bin.

        .. math::
            C^{TG_{i}}(\ell)

        where T stands for the ISW window function.

        .. math::
            C^{\rm TG_{i}}(\ell) =
            \frac{3\Omega_{\rm m} H_0^2}{c^3\left ( \ell + 0.5\right )^2}
            \int {\rm d}z
            b(z)\frac{{\rm d}N}{{\rm d}z}H(z)D(z)\frac{\rm d}{\rm d z}
            \left ( \frac{D(z)}{a(z)} \right )
            P\left ( k = \frac{\ell + 0.5}{\chi(z)} \right )

        Parameters
        ----------
        ells: numpy.ndarray of float
            :math:`\ell`-modes at which :math:`C(\ell)` is evaluated.
        bin_i: int
           Index of first tomographic bin. Tomographic bin
           indices start from 1.

        Returns
        -------
        c_final: float
           Value of angular power spectrum for ISW cross-correlation
           with galaxy clustering.
        """

        same_ells = np.array_equal(ells, self.ells_ISW_X_GC)
        precomputed_ISW = "ISW_X_GC" in self.photo_ins._prefactor_dict.keys()
        precomputed_mag = \
            "mag_ISW_X_GC" in self.photo_ins._prefactor_dict.keys()

        if not precomputed_ISW or not precomputed_mag or not same_ells:
            self.cmbx_set_prefactor(ells_ISW_X_GC=ells)

        prefactor_ISW = self.photo_ins._prefactor_dict["ISW_X_GC"]
        prefactor_mag = self.photo_ins._prefactor_dict["mag_ISW_X_GC"]
        prefactor_mag_col = np.atleast_2d(prefactor_mag).T

        kerngal_i = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.interpwingal[:, 0],
            self.photo_ins.interpwingal[:, bin_i],
        )
        kernisw = np.interp(
            self.photo_ins.z_grid_for_cl,
            self.interpwinisw[:, 0],
            self.interpwinisw[:, 1],
        )
        kernmag_i = prefactor_mag_col * np.interp(
            self.photo_ins.z_grid_for_cl,
            self.photo_ins.interpwinmag[:, 0],
            self.photo_ins.interpwinmag[:, bin_i],
        )

        if self.photo_ins.multiply_bias_cl:
            bi = self.photo_ins.photobias[bin_i - 1]
            kerngal_i = bi * kerngal_i

        self._evaluate_power_ISW_X_GC(force_recompute=(not same_ells))
        pandw_iswgal = kerngal_i * kernisw * self.power_gd_ISW_X_GC
        pandw_iswmag = kernmag_i * kernisw * self.power_dd_ISW_X_GC

        pandwijk = pandw_iswgal + pandw_iswmag

        c_int_arr = self.photo_ins.Cl_generic_integrand(pandwijk)
        c_final = prefactor_ISW * self.theory["c"] * \
            integrate.trapz(c_int_arr, self.photo_ins.z_grid_for_cl)

        return c_final

    def _evaluate_power_ISW_X_GC(self, force_recompute=False):
        r"""Evaluate Power ISW X GC

        Evaluates and store the matter power spectra for ISW X GC.

        Parameters
        ----------
        force_recompute: bool
            Forces to recompute the power spectra.
        """

        if not self._stored_ISW_X_GC or force_recompute:
            # reshape ells to a column vector
            ell_col = np.atleast_2d(self.ells_ISW_X_GC).T
            k_grid = (ell_col + 0.5) / self.photo_ins.f_K_z_grid

            P_dd = self.theory["Pmm_phot"]
            P_gd = self.theory["Pgdelta_phot"]
            self.power_dd_ISW_X_GC = P_dd(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False
            )
            self.power_gd_ISW_X_GC = P_gd(
                self.photo_ins.z_grid_for_cl, k_grid, grid=False
            )
            self._stored_ISW_X_GC = True

    @staticmethod
    def _eval_prefactor_ISW(ell):
        r"""Computes the ISW prefactor in Limber approximation.

        The prefactor is computed as follows.

        .. math::
            \frac{1} / (\ell + 1/2)^2

        Parameters
        ----------
        ell: float or numpy.ndarray of float
           :math:`\ell`-mode(s) at which the prefactor is evaluated

        Returns
        -------
        Pre-factor: float or numpy.ndarray of float
           Value(s) of the prefactor at the given :math:`\ell`
        """
        prefactor = 1 / (ell + 0.5) ** 2
        return prefactor
