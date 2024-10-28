"""
pLL_phot

This module contains recipes for the lensing x lensing power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum


class PLL_phot_model(PowerSpectrum):
    r"""
    Class for computation of lensing x lensing power spectrum.
    """

    def Pmm_phot_def(self, redshift, wavenumber):
        r"""Pmm phot def.

        Computes the matter-matter power spectrum for the photometric probe.

        .. math::
            P_{\rm mm}(z, k) = P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum

        Returns
        -------
        Photometric matter-matter power spectrum: float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and wavenumber for photometric probes
        """
        pval = self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pmm_phot_halo(self, redshift, wavenumber):
        r"""Pmm phot halo.

        Computes the matter-matter power spectrum for the photometric probe
        using halo model based codes, with baryon effects (if selected) added
        as a boost, unless the halo model code already includes it.

        .. math::
            P_{\rm mm}(z, k) = P_{\rm \delta\delta}^{\rm NL}(z, k)\
            S_{\rm bar}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum

        Returns
        -------
        Halo model phot power spectrum: float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and wavenumber for photometric probes
        """
        pval = self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) * \
            self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]
        return pval

    def Pmm_phot_emu(self, redshift, wavenumber):
        r"""Pmm phot def.

        Computes the matter-matter power spectrum for the photometric probe
        with the EuclidEmu2 or the BACCO emulator, with baryon effects
        (if selected) added as a boost.

        .. math::
            P_{\rm mm}(z, k) = P_{\rm \delta\delta}(z, k)\
            B_{\rm NL}(z, k) S_{\rm bar}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum

        Returns
        -------
        Emulator phot matter-matter power spectrum: float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and wavenumber for photometric probes
        """
        pval = (self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])
        return pval

    def Pii_def(self, redshift, wavenumber):
        r"""Pii def.

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2P_{\rm \delta\delta}(z, k)

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to :obj:`numpy.ndarray`
        makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Intrinsic alignment power spectrum: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift)**2.0 * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pii_halo_nla(self, redshift, wavenumber):
        r"""Pii Def NLA

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum
        assuming the nonlinear alignment model. Uses halo model based codes
        for the matter power spectrum, with baryon effects (if selected) added
        as a boost, unless the halo model code already includes it.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2\
            P_{\rm \delta\delta}^{\rm NL}(z, k) S_{\rm bar}(z, k)

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to :obj:`numpy.ndarray`
        makes the code crash)


        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Halo model intrinsic alignment power spectrum: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift)**2.0 * \
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) * \
            self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]
        return pval

    def Pii_ee_halo_tatt(self, redshift, wavenumber):
        r"""Pii Def TATT

        Computes the intrinsic alignment (intrinsic-intrinsic) E-modes power
        spectrum assuming the tidal alignment tidal torquing model. Uses halo
        model based codes for the matter power spectrum.

        .. math::
            P_{\rm II}^{EE}(z, k) = C_{1}^{2}P_{\rm \delta\delta}^{\rm NL}\
            (z, k) S_{\rm bar}(z, k)+2C_{1}C_{1\delta}D(z)^{4}[\rm A_{0|0E}\
            (k)+\rm C_{0|0E}(k)]+C_{1\delta}^{2}D(z)^{4}\rm A_{0E|0E}(k)+\\
            C_{2}^{2}D(z)^{4}\rm A_{E2|E2}(k)+2C_{1}C_{2}D(z)^{4}[\rm \
            A_{0|E2}(k)+B_{0|E2}(k)]+2C_{1\delta}C_{2}D(z)^{4}D_{0E|E2}(k)

        Note: either redshift or wavenumber must be a float (ex. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and wavenumber
        """
        c1, c1d, c2 = self.misc.normalize_tatt_parameters(redshift)
        growth = self.theory['D_z_k_func'](redshift, wavenumber)

        pval = c1**2.0 * self.theory['Pk_halomodel_recipe'].\
            P(redshift, wavenumber) * \
            self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0] \
            + 2.0 * c1 * c1d * (growth**4) * \
            (self.theory['a00e'](wavenumber) + self.theory['c00e']
             (wavenumber)) + c1d**2.0 * (growth**4) * \
            self.theory['a0e0e'](wavenumber) + c2**2.0 * (growth**4) * \
            self.theory['ae2e2'](wavenumber) + 2.0 * c1 * c2 * (growth**4) * \
            (self.theory['a0e2'](wavenumber) +
             self.theory['b0e2'](wavenumber)) + 2.0 * c1d * c2 * \
            (growth**4) * self.theory['d0ee2'](wavenumber)

        return pval

    def Pii_emu_nla(self, redshift, wavenumber):
        r"""Pii emu NLA

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum
        assuming the nonlinear alignment model. Uses the EuclidEmu2 or the
        BACCO emulator for the nonlinear boost to the matter power spectrum,
        with baryon effects (if selected) added as a boost.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2\
            P_{\rm \delta\delta}(z, k) B_{\rm NL}(z, k)\
            S_{\rm bar}(z, k)

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to
        :obj:`numpy.ndarray` makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Emulator intrinsic alignment power spectrum: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and wavenumber
        """
        pval = (self.misc.fia(redshift)**2.0 *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

        return pval

    def Pii_ee_emu_tatt(self, redshift, wavenumber):
        r"""Pii emu TATT

        Computes the intrinsic alignment (intrinsic-intrinsic) E-modes power
        spectrum assuming the tidal alignment tidal torquing model. Uses the
        EuclidEmu2 or the BACCO emulator for the nonlinear boost to the
        matter power spectrum.

        .. math::
            P_{\rm II}^{EE}(z, k) = C_{1}^{2}P_{\rm \delta\delta}(z, k)\
            B_{\rm NL}(z, k) S_{\rm bar}(z, k)\
            +2C_{1}C_{1\delta}D(z)^{4}[\rm A_{0|0E}(k)+\rm C_{0|0E}(k)]\
            +C_{1\delta}^{2}D(z)^{4}\rm A_{0E|0E}(k)+\\C_{2}^{2}D(z)^{4}\rm\
            A_{E2|E2}(k)+2C_{1}C_{2}D(z)^{4}[\rm A_{0|E2}(k)+B_{0|E2}(k)]\
            +2C_{1\delta}C_{2}D(z)^{4}D_{0E|E2}(k)

        Note: either redshift or wavenumber must be a float (ex. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and wavenumber
        """
        c1, c1d, c2 = self.misc.normalize_tatt_parameters(redshift)
        growth = self.theory['D_z_k_func'](redshift, wavenumber)

        pval = (c1**2.0 * self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]) + \
            2.0 * c1 * c1d * (growth**4) * (self.theory['a00e'](wavenumber) +
                                            self.theory['c00e'](wavenumber)) +\
            c1d**2.0 * (growth**4) * self.theory['a0e0e'](wavenumber) + \
            c2**2.0 * (growth**4) * self.theory['ae2e2'](wavenumber) + 2.0 * \
            c1 * c2 * (growth**4) * (self.theory['a0e2'](wavenumber) +
                                     self.theory['b0e2'](wavenumber)) +\
            2.0 * c1d * c2 * (growth**4) * self.theory['d0ee2'](wavenumber)

        return pval

    def Pdeltai_def(self, redshift, wavenumber):
        r"""Pdeltai def.

        Computes the density-intrinsic power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]P_{\rm \delta\delta}(z, k)

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to
        :obj:`numpy.ndarray` makes the code crash).

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Density-intrinsic power spectrum: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pdeltai_halo_nla(self, redshift, wavenumber):
        r"""Pdeltai Def NLA

        Computes the density-intrinsic power spectrum assuming the nonlinear
        linear alignment model. Uses halo model based codes for the matter
        power spectrum, with baryon effects (if selected) added
        as a boost, unless the halo model code already includes it.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]\
            P_{\rm \delta\delta}^{\rm NL}(z, k) S_{\rm bar}(z, k)

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to
        :obj:`numpy.ndarray` makes the code crash).

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Halo model density-intrinsic power spectrum: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) * \
            self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]
        return pval

    def Pdeltai_halo_tatt(self, redshift, wavenumber):
        r"""Pdeltai Def TATT

        Computes the density-intrinsic power spectrum assuming the tidal
        alignment tidal torquing model model. Uses halo model based codes
        for the matter power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = C_{1}P_{\rm \delta\delta}(z, k)^{\rm NL}\
            (z, k) S_{\rm bar}(z, k)+C_{1\delta}D(z)^{4}[\rm A_{0|0E}(k)+\rm \
            C_{0|0E}(k)]+C_{2}D(z)^{4}[\rm A_{0|E2}(k)+B_{0|E2}(k)]

        Note: either redshift or wavenumber must be a float (ex. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        c1, c1d, c2 = self.misc.normalize_tatt_parameters(redshift)
        growth = self.theory['D_z_k_func'](redshift, wavenumber)

        pval = c1 * self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)\
            * self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]\
            + c1d * (growth**4) * (self.theory['a00e'](wavenumber) +
                                   self.theory['c00e'](wavenumber))\
            + c2 * (growth**4) * (self.theory['a0e2'](wavenumber) +
                                  self.theory['b0e2'](wavenumber))
        return pval

    def Pdeltai_emu_nla(self, redshift, wavenumber):
        r"""Pdeltai emu

        Computes the density-intrinsic power spectrum assuming the nonlinear
        alignment model. Uses the EuclidEmu2 or the BACCO emulator for the
        nonlinear boost to the matter power spectrum, with baryon effects (if
        selected) added as a boost.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]\
            P_{\rm \delta\delta}(z, k) B_{\rm NL}(z, k) S_{\rm bar}(z, k)

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to
        :obj:`numpy.ndarray` makes the code crash).

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Emulator density-intrinsic power spectrum: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = (self.misc.fia(redshift) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

        return pval

    def Pdeltai_emu_tatt(self, redshift, wavenumber):
        r"""Pdeltai TATT

        Computes the density-intrinsic power spectrum assuming the tidal
        alignment tidal torquing model. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = C_{1}P_{\rm \delta\delta}(z, k)\
            B_{\rm NL}(z, k) S_{\rm bar}(z, k)+C_{1\delta}D(z)^{4}\
            [\rm A_{0|0E}(k)+\rm C_{0|0E}(k)]+C_{2}D(z)^{4}[\rm \
            A_{0|E2}(k)+B_{0|E2}(k)]

        Note: either redshift or wavenumber must be a float (ex. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        c1, c1d, c2 = self.misc.normalize_tatt_parameters(redshift)
        growth = self.theory['D_z_k_func'](redshift, wavenumber)

        pval = (c1 * self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]) + \
            c1d * (growth**4) * (self.theory['a00e'](wavenumber) +
                                 self.theory['c00e'](wavenumber)) \
            + c2 * (growth**4) * \
            (self.theory['a0e2'](wavenumber) +
             self.theory['b0e2'](wavenumber))

        return pval
