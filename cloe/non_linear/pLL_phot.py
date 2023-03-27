"""
module: pLL_phot

Contains recipes for the lensing x lensing power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum


class PLL_phot_model(PowerSpectrum):
    r"""
    Class for computation of lensing x lensing power spectrum
    """

    def Pmm_phot_def(self, redshift, wavenumber):
        r"""Pmm Phot Def

        Computes the matter-matter power spectrum for the photometric probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and wavenumber for photometric probes
        """
        pval = self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pmm_phot_halo(self, redshift, wavenumber):
        r"""Pmm Phot Halo

        Computes the matter-matter power spectrum for the photometric probe
        using halo model based codes.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and wavenumber for photometric probes
        """
        pval = self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pmm_phot_emu(self, redshift, wavenumber):
        r"""Pmm Phot Def

        Computes the matter-matter power spectrum for the photometric probe
        with the EuclidEmu2 or the BACCO emulator.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and wavenumber for photometric probes
        """
        pval = (self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])
        return pval

    def Pii_def(self, redshift, wavenumber):
        r"""Pii Def

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2P_{\rm \delta\delta}(z, k)

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
        pval = self.misc.fia(redshift)**2.0 * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pii_halo(self, redshift, wavenumber):
        r"""Pii Def

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum
        assuming the nonlinear alignment model. Uses halo model based codes
        for the matter power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
        pval = self.misc.fia(redshift)**2.0 * \
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pii_emu(self, redshift, wavenumber):
        r"""Pii emu

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum
        assuming the nonlinear alignment model. Uses the EuclidEmu2 or the
        BACCO emulator for the nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
        pval = (self.misc.fia(redshift)**2.0 *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval

    def Pdeltai_def(self, redshift, wavenumber):
        r"""Pdeltai Def

        Computes the density-intrinsic power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]P_{\rm \delta\delta}(z, k)

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
        pval = self.misc.fia(redshift) * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pdeltai_halo(self, redshift, wavenumber):
        r"""Pdeltai Def

        Computes the density-intrinsic power spectrum assuming the nonlinear
        linear alignment model. Uses halo model based codes for the matter
        power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
        pval = self.misc.fia(redshift) * \
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pdeltai_emu(self, redshift, wavenumber):
        r"""Pdeltai emu

        Computes the density-intrinsic power spectrum assuming the nonlinear
        alignment model. Uses the EuclidEmu2 or the BACCO emulator for the
        nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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

        pval = (self.misc.fia(redshift) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval

    def Pgi_phot_def(self, redshift, wavenumber):
        r"""Pgi Phot Def

        Computes the photometric galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)P_{\rm \delta\delta}(z, k)\\

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
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_phot_galbias(redshift) * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pgi_phot_halo(self, redshift, wavenumber):
        r"""Pgi Phot Def

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and nonlinear alignment model. Uses halo model based codes
        for the matter power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)\
            P_{\rm \delta\delta}^{\rm NL}(z, k)\\

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
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_phot_galbias(redshift) * \
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pgi_phot_emu(self, redshift, wavenumber):
        r"""Pgi Phot emu

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and nonlinear alignment model. Uses the EuclidEmu2 or
        the BACCO emulator for the nonlinear boost to the matter power
        spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)\
            P_{\rm \delta\delta}^{\rm NL}(z, k)\\

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
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """

        pval = (self.misc.fia(redshift) *
                self.misc.istf_phot_galbias(redshift) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval

    def Pgi_spectro_def(self, redshift, wavenumber):
        r"""Pgi Spectro Def

        Computes the spectroscopic galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm spectro}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm spectro}(z)P_{\rm \delta\delta}(z, k)\\

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
            Value of spectroscopic galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_spectro_galbias(redshift) * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval
