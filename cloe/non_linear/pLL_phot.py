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
        using halo model based codes.

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
        pval = self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pmm_phot_emu(self, redshift, wavenumber):
        r"""Pmm phot def.

        Computes the matter-matter power spectrum for the photometric probe
        with the EuclidEmu2 or the BACCO emulator.

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
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])
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

    def Pii_halo(self, redshift, wavenumber):
        r"""Pii def.

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum
        assuming the nonlinear alignment model. Uses halo model based codes
        for the matter power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pii_emu(self, redshift, wavenumber):
        r"""Pii emu.

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum
        assuming the nonlinear alignment model. Uses the EuclidEmu2 or the
        BACCO emulator for the nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

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

    def Pdeltai_halo(self, redshift, wavenumber):
        r"""Pdeltai def halo model.

        Computes the density-intrinsic power spectrum assuming the nonlinear
        linear alignment model. Uses halo model based codes for the matter
        power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pdeltai_emu(self, redshift, wavenumber):
        r"""Pdeltai emu.

        Computes the density-intrinsic power spectrum assuming the nonlinear
        alignment model. Uses the EuclidEmu2 or the BACCO emulator for the
        nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]\
            P_{\rm \delta\delta}^{\rm NL}(z, k)

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
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval

    def Pgi_phot_def(self, redshift, wavenumber):
        r"""Pgi phot def.

        Computes the photometric galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_{\rm g}^{\rm photo}(z)P_{\rm \delta\delta}(z, k)\\

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
        Photometric galaxy-intrinsic power spectrum: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_phot_galbias(redshift) * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval

    def Pgi_phot_halo(self, redshift, wavenumber):
        r"""Pgi phot def halo.

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and nonlinear alignment model. Uses halo model based codes
        for the matter power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_{\rm g}^{\rm photo}(z)\
            P_{\rm \delta\delta}^{\rm NL}(z, k)\\

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
        Halo model phot galaxy-intrinsic power spectrum: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_phot_galbias(redshift) * \
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber)
        return pval

    def Pgi_phot_emu(self, redshift, wavenumber):
        r"""Pgi phot emu.

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and nonlinear alignment model. Uses the EuclidEmu2 or
        the BACCO emulator for the nonlinear boost to the matter power
        spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_{\rm g}^{\rm photo}(z)\
            P_{\rm \delta\delta}^{\rm NL}(z, k)\\

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
        Emulator phot galaxy-intrinsic power spectrum: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """

        pval = (self.misc.fia(redshift) *
                self.misc.istf_phot_galbias(redshift) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval

    def Pgi_spectro_def(self, redshift, wavenumber):
        r"""Pgi spectro def.

        Computes the spectroscopic galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm spectro}(z, k) &=\
            [f_{\rm IA}(z)]b_{\rm g}^{\rm spectro}(z)\
            P_{\rm \delta\delta}(z, k)\\

        Note: either redshift or wavenumber must be a float
        (e.g. simultaneously setting both of them to
        :obj:`numpy.ndarray` makes the code crash).

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum.

        Returns
        -------
        Spectroscopic galaxy-intrinsic power spectrum: float or numpy.ndarray
            Value of spectroscopic galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_spectro_galbias(redshift) * \
            self.theory['Pk_delta'].P(redshift, wavenumber)
        return pval
