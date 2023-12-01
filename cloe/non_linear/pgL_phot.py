"""
pgL_phot

This module contains  the recipes for the photometric
galaxy x lensing power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum


class PgL_phot_model(PowerSpectrum):
    r"""
    Class for computation of photometric galaxy x lensing power spectrum.
    """

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
        for the matter power spectrum, with baryon effects (if selected) added
        as a boost, unless the halo model code already includes it.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)\
            P_{\rm \delta\delta}^{\rm NL}(z, k) S_{\rm bar}(z, k)\\

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
            self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) * \
            self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0]
        return pval

    def Pgi_phot_emu(self, redshift, wavenumber):
        r"""Pgi Phot emu

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and nonlinear alignment model. Uses the EuclidEmu2 or
        the BACCO emulator for the nonlinear boost to the matter power
        spectrum, with baryon effects (if selected) added as a boost.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)\
            P_{\rm \delta\delta}(z, k) B_{\rm NL}(z, k) S_{\rm bar}(z, k)\\

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
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

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

    def Pgdelta_phot_def(self, redshift, wavenumber):
        r"""Pgdelta phot def.

        Computes the galaxy-matter power spectrum for the photometric probe.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)] P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Photometric galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """
        pval = (self.misc.istf_phot_galbias(redshift) *
                self.theory['Pk_delta'].P(redshift, wavenumber))
        return pval

    def Pgdelta_phot_halo(self, redshift, wavenumber):
        r"""Pgdelta phot halo.

        Computes the galaxy-matter power spectrum for the photometric probe
        assuming a linear bias model. Uses halo model based codes for the
        matter power spectrum, with baryon effects (if selected) added
        as a boost, unless the halo model code already includes it.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)] P_{\rm \delta\delta}^{\rm NL}(z, k)\
            S_{\rm bar}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Halo model phot galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """
        pval = (self.misc.istf_phot_galbias(redshift) *
                self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])
        return pval

    def Pgdelta_phot_emu(self, redshift, wavenumber):
        r"""Pgdelta phot emu.

        Computes the galaxy-matter power spectrum for the photometric probe
        assuming a linear bias model. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum, with
        baryon effects (if selected) added as a boost.

        .. math::
            P_{\rm {\rm g}\delta}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)] P_{\rm \delta\delta}(z, k)\
            B_{\rm NL}(z, k) S_{\rm bar}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Emulator phot galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """

        pval = (self.misc.istf_phot_galbias(redshift) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

        return pval
