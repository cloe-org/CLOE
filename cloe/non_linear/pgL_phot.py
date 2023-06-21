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
        matter power spectrum.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)] P_{\rm \delta\delta}^{\rm NL}(z, k)\\

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
                self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber))
        return pval

    def Pgdelta_phot_emu(self, redshift, wavenumber):
        r"""Pgdelta phot emu.

        Computes the galaxy-matter power spectrum for the photometric probe
        assuming a linear bias model. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm {\rm g}\delta}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)] P_{\rm \delta\delta}^{\rm NL}(z, k)\\

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
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval
