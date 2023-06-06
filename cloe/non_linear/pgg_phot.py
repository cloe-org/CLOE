"""
pgg_phot

This module contains the recipes for the photometric
galaxy x galaxy power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum


class Pgg_phot_model(PowerSpectrum):
    r"""
    Class for computation of photometric galaxy x galaxy power spectrum.
    """

    def Pgg_phot_def(self, redshift, wavenumber):
        r"""Pgg phot def.

        Computes the galaxy-galaxy power spectrum for the photometric probe.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Photometric galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """
        pval = ((self.misc.istf_phot_galbias(redshift) ** 2.0) *
                self.theory['Pk_delta'].P(redshift, wavenumber))
        return pval

    def Pgg_phot_halo(self, redshift, wavenumber):
        r"""Pgg phot halo.

        Computes the galaxy-galaxy power spectrum for the photometric probe
        assuming a linear bias model. Uses halo model based codes for the
        matter power spectrum.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}^{\rm NL}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum

        Returns
        -------
        Halo model phot galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """
        pval = ((self.misc.istf_phot_galbias(redshift) ** 2.0) *
                self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber))
        return pval

    def Pgg_phot_emu(self, redshift, wavenumber):
        r"""Pgg phot emu.

        Computes the galaxy-galaxy power spectrum for the photometric probe
        assuming a linear bias model. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}^{\rm NL}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum

        Returns
        -------
        Emulator phot galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """
        pval = ((self.misc.istf_phot_galbias(redshift) ** 2.0) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0])

        return pval
