"""
module: pLL_phot

Contains recipes for the lensing x lensing power spectrum.
"""

from likelihood.non_linear.power_spectrum import PowerSpectrum


class PLL_phot_model(PowerSpectrum):
    r"""
    Class for computation of lensing x lensing power spectrum
    """

    def Pmm_phot_def(self, redshift, k_scale):
        r"""Pmm Phot Def

        Computes the matter-matter power spectrum for the photometric probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of matter-matter power spectrum
            at a given redshift and k-mode for photometric probes
        """
        pval = self.theory['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pii_def(self, redshift, k_scale):
        r"""Pii Def

        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2P_{\rm \delta\delta}(z, k)

        Note: either redshift or k_scale must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of intrinsic alignment power spectrum
            at a given redshift and k-mode
        """
        pval = self.misc.fia(redshift)**2.0 * \
            self.theory['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pdeltai_def(self, redshift, k_scale):
        r"""Pdeltai Def

        Computes the density-intrinsic power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]P_{\rm \delta\delta}(z, k)

        Note: either redshift or k_scale must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of density-intrinsic power spectrum
            at a given redshift and k-mode
        """
        pval = self.misc.fia(redshift) * \
            self.theory['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_phot_def(self, redshift, k_scale):
        r"""Pgi Phot Def

        Computes the photometric galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)P_{\rm \delta\delta}(z, k)\\

        Note: either redshift or k_scale must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and k-mode
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_phot_galbias(redshift) * \
            self.theory['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_spectro_def(self, redshift, k_scale):
        r"""Pgi Spectro Def

        Computes the spectroscopic galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm spectro}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm spectro}(z)P_{\rm \delta\delta}(z, k)\\

        Note: either redshift or k_scale must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of spectroscopic galaxy-intrinsic power spectrum
            at a given redshift and k-mode
        """
        pval = self.misc.fia(redshift) * \
            self.misc.istf_spectro_galbias(redshift) * \
            self.theory['Pk_delta'].P(redshift, k_scale)
        return pval
