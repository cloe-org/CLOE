"""
module: pgg_spec

Contains recipes for the spectroscopic galaxy x galaxy power spectrum.
"""

# Global
import numpy as np
from likelihood.non_linear.power_spectrum import PowerSpectrum


class Pgg_spec_model(PowerSpectrum):
    r"""
    Class for computation of spectroscopic galaxy x galaxy power spectrum
    """

    def Pgg_spec_def(self, redshift, k_scale, mu_rsd):
        r"""Pgg Spec Def

        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spec}(z, k) &=\
            [b_{\rm g}^{\rm spec}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Note: either k_scale or mu_rsd must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float or numpy.ndarray
            cosine of the angle between the pair separation and
            the line of sight

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.misc.istf_spec_galbias(redshift)
        growth = self.theory['f_z'](redshift)
        power = self.theory['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def Pgd_spec_def(self, redshift, k_scale, mu_rsd):
        r"""Pgd Spec Def

        Computes the redshift-space galaxy-matter power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm g \delta}^{\rm spec}(z, k) &=\
            [b_{\rm g}^{\rm spec}(z) + f(z, k)\mu_{k}^2][1 + f(z, k)\mu_{k}^2]\
            P_{\rm \delta\delta}(z, k)\\

        Note: either k_scale or mu_rsd must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float or numpy.ndarray
            cosine of the angle between the pair separation
            and the line of sight

        Returns
        -------
        pval: float or numpy.ndarray.
            Value of galaxy-matter power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.misc.istf_spec_galbias(redshift)
        growth = self.theory['f_z'](redshift)
        power = self.theory['Pk_delta'].P(redshift, k_scale)
        pval = ((bias + growth * mu_rsd ** 2.0) *
                (1.0 + growth * mu_rsd ** 2.0)) * power
        return pval
