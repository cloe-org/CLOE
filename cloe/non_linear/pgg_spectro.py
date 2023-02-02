"""
module: pgg_spectro

Contains recipes for the spectroscopic galaxy x galaxy power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum


class Pgg_spectro_model(PowerSpectrum):
    r"""
    Class for computation of spectroscopic galaxy x galaxy power spectrum
    """

    def Pgg_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Pgg Spectro Def

        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Note: either wavenumber or mu_rsd must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.
        mu_rsd: float or numpy.ndarray
            cosine of the angle between the pair separation and
            the line of sight

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift, wavenumber and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.misc.istf_spectro_galbias(redshift)
        growth = self.theory['f_z'](redshift)
        power = self.theory['Pk_delta'].P(redshift, wavenumber)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def Pgdelta_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Pgdelta Spectro Def

        Computes the redshift-space galaxy-matter power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm g \delta}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z)+f(z, k)\mu_{k}^2][1+f(z, k)\mu_{k}^2]\
            P_{\rm \delta\delta}(z, k)\\

        Note: either wavenumber or mu_rsd must be a float (e.g. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber at which to evaluate the power spectrum.
        mu_rsd: float or numpy.ndarray
            cosine of the angle between the pair separation
            and the line of sight

        Returns
        -------
        pval: float or numpy.ndarray.
            Value of galaxy-matter power spectrum
            at a given redshift, wavenumber and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.misc.istf_spectro_galbias(redshift)
        growth = self.theory['f_z'](redshift)
        power = self.theory['Pk_delta'].P(redshift, wavenumber)
        pval = ((bias + growth * mu_rsd ** 2.0) *
                (1.0 + growth * mu_rsd ** 2.0)) * power
        return pval
