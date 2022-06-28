"""
module: pgg_phot

Contains recipes for the photometric galaxy x galaxy power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum


class Pgg_phot_model(PowerSpectrum):
    r"""
    Class for computation of photometric galaxy x galaxy power spectrum
    """

    def Pgg_phot_def(self, redshift, k_scale):
        r"""Pgg Phot Def

        Computes the galaxy-galaxy power spectrum for the photometric probe.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and k-mode for galaxy
            clustering photometric
        """
        pval = ((self.misc.istf_phot_galbias(redshift) ** 2.0) *
                self.theory['Pk_delta'].P(redshift, k_scale))
        return pval
