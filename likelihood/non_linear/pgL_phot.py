"""
module: pgL_phot

Contains recipes for the photometric galaxy x lensing power spectrum.
"""

from likelihood.non_linear.power_spectrum import PowerSpectrum


class PgL_phot_model(PowerSpectrum):
    r"""
    Class for computation of photometric galaxy x lensing power spectrum
    """

    def Pgd_phot_def(self, redshift, k_scale):
        r"""Pgd Phot Def

        Computes the galaxy-matter power spectrum for the photometric probe.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_g^{\rm photo}(z)] P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float or list or numpy.ndarray
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and k-mode for galaxy clustering
            photometric
        """
        pval = (self.misc.istf_phot_galbias(redshift) *
                self.theory['Pk_delta'].P(redshift, k_scale))
        return pval
