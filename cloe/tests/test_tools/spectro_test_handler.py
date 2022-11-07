"""SPECTRO TEST HANDLER

This module contains a class with helper methods for spectro test cases.

"""


class SpectroTestParent:
    """Spectroscopic Test Parent

    Parent class to be used in tests of spectro.

    """

    def Pgg_spectro_def(self, redshift, k_scale, mu_rsd):
        r"""PGG Spectro Definition

        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float
            cosinus of the angle between the pair separation and
            the line of sight

        Returns
        -------
        pval: float
            Value of galaxy-galaxy power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy cclustering spectroscopic

        """
        bias = self.istf_spectro_galbias(redshift)
        growth = self.test_dict['f_z'](redshift)
        power = self.test_dict['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval
