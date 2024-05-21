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
            Wavenumber(s) at which to evaluate the power spectrum

        Returns
        -------
        Photometric galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """
        pval = ((self.theory['b1_inter'](redshift) ** 2.0) *
                self.theory['Pk_delta'].P(redshift, wavenumber))
        return pval

    def Pgg_phot_halo(self, redshift, wavenumber):
        r"""Pgg phot halo.

        Computes the galaxy-galaxy power spectrum for the photometric probe
        assuming a linear bias model. Uses halo model based codes for the
        matter power spectrum, with baryon effects (if selected) added
        as a boost, unless the halo model code already includes it.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}^{\rm NL}(z, k)\
            S_{\rm bar}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber(s) at which to evaluate the  power spectrum

        Returns
        -------
        Halo model phot galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """
        pval = ((self.theory['b1_inter'](redshift) ** 2.0) *
                self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])
        return pval

    def Pgg_phot_emu(self, redshift, wavenumber):
        r"""Pgg phot emu.

        Computes the galaxy-galaxy power spectrum for the photometric probe
        assuming a linear bias model. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum, with
        baryon effects (if selected) added as a boost.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_{\rm g}^{\rm photo}(z)]^2 P_{\rm \delta\delta}(z, k)\
            B_{\rm NL}(z, k) S_{\rm bar}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber(s) at which to evaluate the power spectrum

        Returns
        -------
        Emulator phot galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """
        pval = ((self.theory['b1_inter'](redshift) ** 2.0) *
                self.theory['Pk_delta'].P(redshift, wavenumber) *
                self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
                self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

        return pval

    def _add_NLbias_contributions_gg(self, Pnl, redshift, wavenumber):
        r"""Add NL bias contributions

        Gets the non-linear bias contributions, combines them with
        the corresponding bias parameters and adds it to any non-linear
        matter power spectrum it is given to return the galaxy-galaxy
        power spectrum.

        Parameters
        ----------
        Pnl: float or numpy.ndarray
            Value of non-linear matter power spectrum
            at a given redshift and wavenumber.
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """

        b1 = self.theory['b1_inter'](redshift)
        b2 = self.theory['b2_inter'](redshift)
        bG2 = self.theory['bG2_inter'](redshift)
        bG3 = self.theory['bG3_inter'](redshift)

        Pb1b2 = self.nonlinear_dic['Pb1b2_kz'](redshift, wavenumber,
                                               grid=False)
        Pb1bG2 = self.nonlinear_dic['Pb1bG2_kz'](redshift, wavenumber,
                                                 grid=False)
        Pb2b2 = self.nonlinear_dic['Pb2b2_kz'](redshift, wavenumber,
                                               grid=False)
        Pb2bG2 = self.nonlinear_dic['Pb2bG2_kz'](redshift, wavenumber,
                                                 grid=False)
        PbG2bG2 = self.nonlinear_dic['PbG2bG2_kz'](redshift, wavenumber,
                                                   grid=False)
        PZ1bG3 = self.nonlinear_dic['PZ1bG3_kz'](redshift, wavenumber,
                                                 grid=False)
        PZ1bG2 = self.nonlinear_dic['PZ1bG2_kz'](redshift, wavenumber,
                                                 grid=False)

        return ((b1 ** 2.0) * Pnl +
                (b1 * b2) * Pb1b2 +
                (b1 * bG2) * (Pb1bG2 + PZ1bG2) +
                (b1 * bG3) * PZ1bG3 +
                (b2 ** 2.0) * Pb2b2 +
                (b2 * bG2) * Pb2bG2 +
                (bG2 ** 2.0) * PbG2bG2)

    def Pgg_phot_halo_NLbias(self, redshift, wavenumber):
        r"""Pgg Phot Halo NLbias

        Computes the galaxy-galaxy power spectrum for the photometric probe
        with non-linear bias contributions computed with Fast-PT from the
        linear power spectrum. Uses halo model based codes for the matter
        power spectrum. This include terms in b1 (linear bias),
        b2 (quadratic bias), bG2 (quadratic non-local bias)
        and bG3 (cubic non-local bias).

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) =\
                &[b_{1,{\rm g}}^{\rm photo}(z)]^2 P_{\rm \delta\delta}^{\rm NL}
                (z, k) +\\
                &[b_{1,{\rm g}}^{\rm photo}(z) b_{2,{\rm g}}^{\rm photo}(z)]
                P_{b_1b_2}(z, k)+\\
                &[b_{1,{\rm g}}^{\rm photo}(z) b_{{\mathcal{G}_2},{\rm g}}^{\rm
                photo}(z)] P_{b_1b_{\mathcal{G}_2}}(z, k) +\\
                &[b_{1,{\rm g}}^{\rm photo}(z) b_{\Gamma_3,{\rm g}}^{\rm photo}
                (z)] P_{b_1b_{\Gamma_3}}(z, k) +\\
                &[b_{2,{\rm g}}^{\rm photo}(z)]^2 P_{b_2b_2}(z, k) +\\
                &[b_{2,{\rm g}}^{\rm photo}(z) b_{{\mathcal{G}_2},{\rm g}}^{\rm
                photo}(z)] P_{b_2b_{\mathcal{G}_2}}(z, k) +\\
                &[b_{{\mathcal{G}_2},{\rm g}}^{\rm photo}(z)]^2
                P_{b_{\mathcal{G}_2}b_{\mathcal{G}_2}}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """

        Pmm = (self.theory['Pk_halomodel_recipe'].P(redshift, wavenumber) *
               self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

        pval = self._add_NLbias_contributions_gg(Pmm, redshift, wavenumber)

        return pval

    def Pgg_phot_emu_NLbias(self, redshift, wavenumber):
        r"""Pgg Phot Emu NLbias

        Computes the galaxy-galaxy power spectrum for the photometric probe
        with non-linear bias contributions computed with Fast-PT from the
        linear power spectrum. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum.
        This include terms in b1 (linear bias),
        b2 (quadratic bias), bG2 (quadratic non-local bias)
        and bG3 (cubic non-local bias).

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) =\
                &[b_{1,{\rm g}}^{\rm photo}(z)]^2 P_{\rm \delta\delta}^{\rm NL}
                (z, k) +\\
                &[b_{1,{\rm g}}^{\rm photo}(z) b_{2,{\rm g}}^{\rm photo}(z)]
                P_{b_1b_2}(z, k)+\\
                &[b_{1,{\rm g}}^{\rm photo}(z) b_{{\mathcal{G}_2},{\rm g}}^{\rm
                photo}(z)] P_{b_1b_{\mathcal{G}_2}}(z, k) +\\
                &[b_{1,{\rm g}}^{\rm photo}(z) b_{\Gamma_3,{\rm g}}^{\rm photo}
                (z)] P_{b_1b_{\Gamma_3}}(z, k) +\\
                &[b_{2,{\rm g}}^{\rm photo}(z)]^2 P_{b_2b_2}(z, k) +\\
                &[b_{2,{\rm g}}^{\rm photo}(z) b_{{\mathcal{G}_2},{\rm g}}^{\rm
                photo}(z)] P_{b_2b_{\mathcal{G}_2}}(z, k) +\\
                &[b_{{\mathcal{G}_2},{\rm g}}^{\rm photo}(z)]^2
                P_{b_{\mathcal{G}_2}b_{\mathcal{G}_2}}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the  power spectrum.

        Returns
        -------
        pval:  float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift and wavenumber for galaxy
            clustering photometric
        """

        Pmm = (self.theory['Pk_delta'].P(redshift, wavenumber) *
               self.nonlinear_dic['NL_boost'](redshift, wavenumber)[0] *
               self.nonlinear_dic['Bar_boost'](redshift, wavenumber)[0])

        pval = self._add_NLbias_contributions_gg(Pmm, redshift, wavenumber)

        return pval
