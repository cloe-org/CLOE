"""
pgL_phot

This module contains  the recipes for the photometric
galaxy x lensing power spectrum.
"""

from warnings import warn

from cloe.non_linear.power_spectrum import PowerSpectrum
from numpy import sqrt


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
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber(s)
        """
        bterm = self.misc.fia(redshift) * self.theory["b1_inter"](redshift)
        if self.theory["GC_use_cold_matter_tracer"]:
            pterm = sqrt(
                self.theory["Pk_cb"].P(redshift, wavenumber)
                * self.theory["Pk_delta"].P(redshift, wavenumber)
            )
        else:
            pterm = self.theory["Pk_delta"].P(redshift, wavenumber)
        return bterm * pterm

    def Pgi_phot_halo_nla(self, redshift, wavenumber):
        r"""Pgi Phot Def NLA

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
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        bterm = self.misc.fia(redshift) * self.theory["b1_inter"](redshift)
        pterm = (
            self.theory["Pk_halomodel_recipe"].P(redshift, wavenumber)
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )
        if self.theory["GC_use_cold_matter_tracer"]:
            self.nonlinear_neutrino_approx.wrap_linear_neutrino_approx(
                self.theory["Pk_halomodel_recipe"],
                self.nonlinear_dic["Bar_boost"],
            )
            pterm = sqrt(self.nonlinear_neutrino_approx(redshift, wavenumber) * pterm)

        pval = bterm * pterm
        return pval

    def Pgi_phot_halo_tatt(self, redshift, wavenumber):
        r"""Pgi Phot Def TATT

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and the tidal alignment tidal torquing intrinsic alignment
        model. Uses halo model based codes for the matter power spectrum, with
        baryon effects (if selected) added as a boost, unless the halo model
        code already includes it.

        .. math::
            P_{\rm G I}^{\rm photo}(z, k) = b_g^{\rm photo}(z) \
            \left [ C_{1}P_{\rm \delta\delta}^{\rm NL}(z, k) S_{\rm bar}\
            (z, k)+C_{1\delta}D(z)^{4}[\rm A_{0|0E}(k)+\rm C_{0|0E}(k)] \
            +C_{2}D(z)^{4}[\rm A_{0|E2}(k)+B_{0|E2}(k)] \right ]

        Note: either redshift or wavenumber must be a float (ex. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        c1, c1d, c2 = self.misc.normalize_tatt_parameters(redshift)
        growth = self.theory["D_z_k_func"](redshift, wavenumber)

        pterm = (
            self.theory["Pk_halomodel_recipe"].P(redshift, wavenumber)
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )

        bterm = self.theory["b1_inter"](redshift)
        if self.theory["GC_use_cold_matter_tracer"]:
            # we could rescale bterm like for the bNL case
            # but the tidal field terms do not correspond to Pcb in this case
            raise ValueError(
                "The neutrino galaxy bias implementation does"
                + "not exist for the TATT initial alignment model"
            )

        pval = bterm * (
            c1 * pterm
            + c1d
            * (growth**4)
            * (self.theory["a00e"](wavenumber) + self.theory["c00e"](wavenumber))
            + c2
            * (growth**4)
            * (self.theory["a0e2"](wavenumber) + self.theory["b0e2"](wavenumber))
        )

        return pval

    def Pgi_phot_emu_nla(self, redshift, wavenumber):
        r"""Pgi Phot emu NLA

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
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        bterm = self.misc.fia(redshift) * self.theory["b1_inter"](redshift)
        pterm = (
            self.theory["Pk_delta"].P(redshift, wavenumber)
            * self.nonlinear_dic["NL_boost"](redshift, wavenumber)[0]
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )
        if self.theory["GC_use_cold_matter_tracer"]:
            self.nonlinear_neutrino_approx.wrap_linear_neutrino_approx(
                self.theory["Pk_delta"],
                self.nonlinear_dic["Bar_boost"],
                self.nonlinear_dic["NL_boost"],
            )
            pterm = sqrt(
                self.nonlinear_neutrino_approx(
                    redshift,
                    wavenumber,
                )
                * pterm
            )

        pval = bterm * pterm
        return pval

    def Pgi_phot_emu_tatt(self, redshift, wavenumber):
        r"""Pgi Phot emu TATT

        Computes the photometric galaxy-intrinsic power spectrum assuming a
        linear bias and the tidal alignment tidal torquing intrinsic alignment
        model. Uses the EuclidEmu2 or the BACCO emulator for the nonlinear
        boost to the matter power spectrum, with baryon effects (if selected)
        added as a boost.

        .. math::
            P_{\rm G I}^{\rm photo}(z, k) = b_g^{\rm photo}(z) \
            \left [ C_{1}P_{\rm \delta\delta}(z, k) B_{\rm NL}(z, k) \
            S_{\rm bar}(z, k)+C_{1\delta}D(z)^{4}[\rm A_{0|0E}(k)+\rm \
            C_{0|0E}(k)]+C_{2}D(z)^{4}[\rm A_{0|E2}(k)+B_{0|E2}(k)] \right ]

        Note: either redshift or wavenumber must be a float (ex. simultaneously
        setting both of them to numpy.ndarray makes the code crash)

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of photometric galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        c1, c1d, c2 = self.misc.normalize_tatt_parameters(redshift)
        growth = self.theory["D_z_k_func"](redshift, wavenumber)

        pterm = (
            self.theory["Pk_delta"].P(redshift, wavenumber)
            * self.nonlinear_dic["NL_boost"](redshift, wavenumber)[0]
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )

        bterm = self.theory["b1_inter"](redshift)
        if self.theory["GC_use_cold_matter_tracer"]:
            # we could rescale bterm like for the bNL case
            # but the tidal field terms do not correspond to Pcb in this case
            raise ValueError(
                "The neutrino galaxy bias implementation does"
                + "not exist for the TATT initial alignment model"
            )

        pval = bterm * (
            c1 * pterm
            + c1d
            * (growth**4)
            * (self.theory["a00e"](wavenumber) + self.theory["c00e"](wavenumber))
            + c2
            * (growth**4)
            * (self.theory["a0e2"](wavenumber) + self.theory["b0e2"](wavenumber))
        )

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
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of spectroscopic galaxy-intrinsic power spectrum
            at a given redshift and wavenumber
        """
        pval = (
            self.misc.fia(redshift)
            * self.misc.istf_spectro_galbias(redshift)
            * self.theory["Pk_delta"].P(redshift, wavenumber)
        )
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
            Wavenumber(s) at which to evaluate the power spectrum

        Returns
        -------
        Photometric galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """
        bterm = self.theory["b1_inter"](redshift)
        if self.theory["GC_use_cold_matter_tracer"]:
            pterm = sqrt(
                self.theory["Pk_cb"].P(redshift, wavenumber)
                * self.theory["Pk_delta"].P(redshift, wavenumber)
            )
        else:
            pterm = self.theory["Pk_delta"].P(redshift, wavenumber)
        return bterm * pterm

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
            Wavenumber(s) at which to evaluate the power spectrum

        Returns
        -------
        Halo model phot galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """
        bterm = self.theory["b1_inter"](redshift)
        pterm = (
            self.theory["Pk_halomodel_recipe"].P(redshift, wavenumber)
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )

        if self.theory["GC_use_cold_matter_tracer"]:
            self.nonlinear_neutrino_approx.wrap_linear_neutrino_approx(
                self.theory["Pk_halomodel_recipe"],
                self.nonlinear_dic["Bar_boost"],
            )

            pterm = sqrt(self.nonlinear_neutrino_approx(redshift, wavenumber) * pterm)

        pval = bterm * pterm
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
            Wavenumber(s) at which to evaluate the power spectrum

        Returns
        -------
        Emulator phot galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """
        bterm = self.theory["b1_inter"](redshift)
        pterm = (
            self.theory["Pk_delta"].P(redshift, wavenumber)
            * self.nonlinear_dic["NL_boost"](redshift, wavenumber)[0]
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )

        if self.theory["GC_use_cold_matter_tracer"]:
            self.nonlinear_neutrino_approx.wrap_linear_neutrino_approx(
                self.theory["Pk_delta"],
                self.nonlinear_dic["Bar_boost"],
                self.nonlinear_dic["NL_boost"],
            )
            pterm = sqrt(self.nonlinear_neutrino_approx(redshift, wavenumber) * pterm)

        pval = bterm * pterm
        return pval

    def _add_NLbias_contributions_gL(
        self,
        Pmmnl,
        redshift,
        wavenumber,
        Pcbnl=None,
    ):
        r"""Add NL bias contributions

        Gets the non-linear bias contributions, combines it with
        the corresponding bias parameters and adds it to any non-linear
        matter power spectrum it is given to return the galaxy-matter
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

        b1 = self.theory["b1_inter"](redshift)
        if self.theory["GC_use_cold_matter_tracer"]:
            warn(
                "The neutrino implementation only rescales the b1 bias", RuntimeWarning
            )
            b1 *= sqrt(Pcbnl / Pmmnl)

        b2 = self.theory["b2_inter"](redshift)
        bG2 = self.theory["bG2_inter"](redshift)
        bG3 = self.theory["bG3_inter"](redshift)

        Pb1b2 = self.nonlinear_dic["Pb1b2_kz"](redshift, wavenumber, grid=False)
        Pb1bG2 = self.nonlinear_dic["Pb1bG2_kz"](redshift, wavenumber, grid=False)
        PZ1bG3 = self.nonlinear_dic["PZ1bG3_kz"](redshift, wavenumber, grid=False)
        PZ1bG2 = self.nonlinear_dic["PZ1bG2_kz"](redshift, wavenumber, grid=False)

        return (
            b1 * Pmmnl
            + 0.5 * b2 * Pb1b2
            + 0.5 * bG2 * (Pb1bG2 + PZ1bG2)
            + 0.5 * bG3 * PZ1bG3
        )

    def Pgdelta_phot_halo_NLbias(self, redshift, wavenumber):
        r"""Pgdelta Phot Halo NLbias

        Computes the galaxy-matter power spectrum for the photometric probe
        with non-linear bias contributions computed with Fast-PT from the
        linear power spectrum. Uses halo model based codes for the matter
        power spectrum. This include terms in b1 (linear bias),
        b2 (quadratic bias), bG2 (quadratic non-local bias)
        and bG3 (cubic non-local bias).

        In case of massive neutrinos to handle the scale dependance of the bias
        we rescale the linear b1 bias as described in [1807.04672]
        and [2405.06047].

        .. math::
            P_{\rm gm}^{\rm photo}(z, k) =\
            &[b_{1,{\rm g}}^{\rm photo}(z)]
            P_{\rm \delta\delta}^{\rm NL}(z, k) +\\
            &[b_{2,{\rm g}}^{\rm photo}(z)] P_{b_1b_2}(z, k) +\\
            &[b_{{\mathcal{G}_2},{\rm g}}^{\rm photo}(z)]
            P_{b_1b_{\mathcal{G}_2}}(z, k) +\\
            &[b_{\Gamma_3,{\rm g}}^{\rm photo}(z)] P_{b_1b_{\Gamma_3}}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """

        Pmm = (
            self.theory["Pk_halomodel_recipe"].P(redshift, wavenumber)
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )

        if self.theory["GC_use_cold_matter_tracer"]:
            self.nonlinear_neutrino_approx.wrap_linear_neutrino_approx(
                self.theory["Pk_halomodel_recipe"],
                self.nonlinear_dic["Bar_boost"],
            )
            Pcb = self.nonlinear_neutrino_approx(redshift, wavenumber)

            pval = self._add_NLbias_contributions_gL(
                Pmm,
                redshift,
                wavenumber,
                Pcbnl=Pcb,
            )
        else:
            pval = self._add_NLbias_contributions_gL(
                Pmm,
                redshift,
                wavenumber,
            )

        return pval

    def Pgdelta_phot_emu_NLbias(self, redshift, wavenumber):
        r"""Pgdelta Phot Emu NLbias

        Computes the galaxy-matter power spectrum for the photometric probe
        with non-linear bias contributions computed with Fast-PT from the
        linear power spectrum. Uses the EuclidEmu2 or the BACCO
        emulator for the nonlinear boost to the matter power spectrum.
        This include terms in b1 (linear bias),
        b2 (quadratic bias), bG2 (quadratic non-local bias)
        and bG3 (cubic non-local bias).

        In case of massive neutrinos to handle the scale dependance of the bias
        we rescale the linear b1 bias as described in [1807.04672]
        and [2405.06047].

        .. math::
            P_{\rm gm}^{\rm photo}(z, k) =\
            &[b_{1,{\rm g}}^{\rm photo}(z)]
            P_{\rm \delta\delta}^{\rm NL}(z, k) +\\
            &[b_{2,{\rm g}}^{\rm photo}(z)] P_{b_1b_2}(z, k) +\\
            &[b_{{\mathcal{G}_2},{\rm g}}^{\rm photo}(z)]
            P_{b_1b_{\mathcal{G}_2}}(z, k) +\\
            &[b_{\Gamma_3,{\rm g}}^{\rm photo}(z)] P_{b_1b_{\Gamma_3}}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        wavenumber: float or list or numpy.ndarray
            wavenumber(s) at which to evaluate the power spectrum.

        Returns
        -------
        pval: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift and wavenumber for galaxy clustering
            photometric
        """

        Pmm = (
            self.theory["Pk_delta"].P(redshift, wavenumber)
            * self.nonlinear_dic["NL_boost"](redshift, wavenumber)[0]
            * self.nonlinear_dic["Bar_boost"](redshift, wavenumber)[0]
        )

        if self.theory["GC_use_cold_matter_tracer"]:
            self.nonlinear_neutrino_approx.wrap_linear_neutrino_approx(
                self.theory["Pk_delta"],
                self.nonlinear_dic["Bar_boost"],
                self.nonlinear_dic["NL_boost"],
            )
            Pcb = self.nonlinear_neutrino_approx(redshift, wavenumber)

            pval = self._add_NLbias_contributions_gL(
                Pmm,
                redshift,
                wavenumber,
                Pcbnl=Pcb,
            )
        else:
            pval = self._add_NLbias_contributions_gL(
                Pmm,
                redshift,
                wavenumber,
            )

        return pval
