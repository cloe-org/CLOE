# -*- coding: utf-8 -*-
"""Cosmology

Class to store cosmological parameters and functions.
"""

import numpy as np
from scipy import interpolate
from astropy import constants as const


class CosmologyError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Cosmology:
    """
    Class for cosmological observables
    """

    def __init__(self):
        """
        Cosmology dictionary parameters
        -------------------------------
        H0: float
            Present-day Hubble constant (km s^{-1} Mpc^{-1})
        omch2: float
            Present-day Omega_CDM * (H0/100)**2
        ombh2: float
            Present-day Omega_baryon * (H0/100)**2
        omkh2: float
            Present-day curvature energy density
            Omega_k * (H0/100)**2
        As: float
            amplitude of the primordial power spectrum
        ns: float
            spectral tilt of the primordial
            power spectrum
        sigma_8_0: double
            sigma8 evaluated at z = 0
        w: float
           Dark energy equation of state
        omnuh2: float
            Present-day Omega_neutrinos * (H0/100)**2
        mnu: float
            Sum of massive neutrino species masses (eV)
        comov_dist: array-like
            Value of comoving distances at redshifts z_win
        angular_dist: array-like
            Value of angular diameter distances at redshifts z_win
        H: array-like
            Hubble function evaluated at redshifts z_win
        Pk_interpolator: function
            Interpolator function for power spectrum from Boltzmann code
        Pk_delta: function
            Interpolator function for delta from Boltzmann code
        fsigma8: array
            fsigma8 function evaluated at z
        sigma_8: array
            sigma8 functione valuated at z
        c: float
            Speed-of-light in units of km s^{-1}
        r_z_func: function
            Interpolated function for comoving distance
        d_z_func: function
            Interpolated function for angular diameter distance
        sigma8_z_func: function
            Interpolated function for angular sigma8
        fsigma8_z_func: function
            Interpolated function for fsigma8
        H_z_func: function
            Interpolated function for Hubble parameter
        z_win: array-like
            Array of redshifts ar which H and comov_dist are evaluated at
        k_win: array-like
            Array of k values which will be used to evaluate galaxy power
            spectra
        Pgg_phot: function
            Galaxy-galaxy power spectrum for GC-phot
        Pgdelta_phot: function
            Galaxy-matter power spectrum for GC-phot
        Pgg_spec: function
            Galaxy-galaxy power spectrum for GC-spec
        Pgdelta_spec: function
            Galaxy-matter power spectrum for GC-spec
        Pii: function
            Intrinsic alignment (intrinsic-intrinsic) power spectrum
        Pdeltai: function
            Density-intrinsic cross-spectrum
        Pgi_phot: function
            Photometric galaxy-intrinsic cross-spectrum
        Pgi_spec: function
            Spectroscopic galaxy-intrinsic cross-spectrum
        MG_mu: function
            mu function from Modified Gravity parametrization
        MG_msigma: function
            sigma function from Modified Gravity parametrization
        nuisance_parameters: dictionary
            Contains all nuisance bias parameters
            and IA parameters which are sampled over.
            At the moment, we have implemented
            10 constant bias for photo-z
            recipe and 4 for spec recipe,
            and 3 IA parameters. The
            initialized values of the fiducial
            cosmology dictionary corresponds to
            (1) Photo-z: values corrsponding to

            .. math::
                b_{x,i} = \sqrt{1+\bar{z}_{x,i}}\\

            (2) Spec: bias values
            of arXiv:1910.0923
            (3) IA values in arXiv:1910.0923
        """
        # (GCH): initialize cosmo dictionary
        # (ACD): Added speed of light to dictionary.!!!Important:it's in units
        # of km/s to be dimensionally consistent with H0.!!!!
        self.cosmo_dic = {'H0': 67.5,
                          'omch2': 0.122,
                          'ombh2': 0.022,
                          'omnuh2': 0.00028,
                          'omkh2': 0.0,
                          'w': -1.0,
                          'mnu': 0.06,
                          'tau': 0.07,
                          'nnu': 3.046,
                          'ns': 0.9674,
                          'As': 2.1e-9,
                          'sigma_8_0': 0.816,
                          'comov_dist': None,
                          'angular_dist': None,
                          'H': None,
                          'Pk_interpolator': None,
                          'Pk_delta': None,
                          'Pgg_phot': None,
                          'Pgdelta_phot': None,
                          'Pgg_spec': None,
                          'Pgdelta_spec': None,
                          'Pii': None,
                          'Pdeltai': None,
                          'Pgi_phot': None,
                          'Pgi_spec': None,
                          'fsigma8': None,
                          'sigma_8': None,
                          'c': const.c.to('km/s').value,
                          'z_win': None,
                          'k_win': None,
                          'r_z_func': None,
                          'd_z_func': None,
                          'H_z_func': None,
                          'sigma8_z_func': None,
                          'fsigma8_z_func': None,
                          'MG_mu': None,
                          'MG_sigma': None,
                          'nuisance_parameters': {
                             'like_selection': 12,
                             'b1_photo': 1.0997727037892875,
                             'b2_photo': 1.220245876862528,
                             'b3_photo': 1.2723993083933989,
                             'b4_photo': 1.316624471897739,
                             'b5_photo': 1.35812370570578,
                             'b6_photo': 1.3998214171814918,
                             'b7_photo': 1.4446452851824907,
                             'b8_photo': 1.4964959071110084,
                             'b9_photo': 1.5652475842498528,
                             'b10_photo': 1.7429859437184225,
                             'b1_spec': 1.46,
                             'b2_spec': 1.61,
                             'b3_spec': 1.75,
                             'b4_spec': 1.90,
                             'aia': 1.72,
                             'nia': -0.41,
                             'bia': 2.17}}

    def growth_factor(self, zs, ks):
        r"""
        Calculates growth factor according to

        .. math::
                   D(z, k) &=\sqrt{P(z, k)/P(z=0, k)}\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        growth factor

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        try:
            D_z_k = self.cosmo_dic['Pk_delta'].P(zs, ks)
            D_z_k = np.sqrt(D_z_k / self.cosmo_dic['Pk_delta'].P(0.0, ks))
            return D_z_k
        except CosmologyError:
            print('Computation error in D(z, k)')

    def growth_rate(self, zs, ks):
        r"""
        Calculates growth rate according to

        .. math::
                   f(z, k) &=-(1+z)/D dD(z, k)/dz\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        interpolator growth rate

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        # I assume 1+z=1/a where a: scale factor
        D_z_k = self.growth_factor(zs, ks)
        # This will work when k is fixed, not an array
        try:
            f_z_k = -(1 + zs) * np.gradient(D_z_k, zs[1] - zs[0]) / D_z_k
            return interpolate.InterpolatedUnivariateSpline(
                x=zs, y=f_z_k, ext=2)
        except CosmologyError:
            print('Computation error in f(z, k)')
            print('ATTENTION: Check k is a value, not a list')

    def interp_comoving_dist(self):
        """
        Adds an interpolator for comoving distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator comoving distance as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['r_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['comov_dist'], ext=2)

    def interp_angular_dist(self):
        """
        Adds an interpolator for angular distance to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator angular diameter distance  as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['d_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['angular_dist'], ext=2)

    def interp_H(self):
        """
        Adds an interpolator for the Hubble parameter to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator H as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['H_z_func'] = interpolate.InterpolatedUnivariateSpline(
            x=self.cosmo_dic['z_win'], y=self.cosmo_dic['H'], ext=2)

    def interp_sigma8(self):
        """
        Adds an interpolator for sigma8 to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator sigma8 as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['sigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['sigma_8'], ext=2)

    def interp_fsigma8(self):
        """
        Adds an interpolator for fsigma8 to the dictionary so that
        it can be evaluated at redshifts not explictly supplied to cobaya.

        Returns
        -------
        Interpolator fsigma8 as a function of redshift

        """
        if self.cosmo_dic['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')
        self.cosmo_dic['fsigma8_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                x=self.cosmo_dic['z_win'], y=self.cosmo_dic['fsigma8'], ext=2)

    def istf_phot_galbias(self, redshift, bin_edge_list=[0.001, 0.418, 0.560,
                                                         0.678, 0.789, 0.900,
                                                         1.019, 1.155, 1.324,
                                                         1.576, 2.50]):
        r"""
        Updates galaxy bias for the photometric GC probes at a given
        redshift z

        Note: for redshifts above the final bin (z > 2.5), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.001), we use the bias of the first bin.

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of tomographic redshift bin edges for photometic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        Value of galaxy bias at input redshift.
        """
        istf_bias_list = [self.cosmo_dic['nuisance_parameters']['b1_photo'],
                          self.cosmo_dic['nuisance_parameters']['b2_photo'],
                          self.cosmo_dic['nuisance_parameters']['b3_photo'],
                          self.cosmo_dic['nuisance_parameters']['b4_photo'],
                          self.cosmo_dic['nuisance_parameters']['b5_photo'],
                          self.cosmo_dic['nuisance_parameters']['b6_photo'],
                          self.cosmo_dic['nuisance_parameters']['b7_photo'],
                          self.cosmo_dic['nuisance_parameters']['b8_photo'],
                          self.cosmo_dic['nuisance_parameters']['b9_photo'],
                          self.cosmo_dic['nuisance_parameters']['b10_photo']]

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
        elif redshift >= bin_edge_list[-1]:
            bi_val = istf_bias_list[-1]
        elif redshift < bin_edge_list[0]:
            bi_val = istf_bias_list[0]
        return bi_val

    def istf_spec_galbias(self, redshift, bin_edge_list=[0.90, 1.10, 1.30,
                                                         1.50, 1.80]):
        """
        Updates galaxy bias for the spectroscopic GC probe, at given
        redshift, according to default recipe.

        Note: for redshifts above the final bin (z > 1.80), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.90), we use the bias of the first bin.

        Attention: this will change in the future

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of redshift bin edges for spectroscopic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        Value of galaxy bias at input redshift.
        """

        istf_bias_list = [self.cosmo_dic['nuisance_parameters']['b1_spec'],
                          self.cosmo_dic['nuisance_parameters']['b2_spec'],
                          self.cosmo_dic['nuisance_parameters']['b3_spec'],
                          self.cosmo_dic['nuisance_parameters']['b4_spec']]

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
        elif redshift >= bin_edge_list[-1]:
            bi_val = istf_bias_list[-1]
        elif redshift < bin_edge_list[0]:
            bi_val = istf_bias_list[0]
        return bi_val

    def Pgg_phot_def(self, redshift, k_scale):
        r"""
        Calculates the galaxy-galaxy power spectrum for the photometric probe.

        .. math::
            P_{\rm gg}^{\rm photo}(z, k) &=\
            [b_g^{\rm photo}(z)]^2 P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the  power spectrum.

        Returns
        -------
        Value of G-G power spectrum at given k and redshift.
        """
        pval = ((self.istf_phot_galbias(redshift) ** 2.0) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgg_spec_def(self, redshift, k_scale, mu_rsd):
        """
        Calculates the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float
            cosinus of the angle between the pair separation and the l.o.s.

        Returns
        -------
        Value of redshift-space G-G power spectrum at given k, redshift
        and mu
        """
        bias = self.istf_spec_galbias(redshift)
        fs8 = self.cosmo_dic['fsigma8_z_func'](redshift)
        s8 = self.cosmo_dic['sigma8_z_func'](redshift)
        growth = fs8 / s8
        power = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def Pgd_phot_def(self, redshift, k_scale):
        r"""
        Calculates the galaxy-matter power spectrum for the photometric probe.

        .. math::
            P_{\rm g\delta}^{\rm photo}(z, k) &=\
            [b_g^{\rm photo}(z)] P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        Value of G-delta power spectrum at given k and redshift.
        """
        pval = (self.istf_phot_galbias(redshift) *
                self.cosmo_dic['Pk_delta'].P(redshift, k_scale))
        return pval

    def Pgd_spec_def(self, redshift, k_scale, mu_rsd):
        r"""
        Calculates the redshift-space galaxy-matter power spectrum for the
        spectroscopic probe.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float
            cosinus of the angle between the pair separation and the l.o.s.

        Returns
        -------
        Value of G-delta power spectrum at given k and redshift.
        """
        bias = self.istf_spec_galbias(redshift)
        fs8 = self.cosmo_dic['fsigma8_z_func'](redshift)
        s8 = self.cosmo_dic['sigma8_z_func'](redshift)
        growth = fs8 / s8
        power = self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        pval = ((bias + growth * mu_rsd ** 2.0) *
                (1.0 + growth * mu_rsd ** 2.0)) * power
        return pval

    def fia(self, redshift, k_scale=0.001):
        r"""
        Computes the intrinsic alignment function.

        .. math::
            f_{\rm IA}(z) &= -\mathcal{A_{\rm IA}}\mathcal{C_{\rm IA}}\
            \frac{\Omega_{m,0}}{D(z)}(1 + z)^{\eta_{\rm IA}}\
            [\langle L \rangle(z) /L_{\star}(z)]^{\beta_{\rm IA}}\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.

        Returns
        -------
        Value of intrinsic alignment function at given redshift.
        """
        c1 = 0.0134
        aia = self.cosmo_dic['nuisance_parameters']['aia']
        nia = self.cosmo_dic['nuisance_parameters']['nia']
        bia = self.cosmo_dic['nuisance_parameters']['bia']
        omegam = (self.cosmo_dic['ombh2'] + self.cosmo_dic['omch2'] +
                  self.cosmo_dic['omnuh2']) / (self.cosmo_dic['H0'] / 100)**2
        # SJ: temporary lum for now, to be read in from IST:forecast file
        lum = 1.0
        fia = (-aia * c1 * omegam / self.growth_factor(redshift, k_scale) *
               (1 + redshift)**nia * lum**bia)
        return fia

    def Pii_def(self, redshift, k_scale):
        r"""
        Computes the intrinsic alignment (intrinsic-intrinsic) power spectrum.

        .. math::
            P_{\rm II}(z, k) = [f_{\rm IA}(z)]^2P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        Value of intrinsic alignment power spectrum at given k and redshift.
        """
        pval = self.fia(redshift)**2.0 * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pdeltai_def(self, redshift, k_scale):
        r"""
        Computes the density-intrinsic power spectrum.

        .. math::
            P_{\rm \delta I}(z, k) = [f_{\rm IA}(z)]P_{\rm \delta\delta}(z, k)

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        Value of density-intrinsic power spectrum at given k and redshift.
        """
        pval = self.fia(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_phot_def(self, redshift, k_scale):
        r"""
        Computes the photometric galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm photo}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm photo}(z)P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        Value of photometric galaxy-intrinsic power spectrum at given
        k and redshift.
        """
        pval = self.fia(redshift) * self.istf_phot_galbias(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def Pgi_spec_def(self, redshift, k_scale):
        r"""
        Computes the spectroscopic galaxy-intrinsic power spectrum.

        .. math::
            P_{\rm gI}^{\rm spec}(z, k) &=\
            [f_{\rm IA}(z)]b_g^{\rm spec}(z)P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.

        Returns
        -------
        Value of spectroscopic galaxy-intrinsic power spectrum at given
        k and redshift.
        """
        pval = self.fia(redshift) * self.istf_spec_galbias(redshift) * \
            self.cosmo_dic['Pk_delta'].P(redshift, k_scale)
        return pval

    def interp_phot_galaxy_spectra(self):
        """
        Creates interpolators for the photometric galaxy
        clustering and galaxy-matter power spectra, and adds them to cosmo_dic.
        """
        # AP: Removed the interpolation of the spectroscopic galaxy power
        # spectra and renamed this method to reflect that the interpolation
        # is carrioed out only on the photometric spectra. This is because,
        # in order to interface the spec module with Pgg_spec we need it
        # in redshift-space , i.e. Pgg_spec(z, k, rsd_mu), and a 3-d
        # interpolation drastically increases the evaluation time.
        # Pgg_spec and Pgd_spec are added to the cosmo dic in the method
        # update_cosmo_dic
        ks_base = self.cosmo_dic['k_win']
        zs_base = self.cosmo_dic['z_win']

        zs_interp = np.tile(zs_base, len(ks_base))
        ks_interp = np.repeat(ks_base, len(zs_base))

        z_reshape = np.array(zs_interp).reshape((len(zs_interp), 1))
        k_reshape = np.array(ks_interp).reshape((len(ks_interp), 1))

        zk_arr = np.concatenate((z_reshape, k_reshape), axis=1)

        pgg_phot = []
        pgdelta_phot = []
        pii = []
        pdeltai = []
        pgi_phot = []
        pgi_spec = []
        for index in range(len(ks_interp)):
            pgg_phot.append(self.Pgg_phot_def(zs_interp[index],
                                              ks_interp[index]))
            pgdelta_phot.append(self.Pgd_phot_def(zs_interp[index],
                                                  ks_interp[index]))
            pii.append(self.Pii_def(zs_interp[index], ks_interp[index]))
            pdeltai.append(self.Pdeltai_def(zs_interp[index],
                                            ks_interp[index]))
            pgi_phot.append(self.Pgi_phot_def(zs_interp[index],
                                              ks_interp[index]))
            pgi_spec.append(self.Pgi_spec_def(zs_interp[index],
                                              ks_interp[index]))
        self.cosmo_dic['Pgg_phot'] = interpolate.LinearNDInterpolator(zk_arr,
                                                                      pgg_phot)
        self.cosmo_dic['Pgdelta_phot'] = interpolate.LinearNDInterpolator(
            zk_arr, pgdelta_phot)
        self.cosmo_dic['Pii'] = interpolate.LinearNDInterpolator(zk_arr, pii)
        self.cosmo_dic['Pdeltai'] = interpolate.LinearNDInterpolator(zk_arr,
                                                                     pdeltai)
        self.cosmo_dic['Pgi_phot'] = interpolate.LinearNDInterpolator(zk_arr,
                                                                      pgi_phot)
        self.cosmo_dic['Pgi_spec'] = interpolate.LinearNDInterpolator(zk_arr,
                                                                      pgi_spec)
        return

    def MG_mu_def(self, redshift, k_scale, MG_mu):
        """
        Returns the mu function according to mu-sigma Modified Gravity
        parametrization.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate mu.
        k_scale: float
            k-mode at which to evaluate mu.
        MG_mu: float
            Value of constant (for v1.0) MG mu function.

        Returns
        -------
        Value of MG mu function at given k and redshift.
        """

        return MG_mu

    def MG_sigma_def(self, redshift, k_scale, MG_sigma):
        """
        Returns the sigma function according to mu-sigma Modified Gravity
        parametrization.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate mu.
        k_scale: float
            k-mode at which to evaluate mu.
        MG_sigma: float
            Value of constant (for v1.0) MG sigma function

        Returns
        -------
        Value of MG sigma function at given k and redshift.
        """

        return MG_sigma

    def update_cosmo_dic(self, zs, ks, MG_mu=1.0, MG_sigma=1.0):
        """
        Update the dictionary with other cosmological quantities

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum
        """
        # (GCH): this function is superfluous
        # just in case we want to have always
        # an updated dictionary with D_z, f, H(z), r(z)
        self.interp_H()
        self.interp_comoving_dist()
        self.interp_fsigma8()
        self.interp_sigma8()
        self.interp_angular_dist()
        self.interp_phot_galaxy_spectra()
        self.cosmo_dic['Pgg_spec'] = self.Pgg_spec_def
        self.cosmo_dic['Pgdelta_spec'] = self.Pgd_spec_def
        self.cosmo_dic['Pii'] = self.Pii_def
        self.cosmo_dic['Pdeltai'] = self.Pdeltai_def
        self.cosmo_dic['Pgi_phot'] = self.Pgi_phot_def
        self.cosmo_dic['Pgi_spec'] = self.Pgi_spec_def
        self.cosmo_dic['D_z_k'] = self.growth_factor(zs, ks)
        self.cosmo_dic['f_z_k'] = self.growth_rate(zs, ks)
        self.cosmo_dic['sigma_8_0'] = \
            self.cosmo_dic['sigma8_z_func'](0)
        self.cosmo_dic['MG_mu'] = lambda x, y: self.MG_mu_def(x, y, MG_mu)
        self.cosmo_dic['MG_sigma'] = lambda x, y: self.MG_sigma_def(x, y,
                                                                    MG_sigma)
