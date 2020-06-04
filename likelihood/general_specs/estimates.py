# -*- coding: utf-8 -*-
"""Estimates

Module for theory estimates of observables.
"""

import numpy as np
from scipy import integrate
from scipy import interpolate


class Galdist:
    """
    Class for cosmic shear observables.
    """

    def __init__(self, bin_i, n_type='istf', n_file=None,
                 survey_lims=[0.001, 4.0], **kwargs):
        """
        Parameters
        ----------
        bin_i: int
           index of desired tomographic bin. Tomographic bin
           indices should start from 1.
        n_type: str
           Parameter to specify which n(z) to use:
           'istf' - Use Euclid IST:Forecasting n(z).
           'custom' - Input custom n(z). In this case, n_file must also
           be specified.
        n_file: str
           Location of custom n(z) file.
        survey_lims: list, float
           Redshift range of entire survey (lower, higher)
           Euclid default (0.001, 4.0).
        """
        self.survey_min, self.survey_max = survey_lims
        if bin_i is None:
            raise Exception('You must specify the index for the desired '
                            'tomographic bin.')
        elif bin_i == 0:
            raise ValueError('Tomographic bin indices start from 1.')
        if n_type == 'istf':
            istf_bin_list = [0.001, 0.418, 0.560, 0.678, 0.789, 0.900,
                             1.019, 1.155, 1.324, 1.576, 2.50]
            self.n_i = self.p_up(istf_bin_list[bin_i - 1],
                                 istf_bin_list[bin_i])
        elif n_type == 'custom':
            if n_file is None:
                raise Exception('When choosing a custom n(z), you must'
                                'specify the file location using n_file.')
            else:
                ntab = np.loadtxt(n_file)
                self.n_i = interpolate.InterpolatedUnivariateSpline(
                    ntab[:, 0], ntab[:, bin_i], ext=2)
        else:
            raise Exception('n_type must be istf or custom.')

    def n_istf(self, z, n_gal=30.0):
        """
        Implements true, normalised galaxy source distribution as defined by
        Euclid IST: Forecasting.

        See Eq. 113 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.
        n_gal: float
           Galaxy surface density of survey.
           Euclid default - 30 arcmin^{-2}.

        Returns
        -------
        float
           n(z) for ISTF source distribution.
        """
        n_dist_int = integrate.quad(self.n_istf_int, a=self.survey_min,
                                    b=self.survey_max)
        prop_con = n_gal / n_dist_int[0]
        fin_n = prop_con * self.n_istf_int(z=z)
        return fin_n

    def n_istf_int(self, z, zm=0.9):
        """
        Integrand for true galaxy source distribution as defined by Euclid
        IST: Forecasting.

        See Eq. 113 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z: float
           Redshift at which to evaluate distribution.
        zm: float
           Median redshift of survey. Euclid default = 0.9.

        Returns
        -------
        float
           unnormalised n(z) for ISTF source distribution.
        """
        z0 = zm / np.sqrt(2.0)
        n_val = ((z / z0)**2.0) * np.exp((-(z / z0)**1.5))
        return n_val

    def p_phot(self, z_photo, z, c_b=1.0, z_b=0.0, sigma_b=0.05,
               c_outlier=1.0, z_outlier=0.1, sigma_outlier=0.05,
               outlier_frac=0.1):
        """
        Probability distribution function, describing the probability that
        a galaxy with redshift z has a measured redshift z_photo.

        Eq. 115 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z_photo: float
            Measured photometric redshift
        z:  float
            True redshift
        c_b: float
            Multiplicative bias on sample with well-measured redshift.
            Euclid IST: Forecasting default = 1.0.
        z_b: float
            Additive bias on sample with well-measured redshift.
            Euclid IST: Forecasting default = 0.0.
        sigma_b: float
            Sigma for sample with well-measured redshift.
            Euclid IST: Forecasting default = 0.05.
        c_outlier: float
            Multiplicative bias on sample of catastrophic outliers.
            Euclid IST: Forecasting default = 1.0.
        z_outlier: float
            Additive bias on sample of catastrophic outliers.
            Euclid IST: Forecasting default = 0.1.
        sigma_outlier: float
            Sigma for sample of catastrophic outliers.
            Euclid IST: Forecasting default = 0.05.
        outlier_frac: float
            Fraction of catastrophic outliers.
            Euclid IST: Forecasting default = 0.1.

        Returns
        -------
        float
            Probability.
        """

        fac1 = ((1.0 - outlier_frac) / (np.sqrt(2.0 * np.pi) *
                                        sigma_b * (1.0 + z)))
        fac2 = (outlier_frac / (np.sqrt(2.0 * np.pi) *
                                sigma_outlier * (1.0 + z)))

        p_val = ((fac1 * np.exp((-0.5) * ((z - (c_b * z_photo) - z_b) /
                                          (sigma_b * (1.0 + z)))**2.0)) +
                 (fac2 * np.exp((-0.5) * ((z - (c_outlier * z_photo) -
                                           z_outlier) /
                                          (sigma_outlier * (1.0 + z)))**2.0)))

        return p_val

    def _n_phot_int(self, z, bin_z_max, bin_z_min):
        """
        Defines integrand to calculate denominator of Eq. 112 in
        https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        z: float
           Redshift
        bin_z_max: float
                   Upper limit of bin
        bin_z_min: float
                   Lower limit of bin

        Returns
        -------
        float
            Denominator integrand value.
        """
        ret_val = self.n_istf(z) * integrate.quad(self.p_phot, a=bin_z_min,
                                                  b=bin_z_max, args=(z))[0]
        return ret_val

    def p_up(self, bin_z_min, bin_z_max, int_step=0.1):
        """
        Computes the observed galaxy distribution, by carrying out convolution
        of true n(z) with photometric uncertainty PDF.

        Output is the normalized n(z) within the redshift bin.

        Eq. 112 of https://arxiv.org/abs/1910.09273

        Parameters
        ----------
        bin_z_max: float
                   High z end of bin
        bin_z_min: float
                   Low z end of bin
        int_step: float
                  Redshift step size for interpolating final n(z).
                  Default = 0.1.

        Returns
        -------
        float
            Observed n(z) with photometric redshift uncertainties accounted.
        """
        self.int_step = int_step
        z_list = np.arange(self.survey_min, self.survey_max, int_step)
        n_nums_list = []
        for zind in range(len(z_list)):
            finprod = (self.n_istf(z_list[zind]) *
                       integrate.quad(self.p_phot, a=bin_z_min,
                                      b=bin_z_max, args=(z_list[zind]))[0])
            n_nums_list.append(finprod)

        n_denom = integrate.quad(self._n_phot_int, a=z_list[0],
                                 b=z_list[-1], args=(bin_z_max, bin_z_min))[0]

        res = np.array(n_nums_list) / n_denom

        obs_bin = interpolate.InterpolatedUnivariateSpline(z_list, res, ext=2)

        return obs_bin
