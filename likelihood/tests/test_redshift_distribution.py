"""Unit Tests for Redshift Distribution

This module contains unit tests for the RedshiftDistribution
sub-module of the photometric survey module.
=======

"""

from unittest import TestCase
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline
from likelihood.photometric_survey.redshift_distribution \
    import RedshiftDistribution
import numpy as np
import numpy.testing as npt


class RedshiftDistributionTestCase(TestCase):

    def setUp(self) -> None:
        # For testing purposes, three tomographic bins are sufficient
        self.bins = [1, 2, 3]
        # Cases of interest to build the test nuisance dictionaries
        good_dz_dic = {'dz_1_GC': 0.1, 'dz_2_GC': 0.1, 'dz_3_GC': 0.1}
        zero_dz_dic = {'dz_1_GC': 0.0, 'dz_2_GC': 0.0, 'dz_3_GC': 0.0}
        bad_dz_dic = {'dz_0_GC': 0.0, 'dz_2_GC': 0.0, 'dz_3_GC': 0.0}
        # nuisance dictionary to test initialization
        self.good_nuisance_dic = {}
        self.good_nuisance_dic.update(good_dz_dic)
        # nuisance dictionary to test the nuisance=0 scenario
        self.zero_nuisance_dic = {}
        self.zero_nuisance_dic.update(zero_dz_dic)
        # nuisance dictionary to test shift only
        self.shift_only_nuisance_dic = {}
        self.shift_only_nuisance_dic.update(good_dz_dic)
        # nuisance dictionary to test bad formatting in initialization
        self.bad_nuisance_dic = {}
        self.bad_nuisance_dic.update(bad_dz_dic)
        # Three Gaussian distributions for test purposes
        # The three means are in [0.1, 0.2, 0.3], and sigma=1 (for all)
        gauss_list = [norm(loc=0.1 * i) for i in self.bins]
        self.zs = np.linspace(0.0, 1.5, num=1000)
        gauss_interp = \
            [InterpolatedUnivariateSpline(self.zs, gauss.pdf(self.zs))
             for gauss in gauss_list]
        # Dictionary to test the class with the Gaussian distributions
        self.gauss_nz_dic = {'n1': gauss_interp[0], 'n2': gauss_interp[1],
                             'n3': gauss_interp[2]}
        # The peak values should equal 1/sqrt(2pi)
        self.peak_values = [1.0 / np.sqrt(2.0 * np.pi)] * len(self.bins)

    def tearDown(self) -> None:
        pass

    def test_init_success(self):
        nz = RedshiftDistribution('GC', self.gauss_nz_dic,
                                  self.good_nuisance_dic)
        npt.assert_array_equal(nz.get_tomographic_bins(), [1, 2, 3])
        npt.assert_equal(nz.get_num_tomographic_bins(), 3)
        npt.assert_equal(len(nz.dz_dict), 3)

    def test_init_bad_nuisance_dict(self):
        # one key is 'dz_0_GC' instead of 'dz_1_GC -> KeyError
        npt.assert_raises(KeyError, RedshiftDistribution,
                          'GC', self.gauss_nz_dic, self.bad_nuisance_dic)

    def test_evaluates_n_i_z_zero_shift(self):
        nz = RedshiftDistribution('GC', self.gauss_nz_dic,
                                  self.zero_nuisance_dic)
        # Evaluate the n_i(z) gaussian distributions at the peak
        # The peaks should still be at [0.1, 0.2, 0.3]
        evaluated = \
            [nz.evaluates_n_i_z(i, i * 0.1) for i in self.bins]
        # The peak values should equal 1/sqrt(2pi)
        npt.assert_allclose(evaluated, self.peak_values, rtol=1e-3)

    def test_evaluates_n_i_z_only_shift(self):
        nz = RedshiftDistribution('GC', self.gauss_nz_dic,
                                  self.shift_only_nuisance_dic)
        # Evaluate the n_i(z) gaussian distributions at the peak
        # The peak should be shifted by 0.1 now
        evaluated = \
            [nz.evaluates_n_i_z(i, (i + 1) * 0.1) for i in self.bins]
        # The peak values should equal 1/sqrt(2pi)
        npt.assert_allclose(evaluated, self.peak_values, rtol=1e-3)

        # Evaluate the shifted n_i(z) gaussian distributions
        # at the un-shifted peak position i.e. [0.1, 0.2, 0.3]
        evaluated = \
            [nz.evaluates_n_i_z(i, i * 0.1) for i in self.bins]
        # The values should be equal to 1/sqrt(2pi) * norm(0.1)
        norm_at_0_1 = 0.9950124791926823
        desired = norm_at_0_1 * np.array(self.peak_values)
        # n(z-shifted) is zero below the first bin midpoint
        desired[0] = 0
        npt.assert_allclose(evaluated, desired, rtol=1e-3)

    def test_interpolates_n_i_z_only_shift(self):
        nz = RedshiftDistribution('GC', self.gauss_nz_dic,
                                  self.shift_only_nuisance_dic)
        # Interpolates the shifted n_i(z)  distributions over sef.zs
        # The peak should be shifted by 0.1 now
        interpolated = \
            {i: nz.interpolates_n_i(i, self.zs) for i in self.bins}
        interp_eval = [interpolated[i]((i + 1) * 0.1) for i in self.bins]
        # Values at the peaks should approximate to 1/sqrt(2pi)
        npt.assert_allclose(interp_eval, self.peak_values, rtol=1e-3)
