"""UNIT TESTS FOR REDSHIFT_BINS

This module contains unit tests
for the :obj:`redshift_bins` module.

"""

import numpy as np
import numpy.testing as npt
from unittest import TestCase
import cloe.auxiliary.redshift_bins as rb


class RedshiftBinsTestCase(TestCase):

    def setUp(self):
        self.edges = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
        self.z_list = [0.9, 1.5, 3.2, 8.1]
        self.z_ndarray = np.array(self.z_list)
        self.z_in_1 = 1.5
        self.z_in_2 = 3.2
        self.z_low = 0.9
        self.z_high = 8.1

    def tearDown(self):
        self.edges = None

    def test_find_bin_scalar(self):
        # test zs at the 1-st bin lower edge
        npt.assert_equal(rb.find_bin(self.edges[0], self.edges),
                         1,
                         err_msg='Error in find_bin')
        # test zs in 1-st bin
        npt.assert_equal(rb.find_bin(self.z_in_1, self.edges),
                         1,
                         err_msg='Error in find_bin')
        # test zs in 2-nd bin
        npt.assert_equal(rb.find_bin(self.z_in_2, self.edges),
                         2,
                         err_msg='Error in find_bin')
        # test zs below the range without check_bounds
        npt.assert_equal(rb.find_bin(self.z_low, self.edges),
                         0,
                         err_msg='Error in find_bin')
        # test zs at the high boundary with check_bounds
        npt.assert_raises(ValueError,
                          rb.find_bin, zs=self.edges[-1],
                          redshift_edges=self.edges,
                          check_bounds=True)
        # test zs above the range with check_bounds
        npt.assert_raises(ValueError,
                          rb.find_bin, zs=self.z_high,
                          redshift_edges=self.edges,
                          check_bounds=True)
        # test zs below the range with check_bounds
        npt.assert_raises(ValueError,
                          rb.find_bin, zs=self.z_low,
                          redshift_edges=self.edges,
                          check_bounds=True)

    def test_find_indices_vector(self):
        # test zs in range
        test_zs_list = [self.z_in_1, self.z_in_2]
        test_zs_ndarray = np.array(test_zs_list)
        desired_bins = [1, 2]
        # test zs in range as list
        actual_bins = rb.find_bin(test_zs_list, self.edges)
        npt.assert_array_equal(actual_bins,
                               desired_bins,
                               err_msg='Error in find_bin (list)')
        # test zs in range as a ndarray
        actual_bins = rb.find_bin(test_zs_ndarray, self.edges)
        npt.assert_array_equal(actual_bins,
                               desired_bins,
                               err_msg='Error in find_bin (ndarray)')
        # test zs outside range (low) with check_boundaries
        test_zs_ndarray[0] = 0.5
        npt.assert_raises(ValueError,
                          rb.find_bin,
                          zs=test_zs_ndarray,
                          redshift_edges=self.edges,
                          check_bounds=True)
        # test zs outside range (high) with check_boundaries
        test_zs_ndarray = [1.1, 8.1]
        npt.assert_raises(ValueError,
                          rb.find_bin,
                          zs=test_zs_ndarray,
                          redshift_edges=self.edges,
                          check_bounds=True)

    def test_reduce(self):
        desired_reduced = np.array([2.0, 4.0], dtype=float)
        npt.assert_equal(rb.reduce(self.edges, 2, 5.0), desired_reduced,
                         err_msg='Error in reduce')
