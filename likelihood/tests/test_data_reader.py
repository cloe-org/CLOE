"""UNIT TESTS FOR DATA_READER

This module contains unit tests for the data_reader module.
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from ..data_reader import reader


class datareaderTestCase(TestCase):

    def setUp(self):
        self.data_tester = reader.Reader()
        self.main_key_check = ['GC-Spec', 'GC-Phot', 'WL']
        self.nz_key_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.cov_check = 178.4167636283394

    def tearDown(self):
        self.main_key_check = None
        self.nz_key_check = None
        self.cov_check = None

    def test_data_dict_init(self):
        true_dict = list(self.data_tester.data_dict.keys())
        npt.assert_equal(true_dict, self.main_key_check,
                         err_msg='Error in data reading dict initialisation.')

    def test_nz_dict_init(self):
        true_nz_dict = list(self.data_tester.nz_dict.keys())
        npt.assert_equal(true_nz_dict, self.nz_key_check,
                         err_msg='Error in data reading nz '
                                 'dict initialisation.')

    def test_bench_gc_cov_check(self):
        self.data_tester.read_GC_spec()
        test_cov = self.data_tester.data_dict['GC-Spec']['z=1.2']['cov'][1, 1]
        npt.assert_allclose(test_cov, self.cov_check,
                            rtol=1e-3,
                            err_msg='Error in loading external spectroscopic'
                                    ' test data.')
