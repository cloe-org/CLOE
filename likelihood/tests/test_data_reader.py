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
        self.main_key_check = ['GC-Spec', 'GC-Phot', 'WL', 'XC-Phot']
        self.nz_key_check = ['n1', 'n2', 'n3', 'n4', 'n5',
                             'n6', 'n7', 'n8', 'n9', 'n10']
        self.cov_check = 178.4167636283394
        # (GCH): added tests for n(z) data
        self.data_tester.compute_nz()
        self.nz_dict_GC_Phot_check = np.array([0.15197732])
        self.nz_dict_WL_check = np.array([0.15197732])

    def tearDown(self):
        self.main_key_check = None
        self.nz_key_check = None
        self.cov_check = None
        self.nz_dict_GC_Phot_check = None
        self.nz_dict_WL_check = None

    def test_data_dict_init(self):
        true_dict = list(self.data_tester.data_dict.keys())
        npt.assert_equal(true_dict, self.main_key_check,
                         err_msg='Error in data reading dict initialisation.')

    def test_nz_WL_dict_init(self):
        true_nz_dict = list(self.data_tester.nz_dict_WL.keys())
        npt.assert_equal(true_nz_dict, self.nz_key_check,
                         err_msg='Error in data reading nz WL '
                                 'dict initialisation.')

    def test_nz_GC_Phot_dict_init(self):
        true_nz_dict = list(self.data_tester.nz_dict_GC_Phot.keys())
        npt.assert_equal(true_nz_dict, self.nz_key_check,
                         err_msg='Error in data reading nz GC Phot '
                                 'dict initialisation.')

    def test_nz_GC_Phot_dict_interpolator(self):
        npt.assert_allclose(self.data_tester.nz_dict_GC_Phot['n1'](0.399987),
                            self.nz_dict_GC_Phot_check,
                            atol=1e-06,
                            err_msg='Error in the interpolation of '
                            'raw n(z) GC data')

    def test_nz_WL_dict_interpolator(self):
        npt.assert_allclose(self.data_tester.nz_dict_WL['n1'](0.399987),
                            self.nz_dict_WL_check,
                            atol=1e-06,
                            err_msg='Error in the interpolation of '
                            'raw n(z) WL data')

    def test_gc_fname_exception(self):
        npt.assert_raises(Exception, self.data_tester.read_GC_spec,
                          file_names='cov_power_galaxies_dk0p004.fits')

    def test_bench_gc_cov_check(self):
        self.data_tester.read_GC_spec()
        test_cov = self.data_tester.data_dict['GC-Spec']['z=1.2']['cov'][1, 1]
        npt.assert_allclose(test_cov, self.cov_check,
                            rtol=1e-3,
                            err_msg='Error in loading external spectroscopic'
                                    ' test data.')
