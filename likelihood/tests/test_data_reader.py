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
        self.fiducial_key_check = ['H0', 'omch2', 'ombh2', 'ns',
                                   'sigma_8_0', 'w', 'omkh2', 'omnuh2']
        self.cov_check_GC_spec = 10977788.704318
        self.cov_check_WL = 1.236327e-17
        self.cov_check_GC_phot = 1.515497e-11
        self.cov_check_XC = 1.842760e-18
        self.cl_phot_WL_check = 7.324317e-09
        self.cl_phot_GC_check = 2.898418e-05
        self.cl_phot_XC_check = 2.689051e-07

        # (GCH): added tests for n(z) data
        self.data_tester.compute_nz()
        self.nz_dict_GC_Phot_check = 3.190402
        self.nz_dict_WL_check = 3.190402

    def tearDown(self):
        self.main_key_check = None
        self.nz_key_check = None
        self.cov_check_GC_spec = None
        self.cov_check_WL = None
        self.cov_check_GC_phot = None
        self.cov_check_XC = None
        self.cl_phot_WL_check = None
        self.cl_phot_GC_check = None
        self.cl_phot_XC_check = None
        self.nz_dict_GC_Phot_check = None
        self.nz_dict_WL_check = None
        self.fiducial_key_check = None

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
        test_cov = self.data_tester.data_dict['GC-Spec']['1.2']['cov'][1, 1]
        npt.assert_allclose(test_cov, self.cov_check_GC_spec,
                            rtol=1e-3,
                            err_msg='Error in loading external spectroscopic'
                                    ' test data.')

    def test_bench_phot_cov_check(self):
        self.data_tester.read_phot()
        test_cov_WL = self.data_tester.data_dict['WL']['cov'][1, 1]
        test_cov_GC = self.data_tester.data_dict['GC-Phot']['cov'][1, 1]
        test_cov_XC = self.data_tester.data_dict['XC-Phot']['cov'][1, 1]
        npt.assert_allclose([test_cov_WL, test_cov_GC, test_cov_XC],
                            [self.cov_check_WL, self.cov_check_GC_phot,
                             self.cov_check_XC], rtol=1e-3,
                            err_msg='Error in loading external photometric'
                                    ' test covariance data.')

    def test_bench_phot_cls_check(self):
        self.data_tester.read_phot()
        test_Cl_WL = self.data_tester.data_dict['WL']['E1-E1'][1]
        test_Cl_GC = self.data_tester.data_dict['GC-Phot']['P1-P1'][1]
        test_Cl_XC = self.data_tester.data_dict['XC-Phot']['P1-E1'][1]
        npt.assert_allclose([test_Cl_WL, test_Cl_GC, test_Cl_XC],
                            [self.cl_phot_WL_check, self.cl_phot_GC_check,
                             self.cl_phot_XC_check], rtol=1e-3,
                            err_msg='Error in loading external photometric'
                                    ' test power spectra data.')

    def test_fiducial_reading_check(self):
        self.data_tester.read_GC_spec()
        npt.assert_equal(list(
                        self.data_tester.data_spec_fiducial_cosmo.keys()),
                        self.fiducial_key_check)
