"""UNIT TESTS FOR DATA_READER

This module contains unit tests for the data_reader module.
=======

"""

from unittest import TestCase
import numpy.testing as npt
from likelihood.data_reader.reader import Reader


class datareaderTestCase(TestCase):

    def setUp(self):
        mock_data = {
          'sample': 'ExternalBenchmark',
          'spec': {
            'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
            'redshifts': ["1.", "1.2", "1.4", "1.65"]},
          'photo': {
            'ndens_GC': 'niTab-EP10-RB00.dat',
            'ndens_WL': 'niTab-EP10-RB00.dat',
            'root_GC': 'Cls_{:s}_PosPos.fits',
            'root_WL': 'Cls_{:s}_ShearShear.fits',
            'root_XC': 'Cls_{:s}_PosShear.fits',
            'IA_model': 'zNLA',
            'cov_GC': 'CovMat-PosPos-{:s}-20Bins.dat',
            'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.dat',
            'cov_3x2': 'CovMat-3x2pt-{:s}-20Bins.dat',
            'cov_model': 'Gauss'}
        }

        self.data_tester = Reader(mock_data)
        self.main_key_check = ['GC-Spec', 'GC-Phot', 'WL', 'XC-Phot']
        self.nz_key_check = ['n1', 'n2', 'n3', 'n4', 'n5',
                             'n6', 'n7', 'n8', 'n9', 'n10']
        self.fiducial_key_check = ['H0', 'omch2', 'ombh2',
                                   'ns', 'sigma8_0', 'w',
                                   'omkh2', 'omnuh2', 'Omnu']
        self.cov_check_GC_spec = 1.217193e+08
        self.cov_check_WL = 2.654605e-09
        self.cov_check_GC_phot = 1.693992e-05
        self.cov_check_XC = 4.403701e-04
        self.cl_phot_WL_check = 7.144612e-05
        self.cl_phot_GC_check = 2.239632e-03
        self.cl_phot_XC_check = 2.535458e-04

        # Added tests for n(z) data
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
        shape_WL = self.data_tester.data_dict['WL']['cov'].shape
        shape_GC = self.data_tester.data_dict['GC-Phot']['cov'].shape
        shape_XC = self.data_tester.data_dict['XC-Phot']['cov'].shape

        test_cov_WL = 0.0
        test_cov_GC = 0.0
        test_cov_XC = 0.0

        for i in range(shape_WL[0]):
            for j in range(shape_WL[1]):
                test_cov_WL += (i * j * self.data_tester.data_dict['WL'][
                                'cov'][i, j])

        for i in range(shape_GC[0]):
            for j in range(shape_GC[1]):
                test_cov_GC += (i * j * self.data_tester.data_dict['GC-Phot'][
                                'cov'][i, j])

        for i in range(shape_XC[0]):
            for j in range(shape_XC[1]):
                test_cov_XC += (i * j * self.data_tester.data_dict['XC-Phot'][
                                'cov'][i, j])

        npt.assert_allclose([test_cov_WL, test_cov_GC, test_cov_XC],
                            [self.cov_check_WL, self.cov_check_GC_phot,
                             self.cov_check_XC], rtol=1e-3,
                            err_msg='Error in loading external photometric'
                                    ' test covariance data.')

    def test_bench_phot_cls_check(self):
        self.data_tester.read_phot()
        test_Cl_WL = 0.0
        test_Cl_GC = 0.0
        test_Cl_XC = 0.0

        WL_arr = self.data_tester.data_dict['WL']
        GC_arr = self.data_tester.data_dict['GC-Phot']
        XC_arr = self.data_tester.data_dict['XC-Phot']

        for i in range(1, 11):
            for j in range(i, 11):
                test_Cl_WL += (i * j * WL_arr['E{:d}-E{:d}'.format(i, j)][1])
                test_Cl_GC += (i * j * GC_arr['P{:d}-P{:d}'.format(i, j)][1])
                test_Cl_XC += (i * j * XC_arr['P{:d}-E{:d}'.format(i, j)][1])

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
