"""UNIT TESTS FOR DATA_READER

This module contains unit tests for the data_reader module.
=======

"""

from unittest import TestCase
import numpy.testing as npt
from cloe.data_reader.reader import Reader
from cloe.tests.test_input.data import mock_data


class datareaderTestCase(TestCase):

    def setUp(self):
        self.data_tester = Reader(mock_data)
        self.main_key_check = ['GC-Spectro', 'GC-Phot', 'WL', 'XC-Phot']
        self.nz_key_check = ['n1', 'n2', 'n3', 'n4', 'n5',
                             'n6', 'n7', 'n8', 'n9', 'n10']
        self.fiducial_key_check = ['H0', 'omch2', 'ombh2',
                                   'ns', 'sigma8_0', 'w',
                                   'omkh2', 'omnuh2', 'Omnu']
        self.cov_check_GC_spectro = 1.217193e+08
        self.cov_check_3x2pt = 0.016542
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
        npt.assert_raises(Exception, self.data_tester.read_GC_spectro,
                          file_names='cov_power_galaxies_dk0p004.fits')

    def test_gc_spec_cov_num_check_cov_nsim_exists(self):
        mock_data['spectro']['cov_is_num'] = True
        if 'cov_nsim' in mock_data['spectro'].keys():
            mock_data['spectro'].pop('cov_nsim')
        npt.assert_raises(Exception, self.data_tester.read_GC_spectro)
        mock_data['spectro']['cov_is_num'] = False

    def test_gc_spec_cov_num_check_cov_nsim_is_number(self):
        mock_data['spectro']['cov_is_num'] = True
        mock_data['spectro']['cov_nsim'] = None
        npt.assert_raises(TypeError, self.data_tester.read_GC_spectro)
        mock_data['spectro']['cov_is_num'] = False
        mock_data['spectro'].pop('cov_nsim')

    def test_gc_spec_cov_num_check_cov_nsim_is_positive(self):
        mock_data['spectro']['cov_is_num'] = True
        mock_data['spectro']['cov_nsim'] = -10
        npt.assert_raises(ValueError, self.data_tester.read_GC_spectro)
        mock_data['spectro']['cov_is_num'] = False
        mock_data['spectro'].pop('cov_nsim')

    def test_phot_cov_num_check_cov_nsim_exists(self):
        mock_data['photo']['cov_is_num'] = True
        npt.assert_raises(Exception, self.data_tester.read_phot)
        mock_data['photo']['cov_is_num'] = False

    def test_phot_cov_num_check_cov_nsim_is_number(self):
        mock_data['photo']['cov_is_num'] = True
        mock_data['photo']['cov_nsim'] = None
        npt.assert_raises(TypeError, self.data_tester.read_phot)
        mock_data['photo']['cov_is_num'] = False
        mock_data['photo'].pop('cov_nsim')

    def test_photo_cov_num_check_cov_nsim_is_positive(self):
        mock_data['photo']['cov_is_num'] = True
        mock_data['photo']['cov_nsim'] = -10
        npt.assert_raises(ValueError, self.data_tester.read_phot)
        mock_data['photo']['cov_is_num'] = False
        mock_data['photo'].pop('cov_nsim')

    def test_bench_gc_cov_check(self):
        self.data_tester.read_GC_spectro()
        test_cov = self.data_tester.data_dict['GC-Spectro']['1.2']['cov'][1, 1]
        npt.assert_allclose(test_cov, self.cov_check_GC_spectro,
                            rtol=1e-3,
                            err_msg='Error in loading external spectroscopic'
                                    ' test data.')

    def test_bench_phot_cov_check(self):
        self.data_tester.read_phot()
        num_ells = len(self.data_tester.data_dict['WL']['ells'])
        num_WL_bins = self.data_tester.numtomo_wl
        num_GC_bins = self.data_tester.numtomo_gcphot
        shape_WL = int(num_WL_bins * (num_WL_bins + 1) / 2) * num_ells
        shape_XC = int(num_WL_bins * num_GC_bins) * num_ells
        shape_GC = int(num_GC_bins * (num_GC_bins + 1) / 2) * num_ells
        shape_3x2pt = shape_WL + shape_XC + shape_GC

        test_cov_3x2pt = 0.0

        for i in range(shape_3x2pt):
            for j in range(shape_3x2pt):
                bin_i = i
                bin_j = j
                test_cov_3x2pt += i * j * \
                    self.data_tester.data_dict['3x2pt_cov'][bin_i, bin_j]

        npt.assert_allclose(test_cov_3x2pt,
                            [self.cov_check_3x2pt], rtol=1e-3,
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
        self.data_tester.read_GC_spectro()
        npt.assert_equal(list(
                        self.data_tester.data_spectro_fiducial_cosmo.keys()),
                        self.fiducial_key_check)
