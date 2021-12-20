from unittest import TestCase
from likelihood.masking.data_handler import Data_handler
import numpy.testing as npt
import numpy as np
from likelihood.tests.test_input.mock_observables import build_mock_observables
from likelihood.tests.test_input.data import mock_data
from likelihood.data_reader.reader import Reader


class datahandlerTestCase(TestCase):

    @classmethod
    def setUpClass(self):
        self.wl_size = 1100
        self.xc_phot_size = 2000
        self.gc_phot_size = 1100
        self.tx2_size = self.wl_size + self.xc_phot_size + self.gc_phot_size
        self.gc_spectro_size = 1500
        self.datavec_shape = (self.wl_size +
                              self.xc_phot_size +
                              self.gc_phot_size +
                              self.gc_spectro_size,)
        self.cov_shape = (self.datavec_shape[0], self.datavec_shape[0])

        self.wl_vec = np.random.randn(self.wl_size)
        self.xc_phot_vec = np.random.randn(self.xc_phot_size)
        self.gc_phot_vec = np.random.randn(self.gc_phot_size)
        self.gc_spectro_vec = np.random.randn(self.gc_spectro_size)

        self.wl_cov = np.random.randn(self.wl_size, self.wl_size)
        self.xc_phot_cov = np.random.randn(self.xc_phot_size,
                                           self.xc_phot_size)
        self.gc_phot_cov = np.random.randn(self.gc_phot_size,
                                           self.gc_phot_size)
        self.tx2_cov = np.random.randn(self.tx2_size, self.tx2_size)
        self.gc_spectro_cov = np.random.randn(self.gc_spectro_size,
                                              self.gc_spectro_size)

        data = {'WL': self.wl_vec,
                'XC-Phot': self.xc_phot_vec,
                'GC-Phot': self.gc_phot_vec,
                'GC-Spectro': self.gc_spectro_vec}
        cov = {'3x2pt': self.tx2_cov,
               'GC-Spectro': self.gc_spectro_cov}
        self.data_reader = Reader(mock_data)
        self.data_reader.compute_nz()
        self.data_reader.read_GC_spectro()
        self.data_reader.read_phot()

        self.observables = build_mock_observables(self.data_reader)
        self.data_handler = Data_handler(
            data, cov, self.observables, self.data_reader)
        self.masking_vector = np.ones(self.datavec_shape[0]).astype(int)

        if not self.observables['selection']['WL']['WL']:
            self.masking_vector[0:self.wl_size] = 0
        if not self.observables['selection']['WL']['GCphot']:
            self.masking_vector[self.wl_size:
                                (self.wl_size + self.xc_phot_size)] = 0
        if not self.observables['selection']['GCphot']['GCphot']:
            self.masking_vector[(self.wl_size + self.xc_phot_size):
                                (self.wl_size +
                                 self.xc_phot_size +
                                 self.gc_phot_size)] = 0
        if not self.observables['selection']['GCspectro']['GCspectro']:
            self.masking_vector[(self.wl_size +
                                 self.xc_phot_size +
                                 self.gc_phot_size):
                                (self.wl_size +
                                 self.xc_phot_size +
                                 self.gc_phot_size +
                                 self.gc_spectro_size)] = 0

    @classmethod
    def tearDownClass(self):
        self.wl_size = None
        self.xc_phot_size = None
        self.gc_phot_size = None
        self.tx2_size = None
        self.gc_spectro_size = None
        self.datavec_size = None
        self.wl_vec = None
        self.xc_phot_vec = None
        self.gc_phot_vec = None
        self.gc_spectro_vec = None
        self.data_handler = None

    def test_create_data_vector(self):
        self.data_handler._create_data_vector()

        npt.assert_equal(self.data_handler._data_vector.shape,
                         self.datavec_shape,
                         err_msg=f'Shape of full data vector does not match'
                         f' the expected shape.')

    def test_create_cov_matrix(self):
        self.data_handler._create_cov_matrix()

        npt.assert_equal(self.data_handler._cov_matrix.shape,
                         self.cov_shape,
                         err_msg=f'Shape of full covariance matrix'
                         f' does not match the expected shape.')

    def test_create_masking_vector(self):
        self.data_handler._create_masking_vector(self.data_reader)

        npt.assert_equal(self.data_handler._masking_vector,
                         self.masking_vector,
                         err_msg=f'Values of masking vector do not match'
                         f' expected values.')

    # test whether the observables selection is read correctly from the input
    # dictionary: assing random values to the input flags and check whether
    # the same values are obtained by calling the corresponding getters
    def test_observables_selection(self):
        data = {'WL': self.wl_vec,
                'XC-Phot': self.xc_phot_vec,
                'GC-Phot': self.gc_phot_vec,
                'GC-Spectro': self.gc_spectro_vec}
        cov = {'3x2pt': self.tx2_cov,
               'GC-Spectro': self.gc_spectro_cov}

        use_wl, use_xc_phot, use_gc_phot, use_gc_spectro = (
            np.random.choice([True, False], size=4)
        )
        obs = {'WL': {'WL': use_wl, 'GCphot': use_xc_phot, 'GCspectro': False},
               'GCphot': {'GCphot': use_gc_phot, 'GCspectro': False},
               'GCspectro': {'GCspectro': use_gc_spectro}}
        observables = self.observables
        observables['selection'] = obs
        data_handler = Data_handler(data, cov, observables, self.data_reader)
        npt.assert_equal(data_handler.use_wl, use_wl,
                         err_msg=f'Unexpected value of use_wl flag:'
                         f' {data_handler.use_wl} instead of {use_wl}')
        npt.assert_equal(data_handler.use_xc_phot, use_xc_phot,
                         err_msg=f'Unexpected value of use_xc_phot flag:'
                         f' {data_handler.use_xc_phot} instead of'
                         f' {use_xc_phot}')
        npt.assert_equal(data_handler.use_gc_phot, use_gc_phot,
                         err_msg=f'Unexpected value of use_gc_phot flag:'
                         f' {data_handler.use_gc_phot} instead of'
                         f' {use_gc_phot}')
        npt.assert_equal(data_handler.use_gc_spectro, use_gc_spectro,
                         err_msg=f'Unexpected value of use_wl flag:'
                         f' {data_handler.use_gc_spectro} instead of'
                         f' {use_gc_spectro}')

    # test whether the size of the gc_spectro part of the data vector is as
    # expected: just read the gc_spectro_size property of the data handler
    # and check whether it matches the value specified in setUp()
    def test_spectro_data_size(self):
        npt.assert_equal(self.data_handler.gc_spectro_size,
                         self.gc_spectro_size,
                         err_msg=f'Unexpected size of the gc_spectro part of'
                         f' the data vector:'
                         f' {self.data_handler.gc_spectro_size} instead of'
                         f' {self.gc_spectro_size}')
