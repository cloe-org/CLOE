from unittest import TestCase
from likelihood.masking.data_handler import Data_handler
import numpy.testing as npt
import numpy as np


class datahandlerTestCase(TestCase):

    def setUp(self):
        self.wl_size = 1100
        self.xc_phot_size = 2000
        self.gc_phot_size = 1100
        self.gc_spectro_size = 1500
        self.datavec_shape = (self.wl_size +
                              self.xc_phot_size +
                              self.gc_phot_size +
                              self.gc_spectro_size,)
        self.invcov_shape = (self.datavec_shape[0], self.datavec_shape[0])

        self.wl_vec = np.random.randn(self.wl_size)
        self.xc_phot_vec = np.random.randn(self.xc_phot_size)
        self.gc_phot_vec = np.random.randn(self.gc_phot_size)
        self.gc_spectro_vec = np.random.randn(self.gc_spectro_size)

        self.wl_cov = np.random.randn(self.wl_size, self.wl_size)
        self.xc_phot_cov = np.random.randn(self.xc_phot_size,
                                           self.xc_phot_size)
        self.gc_phot_cov = np.random.randn(self.gc_phot_size,
                                           self.gc_phot_size)
        self.gc_spectro_cov = np.random.randn(self.gc_spectro_size,
                                              self.gc_spectro_size)

        data = {'WL': self.wl_vec,
                'XC-Phot': self.xc_phot_vec,
                'GC-Phot': self.gc_phot_vec,
                'GC-Spectro': self.gc_spectro_vec}
        cov = {'WL': self.wl_cov,
               'XC-Phot': self.xc_phot_cov,
               'GC-Phot': self.gc_phot_cov,
               'GC-Spectro': self.gc_spectro_cov}
        obs = {'WL': {'WL': True, 'GCphot': False, 'GCspectro': False},
               'GCphot': {'GCphot': True, 'GCspectro': False},
               'GCspectro': {'GCspectro': True}}

        self.data_handler = Data_handler(data, cov, obs)
        self.masking_vector = np.ones(self.datavec_shape[0]).astype(int)
        self.masking_vector[self.wl_size:(self.wl_size +
                                          self.xc_phot_size)] = 0

    def tearDown(self):
        self.wl_size = None
        self.xc_phot_size = None
        self.gc_phot_size = None
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

    def test_create_invcov_matrix(self):
        self.data_handler._create_invcov_matrix()

        npt.assert_equal(self.data_handler._invcov_matrix.shape,
                         self.invcov_shape,
                         err_msg=f'Shape of full inverse covariance matrix'
                         f' does not match the expected shape.')

    def test_create_masking_vector(self):
        self.data_handler._create_masking_vector()

        npt.assert_equal(self.data_handler._masking_vector,
                         self.masking_vector,
                         err_msg=f'Values of masking vector do not match'
                         f' expected values.')
