from unittest import TestCase
from unittest.mock import patch
from cloe.masking.masking import Masking
import numpy.testing as npt
import numpy


class Masking_test(TestCase):

    def setUp(self):
        self.fake_vec_size = 100
        self.fake_masking_vector = (
            numpy.random.randint(2, size=self.fake_vec_size))
        self.fake_data_vector = numpy.random.randn(self.fake_vec_size)
        self.fake_theory_vector = numpy.random.randn(self.fake_vec_size)
        self.fake_covariance_matrix = (
            numpy.random.randn(self.fake_vec_size, self.fake_vec_size))
        self.covariance_matrix_not_2d = numpy.random.randn(10, 10, 10)
        self.covariance_matrix_not_square = numpy.random.randn(3, 2)
        self.masking = Masking()
        self.masking.set_data_vector(self.fake_data_vector)
        self.masking.set_theory_vector(self.fake_theory_vector)
        self.masking.set_covariance_matrix(
            self.fake_covariance_matrix)
        self.masking.set_masking_vector(self.fake_masking_vector)
        self.masked_vec_size = numpy.count_nonzero(self.fake_masking_vector)

    def tearDown(self):
        self.fake_vec_size = None
        self.fake_masking_vector = None
        self.fake_data_vector = None
        self.fake_covariance_matrix = None
        self.covariance_matrix_not_2d = None
        self.covariance_matrix_not_square = None
        self.masking = None
        self.masked_vec_size = None

    # test that calling set_masking_vector() actually sets the masking vector
    # and resets the value of masked data vector and masked covariance matrix
    def test_set_masking_vector(self):
        self.masking.get_masked_data_vector()
        self.masking.get_masked_covariance_matrix()
        self.masking.set_masking_vector(self.fake_masking_vector)

        npt.assert_array_equal(
            self.masking._masking_vector,
            self.fake_masking_vector,
            err_msg=f'masking vector was not set in set_masking_vector()'
        )
        self.assertIsNone(
            self.masking._masked_data_vector,
            msg=f'masked data vector is not None after calling'
            f' set_masking_vector()'
        )
        self.assertIsNone(
            self.masking._masked_covariance_matrix,
            msg=f'masked covariance matrix is not None after calling'
            f' set_masking_vector()')

    # test that the data vector is set and the masked data vector is reset
    def test_set_data_vector(self):
        self.masking.get_masked_data_vector()
        self.masking.set_data_vector(self.fake_data_vector)
        npt.assert_array_equal(
            self.masking._data_vector,
            self.fake_data_vector,
            err_msg=f'data vector was not set in set_data_vector()'
        )
        self.assertIsNone(
            self.masking._masked_data_vector,
            msg=f'masked data vector was not reset in set_data_vector()'
        )

    # test that the theory vector is set and the masked theory vector is reset
    def test_set_theory_vector(self):
        self.masking.get_masked_theory_vector()
        self.masking.set_theory_vector(self.fake_theory_vector)
        npt.assert_array_equal(
            self.masking._theory_vector,
            self.fake_theory_vector,
            err_msg=f'theory vector was not set in set_theory_vector()'
        )
        self.assertIsNone(
            self.masking._masked_theory_vector,
            msg=f'masked theory vector was not reset in set_theory_vector()'
        )

    # test that the covariance matrix is set and the masked covariance matrix
    # is reset
    def test_set_covariance_matrix(self):
        self.masking.get_masked_covariance_matrix()
        self.masking.set_covariance_matrix(self.fake_covariance_matrix)
        npt.assert_array_equal(
            self.masking._covariance_matrix,
            self.fake_covariance_matrix,
            err_msg=f'covariance_matrix was not set in'
            f' set_covariance_matrix()'
        )
        self.assertIsNone(
            self.masking._masked_covariance_matrix,
            msg=f'masked covariance matrix was not reset in'
            f' set_covariance_matrix()'
        )

    # test that an error is raised if the covariance matrix has a bad format
    def test_set_covariance_matrix_invalid(self):
        self.assertRaises(
            TypeError,
            self.masking.set_covariance_matrix,
            self.covariance_matrix_not_2d,
        )

        self.assertRaises(
            TypeError,
            self.masking.set_covariance_matrix,
            self.covariance_matrix_not_square,
        )

    # Test that the masked data vector has the expected size.
    # Test that an error is raised when the data vector or the masking
    # vector are not set.
    def test_get_masked_data_vector(self):
        self.assertEqual(
            self.masked_vec_size,
            len(self.masking.get_masked_data_vector()),
            msg=f'Masked data vector has unexpected size:'
            f' {len(self.masking.get_masked_data_vector())} instead of'
            f' {self.masked_vec_size}'
        )

        self.masking._masking_vector = None
        self.assertRaises(
            TypeError,
            self.masking.get_masked_data_vector,
        )

        self.masking._masking_vector = self.fake_masking_vector
        self.masking._data_vector = None
        self.assertRaises(
            TypeError,
            self.masking.get_masked_data_vector,
        )

    # Test that the masked theory vector has the expected size.
    # Test that an error is raised when the theory vector or the masking
    # vector are not set.
    def test_get_masked_theory_vector(self):
        self.assertEqual(
            self.masked_vec_size,
            len(self.masking.get_masked_theory_vector()),
            msg=f'Masked theory vector has unexpected size:'
            f' {len(self.masking.get_masked_theory_vector())} instead of'
            f' {self.masked_vec_size}'
        )

        self.masking._masking_vector = None
        self.assertRaises(
            TypeError,
            self.masking.get_masked_theory_vector,
        )

        self.masking._masking_vector = self.fake_masking_vector
        self.masking._theory_vector = None
        self.assertRaises(
            TypeError,
            self.masking.get_masked_theory_vector,
        )

    # Test that the masked covariance matrix has the expected dimensions.
    # Test that an error is raised when the masking vector or the covariance
    # matrix are not set.
    def test_get_masked_covariance_matrix(self):

        npt.assert_array_equal(
            (self.masked_vec_size, self.masked_vec_size),
            self.masking.get_masked_covariance_matrix().shape,
            err_msg=f'Masked covariance matrix has unexpected shape:'
            f' {self.masking.get_masked_covariance_matrix().shape}'
            f' instead of ({self.masked_vec_size}, {self.masked_vec_size}),'
        )

        # repeat the call to the function, just for the sake of coverage
        self.masking.get_masked_covariance_matrix()

        self.masking._masking_vector = None
        self.assertRaises(
            TypeError,
            self.masking.get_masked_covariance_matrix,
        )

        self.masking._masking_vector = self.fake_masking_vector
        self.masking._covariance_matrix = None
        self.assertRaises(
            TypeError,
            self.masking.get_masked_covariance_matrix,
        )
