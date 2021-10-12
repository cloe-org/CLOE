from unittest import TestCase
from unittest.mock import patch
from likelihood.masking.masking_vector_wrapper import MaskingVectorWrapper
import numpy.testing as npt
import numpy
import random


class MaskingVectorWrapper_test(TestCase):

    def setUp(self):
        self.masking_vector_wrapper = MaskingVectorWrapper()
        self.wl_first_element = 0
        self.xc_phot_first_element = self.wl_first_element + 1100
        self.gc_phot_first_element = self.xc_phot_first_element + 2000
        self.gc_spectro_first_element = self.gc_phot_first_element + 1100
        self.masking_vector_size = 5700
        self.masking_vector = numpy.zeros(self.masking_vector_size)

    def tearDown(self):
        self.masking_vector_wrapper = None
        self.wl_first_element = None
        self.xc_phot_first_element = None
        self.gc_phot_first_element = None
        self.gc_spectro_first_element = None
        self.masking_vector_size = None
        self.masking_vector = None

    def util_test_reset_enable_flags(self, caller):
        self.assertIsNone(
            self.masking_vector_wrapper._wl_enabled,
            msg=f'wl_enabled flag is not None after calling {caller}'
        )
        self.assertIsNone(
            self.masking_vector_wrapper._xc_phot_enabled,
            msg=f'xc_phot_enabled flag is not None after calling {caller}'
        )
        self.assertIsNone(
            self.masking_vector_wrapper._gc_phot_enabled,
            msg=f'gc_phot_enabled flag is not None after calling {caller}'
        )
        self.assertIsNone(
            self.masking_vector_wrapper._gc_spectro_enabled,
            msg=f'gc_spectro_enabled flag is not None after calling {caller}'
        )

    # test that the masking vector and the enable flags are set to None in
    # the initialization
    def test_init(self):
        self.assertIsNone(
            self.masking_vector_wrapper._masking_vector,
            msg=f'masking vector is not none after calling __init__()'
        )
        self.util_test_reset_enable_flags(caller='__init__()')

    # Test that calling set_masking_vector with a properly formatted masking
    # vector actually sets the masking vector, and that the enable flags are
    # reset.
    def test_set_masking_vector(self):
        random_masking_vector = (
            numpy.random.randint(2, size=self.masking_vector_size))

        self.masking_vector_wrapper.set_masking_vector(random_masking_vector)
        npt.assert_array_equal(
            random_masking_vector,
            self.masking_vector_wrapper._masking_vector,
            err_msg=f'masking vector was not set in set_masking_vector()'
        )
        self.util_test_reset_enable_flags(caller='set_masking_vector()')

    # Test that when the masking vector has an unexpected size, an exception
    # is raised, and nothing is changed in the status of the object.
    def test_set_invalid_masking_vector(self):
        # first step: set a valid masking vector and call the getters so that
        # the boolean flags are evaluated
        self.masking_vector_wrapper.set_masking_vector(self.masking_vector)
        self.masking_vector_wrapper.get_wl_enabled()
        self.masking_vector_wrapper.get_xc_phot_enabled()
        self.masking_vector_wrapper.get_gc_phot_enabled()
        self.masking_vector_wrapper.get_gc_spectro_enabled()
        # second step: set an invalid masking vector and verify that the
        # boolean flags are not reset.
        self.assertRaises(ValueError,
                          self.masking_vector_wrapper.set_masking_vector,
                          numpy.zeros(self.masking_vector_size - 1))

        self.assertIsNotNone(self.masking_vector_wrapper._wl_enabled,
                             msg=f'wl_enabled flag is modified after a'
                             f' failing call to set_masking_vector()')
        self.assertIsNotNone(self.masking_vector_wrapper._xc_phot_enabled,
                             msg=f'xc_phot_enabled flag is modified after a'
                             f' failing call to set_masking_vector()')
        self.assertIsNotNone(self.masking_vector_wrapper._gc_phot_enabled,
                             msg=f'gc_phot_enabled flag is modified after a'
                             f' failing call to set_masking_vector()')
        self.assertIsNotNone(self.masking_vector_wrapper._gc_spectro_enabled,
                             msg=f'gc_spectro_enabled flag is modified after a'
                             f' failing call to set_masking_vector()')

    # test that all the enable flags are reset in reset_enable_flags()
    def test_reset_enable_flags(self):
        self.util_test_reset_enable_flags(caller='reset_enable_flags()')

    # test that get_wl_enabled() returns true when the appropriate elements
    # are set to 1, false otherwise
    def test_get_wl_enabled(self):
        # set to 1 elements of the masking vector corresponding to wl
        self.masking_vector[self.wl_first_element:
                            self.xc_phot_first_element] = (
                numpy.ones(self.xc_phot_first_element - self.wl_first_element))

        # verify that with this masking vector, get_wl_enabled() returns True
        self.masking_vector_wrapper.set_masking_vector(self.masking_vector)
        self.assertTrue(self.masking_vector_wrapper.get_wl_enabled(),
                        msg=f'get_wl_enabled() does not return True when'
                        f' the wl flag is enabled')

        # verify that with the negated masking vector, get_wl_enabled()
        # returns False
        self.masking_vector_wrapper.set_masking_vector(
            numpy.logical_not(self.masking_vector).astype(int))
        self.assertFalse(self.masking_vector_wrapper.get_wl_enabled(),
                         msg=f'get_wl_enabled() does not return False when'
                         f' the wl flag is not enabled')

        # repeat the call, for increasing the coverage
        self.masking_vector_wrapper.get_wl_enabled()

    # test that get_xc_phot_enabled() returns true when the appropriate
    # elements are set to 1, false otherwise
    def test_get_xc_phot_enabled(self):
        # set to 1 elements of the masking vector corresponding to xc_phot
        self.masking_vector[self.xc_phot_first_element:
                            self.gc_phot_first_element] = (
                numpy.ones(self.gc_phot_first_element -
                           self.xc_phot_first_element))

        # verify that with this masking vector, get_xc_phot_enabled()
        # returns True
        self.masking_vector_wrapper.set_masking_vector(self.masking_vector)
        self.assertTrue(self.masking_vector_wrapper.get_xc_phot_enabled(),
                        msg=f'get_xc_phot_enabled() does not return True when'
                        f' the xc_phot flag is enabled')

        # verify that with the negated masking vector, get_xc_phot_enabled()
        # returns False
        self.masking_vector_wrapper.set_masking_vector(
            numpy.logical_not(self.masking_vector).astype(int))
        self.assertFalse(self.masking_vector_wrapper.get_xc_phot_enabled(),
                         msg=f'get_xc_phot_enabled() does not return False'
                         f' when the xc_phot flag is not enabled')

        # repeat the call, for increasing the coverage
        self.masking_vector_wrapper.get_xc_phot_enabled()

    # test that get_gc_phot_enabled() returns true when the appropriate
    # elements are set to 1, false otherwise
    def test_get_gc_phot_enabled(self):
        # set to 1 elements of the masking vector corresponding to gc_phot
        self.masking_vector[self.gc_phot_first_element:
                            self.gc_spectro_first_element] = (
                numpy.ones(self.gc_spectro_first_element -
                           self.gc_phot_first_element))

        # verify that with this masking vector, get_gc_phot_enabled()
        # returns True
        self.masking_vector_wrapper.set_masking_vector(self.masking_vector)
        self.assertTrue(self.masking_vector_wrapper.get_gc_phot_enabled(),
                        msg=f'get_gc_phot_enabled() does not return True when'
                        f' the gc_phot flag is enabled')

        # verify that with the negated masking vector, get_gc_phot_enabled()
        # returns False
        self.masking_vector_wrapper.set_masking_vector(
            numpy.logical_not(self.masking_vector).astype(int))
        self.assertFalse(self.masking_vector_wrapper.get_gc_phot_enabled(),
                         msg=f'get_gc_phot_enabled() does not return False'
                         f' when the gc_phot flag is not enabled')

        # repeat the call, for increasing the coverage
        self.masking_vector_wrapper.get_gc_phot_enabled()

    # test that get_gc_spectro_enabled() returns true when the appropriate
    # elements are set to 1, false otherwise
    def test_get_gc_spectro_enabled(self):
        # set to 1 elements of the masking vector corresponding to gc_phot
        self.masking_vector[self.gc_spectro_first_element:
                            self.masking_vector_size] = (
                numpy.ones(self.masking_vector_size -
                           self.gc_spectro_first_element))

        # verify that with this masking vector, get_gc_spectro_enabled()
        # returns True
        self.masking_vector_wrapper.set_masking_vector(self.masking_vector)
        self.assertTrue(self.masking_vector_wrapper.get_gc_spectro_enabled(),
                        msg=f'get_gc_spectro_enabled() does not return True'
                        f' when the gc_specto flag is enabled')

        # verify that with the negated masking vector, get_gc_spectro_enabled()
        # returns False
        self.masking_vector_wrapper.set_masking_vector(
            numpy.logical_not(self.masking_vector).astype(int))
        self.assertFalse(self.masking_vector_wrapper.get_gc_spectro_enabled(),
                         msg=f'get_gc_spectro_enabled() does not return False'
                         f' when the gc_spectro flag is not enabled')

        # repeat the call, for increasing the coverage
        self.masking_vector_wrapper.get_gc_spectro_enabled()

    # test that a RuntimeError is raised if the masking vector is not set
    def test_check_slice_enabled(self):
        self.assertRaises(RuntimeError,
                          self.masking_vector_wrapper._check_slice_enabled,
                          random.randint(0, 1000),
                          random.randint(0, 1000))

    # test that a ValueError is raised if the masking vector has not the
    # expected size
    def test_check_masking_vector(self):
        self.assertRaises(ValueError,
                          self.masking_vector_wrapper.set_masking_vector,
                          numpy.zeros(self.masking_vector_size - 1))
