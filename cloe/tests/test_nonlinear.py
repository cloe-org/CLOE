"""UNIT TESTS FOR NON-LINEAR

This module contains unit tests for the non-linear module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.non_linear.nonlinear import Nonlinear
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class nonlinearinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # Load cosmology dictionary for tests
        cosmo_dic = load_test_pickle('cosmo_test_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl = Nonlinear(cosmo_dic)

    def setUp(self) -> None:
        # Check values
        self.Pgg_phot_test = 58392.759202
        self.Pgdelta_phot_test = 41294.412017
        self.Pgg_spectro_test = 82548.320427
        self.Pgdelta_spectro_test = 59890.445816
        self.Pii_test = np.array([2.417099, 0.902693, 0.321648])
        self.Pdeltai_test = -265.680094
        self.Pgi_phot_test = -375.687484
        self.Pgi_spectro_test = -388.28625

        self.rtol = 1e-3
        self.z1 = 1.0
        self.z2 = 0.5
        self.z3 = 2.0
        self.k1 = 0.01
        self.k2 = 0.05
        self.k3 = 0.1
        self.mu = 0.5
        self.arrsize = 3

        self.D_test = 0.6061337447181263
        self.fia_test = -0.00909382071134854
        self.bspec = 1.46148
        self.bphot = 1.41406

    def tearDown(self):
        self.Pgg_phot_test = None
        self.Pgdelta_phot_test = None
        self.Pgg_spectro_test = None
        self.Pgdelta_spectro_test = None
        self.Pii_test = None
        self.Pdeltai_test = None
        self.Pgi_phot_test = None
        self.Pgi_spectro_test = None

    def test_Pgg_phot_def(self):
        npt.assert_allclose(
            self.nl.Pgg_phot_def(self.z1, self.k1),
            self.Pgg_phot_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def',
        )

    def test_Pgdelta_phot_def(self):
        npt.assert_allclose(
            self.nl.Pgdelta_phot_def(self.z1, self.k1),
            self.Pgdelta_phot_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def',
        )

    def test_Pgg_spectro_def(self):
        npt.assert_allclose(
            self.nl.Pgg_spectro_def(self.z1, self.k1, self.mu),
            self.Pgg_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_spectro_def',
        )

    def test_Pgdelta_spectro_def(self):
        npt.assert_allclose(
            self.nl.Pgdelta_spectro_def(self.z1, self.k1, self.mu),
            self.Pgdelta_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_spectro_def',
        )

    def test_Pii_def(self):
        test_p = self.nl.Pii_def(self.z1, [self.k1, self.k2, self.k3])
        type_check = isinstance(test_p, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def'
        assert test_p.size == self.arrsize, (
            'Error in size of array returned by Pii_def'
        )

        npt.assert_allclose(
            test_p,
            self.Pii_test,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def',
        )

    def test_Pdeltai_def(self):
        npt.assert_allclose(
            self.nl.Pdeltai_def(self.z1, self.k1),
            self.Pdeltai_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def',
        )

    def test_Pgi_phot_def(self):
        npt.assert_allclose(
            self.nl.Pgi_phot_def(self.z1, self.k1),
            self.Pgi_phot_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def',
        )

    def test_Pgi_spectro_def(self):
        npt.assert_allclose(
            self.nl.Pgi_spectro_def(self.z1, self.k1),
            self.Pgi_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_spectro_def',
        )

    def test_fia(self):
        npt.assert_allclose(
            self.nl.misc.fia(self.z1, self.k1),
            self.fia_test,
            rtol=self.rtol,
            err_msg='Error in value returned by fia',
        )

    def test_fia_k_array(self):
        k_array = np.array([self.k1, self.k2, self.k3])
        fia = self.nl.misc.fia(self.z1, k_array)
        npt.assert_equal(
            isinstance(fia, np.ndarray),
            True,
            err_msg=(
                'Error in type returned by fia when z is scalar and k is ' +
                'array'
            ),
        )
        npt.assert_equal(
            np.size(fia),
            np.size(k_array),
            err_msg=(
                'Error in size returned by fia when z is scalar and k ' +
                'is array'
            ),
        )

    def test_fia_z_array(self):
        z_array = np.array([self.z1, self.z1])
        fia = self.nl.misc.fia(z_array, self.k1)
        npt.assert_equal(
            isinstance(fia, np.ndarray),
            True,
            err_msg=(
                'Error in type returned by fia when z is array and k is ' +
                'scalar'
            ),
        )
        npt.assert_equal(
            np.size(fia),
            np.size(z_array),
            err_msg=(
                'Error in size returned by fia when z is array and k is ' +
                ' scalar'
            ),
        )

    def test_fia_zk_array(self):
        z_array = np.array([self.z1, self.z1])
        k_array = np.array([self.k1, self.k2, self.k3])
        fia = self.nl.misc.fia(z_array, k_array)
        npt.assert_equal(
            isinstance(fia, np.ndarray),
            True,
            err_msg=(
                'Error in type returned by fia when z is array and k is array'
            ),
        )
        npt.assert_equal(
            np.size(fia),
            np.size(z_array) * np.size(k_array),
            err_msg=(
                'Error in size returned by fia when z is array and k is array'
            ),
        )
        npt.assert_array_equal(
            np.shape(fia),
            (2, 3),
            err_msg=(
                'Error in size returned by fia when z is array and k is array'
            ),
        )

    def test_istf_spectro_galbias(self):
        npt.assert_allclose(
            self.nl.misc.istf_spectro_galbias(self.z1),
            self.bspec,
            rtol=self.rtol,
            err_msg='Error in istf_spectro_galbias',
        )
        self.assertRaises(
            ValueError,
            self.nl.misc.istf_spectro_galbias,
            self.z2,
        )
        self.assertRaises(
            ValueError,
            self.nl.misc.istf_spectro_galbias,
            self.z3,
        )

    def test_istf_phot_galbias(self):
        npt.assert_allclose(
            self.nl.misc.istf_phot_galbias(self.z1),
            self.bphot,
            rtol=self.rtol,
            err_msg='Error in istf_phot_galbias',
        )
