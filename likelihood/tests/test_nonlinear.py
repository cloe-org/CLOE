"""UNIT TESTS FOR NON-LINEAR

This module contains unit tests for the non-linear module.
=======

"""


from unittest import TestCase
import numpy as np
import numpy.testing as npt
from astropy import constants as const
from likelihood.cosmo.cosmology import Cosmology
from likelihood.tests.test_wrapper import CobayaModel


class nonlinearinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Define cosmology values in Cosmology dict
        cosmo = Cosmology()
        cosmo.cosmo_dic['ombh2'] = 0.022
        cosmo.cosmo_dic['omch2'] = 0.12
        cosmo.cosmo_dic['H0'] = 67.0
        cosmo.cosmo_dic['tau'] = 0.07
        cosmo.cosmo_dic['mnu'] = 0.06
        cosmo.cosmo_dic['nnu'] = 3.046
        cosmo.cosmo_dic['ns'] = 0.9674
        cosmo.cosmo_dic['As'] = 2.1e-9
        cosmo.cosmo_dic['H0_Mpc'] = \
            cosmo.cosmo_dic['H0'] / const.c.to('km/s').value,
        # Create wrapper model
        cls.model_test = CobayaModel(cosmo)
        cls.model_test.update_cosmo()
        cls.nl = cls.model_test.cosmology.nonlinear

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
        test_p = self.nl.Pgg_phot_def(self.z1, self.k1)
        npt.assert_allclose(test_p, self.Pgg_phot_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by Pgg_phot_def')

    def test_Pgdelta_phot_def(self):
        test_p = self.nl.Pgdelta_phot_def(self.z1, self.k1)
        npt.assert_allclose(
            test_p,
            self.Pgdelta_phot_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def')

    def test_Pgg_spectro_def(self):
        test_p = self.nl.Pgg_spectro_def(self.z1, self.k1, self.mu)
        npt.assert_allclose(test_p, self.Pgg_spectro_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned '
                                    'by Pgg_spectro_def')

    def test_Pgdelta_spectro_def(self):
        test_p = self.nl.Pgdelta_spectro_def(self.z1, self.k1, self.mu)
        npt.assert_allclose(
            test_p, self.Pgdelta_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_spectro_def')

    def test_Pii_def(self):
        test_p = self.nl.Pii_def(self.z1, [self.k1, self.k2, self.k3])
        type_check = isinstance(test_p, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def'

        assert test_p.size == self.arrsize, 'Error in size of array ' \
                                            'returned by Pii_def'

        npt.assert_allclose(test_p, self.Pii_test,
                            rtol=self.rtol,
                            err_msg='Error in values returned '
                                    'by Pii_def')

    def test_Pdeltai_def(self):
        test_p = self.nl.Pdeltai_def(self.z1, self.k1)
        npt.assert_allclose(test_p, self.Pdeltai_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned '
                                    'by Pdeltai_def')

    def test_Pgi_phot_def(self):
        test_p = self.nl.Pgi_phot_def(self.z1, self.k1)
        npt.assert_allclose(test_p, self.Pgi_phot_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned '
                                    'by Pgi_phot_def')

    def test_Pgi_spectro_def(self):
        test_p = self.nl.Pgi_spectro_def(self.z1, self.k1)
        npt.assert_allclose(test_p, self.Pgi_spectro_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned '
                                    'by Pgi_spectro_def')

    def test_fia(self):
        fia = self.nl.misc.fia(self.z1, self.k1)
        npt.assert_allclose(fia, self.fia_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by fia')

    def test_fia_k_array(self):
        k_array = np.array([self.k1, self.k2, self.k3])
        fia = self.nl.misc.fia(self.z1, k_array)
        type_check = isinstance(fia, np.ndarray)
        npt.assert_equal(type_check, True,
                         err_msg='Error in type returned by fia '
                                 'when z is scalar and k is array')
        size_check = np.size(fia)
        npt.assert_equal(size_check, np.size(k_array),
                         err_msg='Error in size returned by fia '
                         'when z is scalar and k is array')

    def test_fia_z_array(self):
        z_array = np.array([self.z1, self.z1])
        fia = self.nl.misc.fia(z_array, self.k1)
        type_check = isinstance(fia, np.ndarray)
        npt.assert_equal(type_check, True,
                         err_msg='Error in type returned by fia '
                         'when z is array and k is scalar')
        size_check = np.size(fia)
        npt.assert_equal(size_check, np.size(z_array),
                         err_msg='Error in size returned by fia '
                         'when z is array and k is scalar')

    def test_fia_zk_array(self):
        z_array = np.array([self.z1, self.z1])
        k_array = np.array([self.k1, self.k2, self.k3])
        fia = self.nl.misc.fia(z_array, k_array)
        type_check = isinstance(fia, np.ndarray)
        npt.assert_equal(type_check, True,
                         err_msg='Error in type returned by fia '
                                 'when z is array and k is array')
        size_check = np.size(fia)
        npt.assert_equal(size_check, np.size(z_array) * np.size(k_array),
                         err_msg='Error in size returned by fia '
                                 'when z is array and k is array')
        shape_check = np.shape(fia)
        npt.assert_array_equal(shape_check, (2, 3),
                               err_msg='Error in size returned by fia '
                                       'when z is array and k is array')

    def test_istf_spectro_galbias(self):
        b = self.nl.misc.istf_spectro_galbias(self.z1)
        npt.assert_allclose(b, self.bspec,
                            rtol=self.rtol,
                            err_msg='Error in istf_spectro_galbias')
        self.assertRaises(ValueError,
                          (self.model_test.cosmology
                           .nonlinear.misc
                           .istf_spectro_galbias),
                          self.z2)
        self.assertRaises(ValueError,
                          (self.model_test.cosmology
                           .nonlinear.misc
                           .istf_spectro_galbias),
                          self.z3)

    def test_istf_phot_galbias(self):
        b = self.nl.misc.istf_phot_galbias(self.z1)
        npt.assert_allclose(b, self.bphot,
                            rtol=self.rtol,
                            err_msg='Error in istf_phot_galbias')
