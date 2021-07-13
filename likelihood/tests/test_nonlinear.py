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

    def setUp(self):
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
        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()
        # Check values
        self.Pgg_phot_test = 58392.759202
        self.Pgdelta_phot_test = 41294.412017
        self.Pgg_spec_test = 82548.320427
        self.Pgdelta_spec_test = 59890.445816
        self.Pii_test = np.array([2.417099, 0.902693, 0.321648])
        self.Pdeltai_test = -265.680094
        self.Pgi_phot_test = -375.687484
        self.Pgi_spec_test = -388.28625

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
        self.Pgg_spec_test = None
        self.Pgdelta_spec_test = None
        self.Pii_test = None
        self.Pdeltai_test = None
        self.Pgi_phot_test = None
        self.Pgi_spec_test = None

    def test_Pgg_phot_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pgg_phot_def(self.z1,
                                                                  self.k1)
        npt.assert_allclose(test_p, self.Pgg_phot_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by Pgg_phot_def')

    def test_Pgdelta_phot_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pgdelta_phot_def(self.z1,
                                                                      self.k1)
        npt.assert_allclose(
            test_p,
            self.Pgdelta_phot_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def')

    def test_Pgg_spec_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pgg_spec_def(self.z1,
                                                                  self.k1,
                                                                  self.mu)
        npt.assert_allclose(test_p, self.Pgg_spec_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by Pgg_spec_def')

    def test_Pgdelta_spec_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pgdelta_spec_def(self.z1,
                                                                      self.k1,
                                                                      self.mu)
        npt.assert_allclose(
            test_p,
            self.Pgdelta_spec_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_spec_def')

    def test_Pii_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pii_def(self.z1,
                                                             [self.k1,
                                                              self.k2,
                                                              self.k3])
        type_check = isinstance(test_p, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def'

        assert test_p.size == self.arrsize, '''Error in size of array returned
                                             by Pii_def'''

        npt.assert_allclose(test_p, self.Pii_test,
                            rtol=self.rtol,
                            err_msg='Error in values returned by Pii_def')

    def test_Pdeltai_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pdeltai_def(self.z1,
                                                                 self.k1)
        npt.assert_allclose(test_p, self.Pdeltai_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by Pdeltai_def')

    def test_Pgi_phot_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pgi_phot_def(self.z1,
                                                                  self.k1)
        npt.assert_allclose(test_p, self.Pgi_phot_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by Pgi_phot_def')

    def test_Pgi_spec_def(self):
        test_p = self.model_test.cosmology.nonlinear.Pgi_spec_def(self.z1,
                                                                  self.k1)
        npt.assert_allclose(test_p, self.Pgi_spec_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by Pgi_spec_def')

    def test_fia(self):
        fia = self.model_test.cosmology.nonlinear.misc.fia(self.z1, self.k1)
        npt.assert_allclose(fia, self.fia_test,
                            rtol=self.rtol,
                            err_msg='Error in value returned by fia')

    def test_istf_spec_galbias(self):
        b = self.model_test.cosmology.nonlinear.misc.istf_spec_galbias(self.z1)
        npt.assert_allclose(b, self.bspec,
                            rtol=self.rtol,
                            err_msg='Error in istf_spec_galbias')
        self.assertRaises(ValueError,
                          (self.model_test.cosmology
                           .nonlinear.misc
                           .istf_spec_galbias),
                          self.z2)
        self.assertRaises(ValueError,
                          (self.model_test.cosmology
                           .nonlinear.misc
                           .istf_spec_galbias),
                          self.z3)

    def test_istf_phot_galbias(self):
        b = self.model_test.cosmology.nonlinear.misc.istf_phot_galbias(self.z1)
        npt.assert_allclose(b, self.bphot,
                            rtol=self.rtol,
                            err_msg='Error in istf_phot_galbias')
