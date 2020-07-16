"""UNIT TESTS FOR COSMO

This module contains unit tests for the cosmo module.
=======

"""

# (GCH): Use Cobaya Model wrapper

from cobaya.model import get_model
from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import integrate
from ..cosmo.cosmology import Cosmology
from likelihood.cobaya_interface import EuclidLikelihood
from likelihood.tests.test_wrapper import CobayaModel
from scipy.interpolate import InterpolatedUnivariateSpline


class cosmoinitTestCase(TestCase):

    def setUp(self):
        # (GCH): define cosmology values in Cosmology dict
        cosmo = Cosmology()
        cosmo.cosmo_dic['ombh2'] = 0.022
        cosmo.cosmo_dic['omch2'] = 0.12
        cosmo.cosmo_dic['H0'] = 68.0
        cosmo.cosmo_dic['tau'] = 0.07
        cosmo.cosmo_dic['mnu'] = 0.06
        cosmo.cosmo_dic['nnu'] = 3.046
        cosmo.cosmo_dic['ns'] = 0.9674
        cosmo.cosmo_dic['As'] = 2.1e-9

        # (GCH): create wrapper model
        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()
        # (GCH): Check values
        self.H0check = 68.0
        self.Dcheck = 1.0
        self.fcheck = 0.518508
        self.Hcheck = 75.251876

    def tearDown(self):
        self.H0check = None

    def test_cosmo_init(self):
        emptflag = bool(self.model_test.cosmology.cosmo_dic)
        npt.assert_equal(emptflag, True,
                         err_msg='Cosmology dictionary not initialised '
                                 'correctly.')

    def test_cosmo_asign(self):
        npt.assert_allclose(self.model_test.cosmology.cosmo_dic['H0'],
                            self.H0check,
                            err_msg='Cosmology dictionary assignment '
                                    'failed')
        npt.assert_allclose(
            self.model_test.cosmology.cosmo_dic['H_z_func'](0.2),
            self.Hcheck,
            err_msg='Cosmology dictionary assignment '
            'failed')

    def test_cosmo_growth_factor(self):
        D = self.model_test.cosmology.growth_factor(0.0, 0.002)
        npt.assert_equal(D, self.Dcheck,
                         err_msg='Error in D_z_k calculation ')

    def test_cosmo_growth_rate(self):
        f = self.model_test.cosmology.growth_rate(
            self.model_test.cosmology.cosmo_dic['z_win'], 0.002)
        npt.assert_allclose(f(0), self.fcheck,
                            rtol=1e-3,
                            err_msg='Error in f_z_k calculation ')

    def test_update_cosmo_dic(self):
        self.model_test.cosmology.update_cosmo_dic(
            self.model_test.cosmology.cosmo_dic['z_win'], 0.002)
        if 'D_z_k' in self.model_test.cosmology.cosmo_dic:
            emptflag_D = True
        if 'f_z_k' in self.model_test.cosmology.cosmo_dic:
            emptflag_f = True
        npt.assert_equal(emptflag_D, True,
                         err_msg='D_z_k not calculated ')
        npt.assert_equal(emptflag_f, True,
                         err_msg='f_z_k not calculated ')
