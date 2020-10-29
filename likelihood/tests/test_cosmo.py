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
        self.fcheck = 0.516266
        self.Hcheck = 75.234214
        self.bias_gc_phot_check = 1.220245876862528
        self.bias_gc_spec_check = 1.46
        self.Pgg_phot_test = 57627.032026
        self.Pgd_phot_test = 41167.417014
        self.Pgg_spec_test = 83066.735675
        self.Pgd_spec_test = 42937.21192

    def tearDown(self):
        self.H0check = None
        self.Dcheck = None
        self.fcheck = None
        self.Hcheck = None
        self.bias_gc_phot_check = None
        self.bias_gc_spec_check = None
        self.Pgg_phot_test = None
        self.Pgd_phot_test = None
        self.Pgg_spec_test = None
        self.Pgd_spec_test = None

    def test_cosmo_init(self):
        emptflag = bool(self.model_test.cosmology.cosmo_dic)
        npt.assert_equal(emptflag, True,
                         err_msg='Cosmology dictionary not initialised '
                                 'correctly.')

    def test_cosmo_nuisance_init(self):
        emptflag = bool(
            self.model_test.cosmology.cosmo_dic['nuisance_parameters'])
        npt.assert_equal(emptflag, True,
                         err_msg='nuisance parameters '
                         'not initialized within '
                         'cosmology dictionary')

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

    def test_phot_bias(self):
        val = self.model_test.cosmology.istf_phot_galbias(0.43)
        npt.assert_allclose(val, self.bias_gc_phot_check,
                            rtol=1e-3,
                            err_msg='Error in GC-phot bias calculation')

    def test_spec_bias(self):
        val = self.model_test.cosmology.istf_spec_galbias(1.0)
        npt.assert_allclose(val, self.bias_gc_spec_check,
                            rtol=1e-3,
                            err_msg='Error in GC-spec bias calculation')

    def test_Pgg_phot(self):
        test_p = self.model_test.cosmology.Pgg_phot_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgg_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgg calculation')

    def test_Pg_delta_phot(self):
        test_p = self.model_test.cosmology.Pgd_phot_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgd_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgdelta calculation')

    def test_Pgg_spec(self):
        test_p = self.model_test.cosmology.Pgg_spec_def(1.0, 0.01, 0.5)
        npt.assert_allclose(test_p, self.Pgg_spec_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spec Pgg calculation')

    def test_Pg_delta_spec(self):
        test_p = self.model_test.cosmology.Pgd_spec_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgd_spec_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spec Pgdelta calculation')

    def test_Pgg_phot_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgg_phot'](1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgg_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgg interpolation')

    def test_Pg_delta_phot_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgdelta_phot'](1.0,
                                                                     0.01)
        npt.assert_allclose(test_p, self.Pgd_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgdelta interpolation')

    def test_Pgg_spec_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgg_spec'](1.0,
                                                                 0.01,
                                                                 0.5)
        npt.assert_allclose(test_p, self.Pgg_spec_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spec Pgg interpolation')

    def test_Pg_delta_spec_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgdelta_spec'](1.0,
                                                                     0.01)
        npt.assert_allclose(test_p, self.Pgd_spec_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spec Pgdelta interpolation')
