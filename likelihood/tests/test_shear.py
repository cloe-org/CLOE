"""UNIT TESTS FOR SHEAR

This module contains unit tests for the Shear sub-module of the
photometric survey module.
=======

"""

# (GCH): Use Cobaya Model wrapper

from cobaya.model import get_model
from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import integrate
from scipy import interpolate
from ..cosmo.cosmology import Cosmology
from likelihood.cobaya_interface import EuclidLikelihood
from ..photometric_survey import shear
from astropy import constants as const
from likelihood.tests.test_wrapper import CobayaModel


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
        self.rfn = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 4.6, 20), np.linspace(0.0, 4.6, 20), ext=2)

        self.model_test.cosmology.cosmo_dic['r_z_func'] = self.rfn
        self.integrand_check = -1.0
        self.wbincheck = 1.715463e-09
        self.H0 = 67.0
        self.c = const.c.to('km/s').value
        self.omch2 = 0.12
        self.ombh2 = 0.022
        self.shear = shear.Shear(self.model_test.cosmology.cosmo_dic)
        self.W_i_Gcheck = 0.0027291100226392064
        self.phot_galbias_check = 1.09544512
        self.cl_integrand_check = 8.953918
        self.cl_WL_check = 7.933128e-10
        self.cl_GC_check = 4.246989e+12
        self.cl_cross_check = 27.401784
        self.flatnz = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 4.6, 20), np.ones(20), ext=2)

    def tearDown(self):
        self.W_i_Gcheck = None
        self.phot_galbias_check = None
        self.integrand_check = None
        self.wbincheck = None
        self.cl_integrand_check = None

    def test_GC_window(self):
        npt.assert_allclose(self.shear.GC_window(0.2, 0.001, 1),
                            self.W_i_Gcheck, rtol=1e-05,
                            err_msg='GC_window failed')

    def test_phot_galbias(self):
        npt.assert_allclose(self.shear.phot_galbias(0.1, 0.3),
                            self.phot_galbias_check,
                            err_msg='Photometric galaxy bias failed')

    def test_w_integrand(self):
        int_comp = self.shear.WL_window_integrand(0.1, 0.2, self.flatnz)
        npt.assert_allclose(int_comp, self.integrand_check, rtol=1e-05,
                            err_msg='Integrand of WL kernel failed')

    def test_WL_window(self):
        int_comp = self.shear.WL_window(0.1, 1)
        npt.assert_allclose(int_comp, self.wbincheck, rtol=1e-05,
                            err_msg='WL_window failed')

    def test_rzfunc_exception(self):
        npt.assert_raises(Exception, self.shear,
                          {'H0': self.H0,
                           'c': self.c,
                           'omch2': self.omch2,
                           'ombh2': self.ombh2})

    def test_cl_integrand(self):
        cl_int = self.shear.Cl_generic_integrand(1.0, 1.0, 1.0, 100.0)
        npt.assert_allclose(cl_int, self.cl_integrand_check, rtol=1e-03,
                            err_msg='Cl integrand failed')

    def test_cl_WL(self):
        cl_int = self.shear.Cl_WL(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_WL_check, rtol=1e-05,
                            err_msg='Cl WL test failed')

    def test_cl_GC(self):
        cl_int = self.shear.Cl_GC_phot(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_GC_check, rtol=1e-05,
                            err_msg='Cl GC photometric test failed')

    def test_cl_cross(self):
        cl_int = self.shear.Cl_cross(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_cross_check, rtol=1e-05,
                            err_msg='Cl photometric cross test failed')
