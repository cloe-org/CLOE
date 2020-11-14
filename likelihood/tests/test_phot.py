"""UNIT TESTS FOR PHOTO

This module contains unit tests for the Photo sub-module of the
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
from ..photometric_survey import photo
from ..data_reader import reader
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

        test_reading = reader.Reader()
        test_reading.compute_nz()
        nz_dic_WL = test_reading.nz_dict_WL
        nz_dic_GC = test_reading.nz_dict_GC_Phot
        # (GCH): create wrapper model
        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()
        self.integrand_check = -0.950651
        self.wbincheck = 7.200027e-06
        self.H0 = 67.0
        self.c = const.c.to('km/s').value
        self.omch2 = 0.12
        self.ombh2 = 0.022
        self.phot = photo.Photo(self.model_test.cosmology.cosmo_dic,
                                nz_dic_WL, nz_dic_GC)
        self.W_i_Gcheck = 5.319691e-09
        self.W_IA_check = 0.000106330045837
        self.cl_integrand_check = 0.000718
        self.cl_WL_check = 8.517887e-09
        self.cl_GC_check = 2.95197263e-05
        self.cl_cross_check = 3.14508861e-07
        self.flatnz = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 4.6, 20), np.ones(20), ext=2)

    def tearDown(self):
        self.integrand_check = None
        self.wbincheck = None
        self.W_i_Gcheck = None
        self.W_IA_check = None
        self.cl_integrand_check = None
        self.cl_WL_check = None
        self.cl_GC_check = None
        self.cl_cross_check = None
        self.flatnz = None

    def test_GC_window(self):
        npt.assert_allclose(self.phot.GC_window(0.001, 1),
                            self.W_i_Gcheck, rtol=1e-05,
                            err_msg='GC_window failed')

    def test_IA_window(self):
        npt.assert_allclose(self.phot.IA_window(0.1, 1),
                            self.W_IA_check, rtol=1e-05,
                            err_msg='IA_window failed')

    def test_w_integrand(self):
        int_comp = self.phot.WL_window_integrand(0.1, 0.2, self.flatnz)
        npt.assert_allclose(int_comp, self.integrand_check, rtol=1e-05,
                            err_msg='Integrand of WL kernel failed')

    def test_WL_window(self):
        int_comp = self.phot.WL_window(0.1, 1, 0.1)
        npt.assert_allclose(int_comp, self.wbincheck, rtol=1e-05,
                            err_msg='WL_window failed')

    def test_rzfunc_exception(self):
        npt.assert_raises(Exception, self.phot,
                          {'H0': self.H0,
                           'c': self.c,
                           'omch2': self.omch2,
                           'ombh2': self.ombh2})

    def test_power_exception(self):
        npt.assert_raises(Exception, self.phot.Cl_generic_integrand,
                          10.0, 1.0, 2.0, 12.0,
                          self.model_test.cosmology.cosmo_dic['Pgg_phot'])

    def test_cl_WL(self):
        cl_int = self.phot.Cl_WL(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_WL_check, rtol=1e-05,
                            err_msg='Cl WL test failed')

    def test_cl_GC(self):
        cl_int = self.phot.Cl_GC_phot(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_GC_check, rtol=1e-02,
                            err_msg='Cl GC photometric test failed')

    def test_cl_cross(self):
        cl_int = self.phot.Cl_cross(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_cross_check, rtol=1e-02,
                            err_msg='Cl photometric cross test failed')
