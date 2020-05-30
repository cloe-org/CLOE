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
from likelihood.cobaya_interface import loglike
from ..photometric_survey import shear
from scipy.interpolate import UnivariateSpline
from astropy import constants as const


class cosmoinitTestCase(TestCase):

    def setUp(self):
        # SJ: For now, example sampling in redshift (z)
        z_min = 0.0
        z_max = 2.5
        z_samp = 10
        self.z_win = np.linspace(z_min, z_max, z_samp)
        info = {
            'params': {
                'ombh2': 0.022, 'omch2': 0.12, 'H0': 68.0, 'tau': 0.07,
                'mnu': 0.06, 'nnu': 3.046, 'num_massive_neutrinos': 1,
                'ns': 0.9674, 'like_selection': 12, 'As': 2.1e-9},
            'theory': {'camb': {'stop_at_error': True}},
            'likelihood': {'euclid': loglike},
        }
        model = get_model(info)
        model.logposterior({})
        self.cosmology = Cosmology()
        self.cosmology.cosmo_dic['H0'] = \
            model.parameterization.constant_params()['H0']
        self.cosmology.cosmo_dic['omch2'] =  \
            model.parameterization.constant_params()['omch2']
        self.cosmology.cosmo_dic['ombh2'] = \
            model.parameterization.constant_params()['ombh2']
        self.cosmology.cosmo_dic['mnu'] = \
            model.parameterization.constant_params()['mnu']
        self.cosmology.cosmo_dic['comov_dist'] = \
            model.likelihood.theory.get_comoving_radial_distance(
            self.z_win)
        self.cosmology.cosmo_dic['H'] = UnivariateSpline(
            self.z_win, model.likelihood.theory.get_H(self.z_win))
        self.cosmology.cosmo_dic['Pk_interpolator'] = \
            model.likelihood.theory.get_Pk_interpolator()
        self.cosmology.cosmo_dic['Pk_delta'] = \
            (self.cosmology.cosmo_dic['Pk_interpolator']
                ['delta_tot_delta_tot'])
        self.cosmology.cosmo_dic['fsigma8'] = \
            model.likelihood.theory.get_fsigma8(self.z_win)
        # (GCH): required by Anurag
        self.cosmology.cosmo_dic['c'] = self.c = 3.0e5
        self.rfn = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 4.6, 20), np.linspace(0.0, 4.6, 20), ext=2)
        self.flatnz = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 4.6, 20), np.ones(20), ext=2)
        self.cosmology.cosmo_dic['r_z_func'] = self.rfn
        self.integrand_check = -1.0
        self.wbincheck = 9.143694637057992e-09
        self.H0 = 67.0
        self.c = const.c.to('km/s').value
        self.omch2 = 0.12
        self.ombh2 = 0.022
        # (GCH): import Shear
        self.shear = shear.Shear(self.cosmology.cosmo_dic)
        self.W_i_Gcheck = 0.002724124626761272
        self.phot_galbias_check = 1.09544512

    def tearDown(self):
        self.W_i_Gcheck = None
        self.phot_galbias_check = None
        self.integrand_check = None
        self.wbincheck = None

    def test_GC_window(self):
        npt.assert_almost_equal(self.shear.GC_window(0.2, 0.001, 1),
                                self.W_i_Gcheck,
                                err_msg='GC_window failed')

    def test_phot_galbias(self):
        npt.assert_almost_equal(self.shear.phot_galbias(0.1, 0.3),
                                self.phot_galbias_check,
                                err_msg='Photometric galaxy bias failed')

    def test_w_integrand(self):
        int_comp = self.shear.WL_window_integrand(0.1, 0.2, self.flatnz)
        npt.assert_almost_equal(int_comp, self.integrand_check,
                                err_msg='Integrand of WL kernel failed')

    def test_WL_window(self):
        int_comp = self.shear.WL_window(0.1, 1)
        npt.assert_almost_equal(int_comp, self.wbincheck,
                                err_msg='WL_window failed')

    def test_rzfunc_exception(self):
        npt.assert_raises(Exception, shear.Shear,
                          {'H0': self.H0,
                           'c': self.c,
                           'omch2': self.omch2,
                           'ombh2': self.ombh2})
