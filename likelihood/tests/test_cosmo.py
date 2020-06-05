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
from likelihood.cobaya_interface import loglike
from scipy.interpolate import UnivariateSpline


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
        # (GCH): checks
        self.H0check = 68.0
        self.Dcheck = 1.0
        self.fcheck = 0.135876
        self.Hcheck = 75.36382368

    def tearDown(self):
        self.H0check = None

    def test_cosmo_init(self):
        emptflag = bool(self.cosmology.cosmo_dic)
        npt.assert_equal(emptflag, True,
                         err_msg='Cosmology dictionary not initialised '
                                 'correctly.')

    def test_cosmo_asign(self):
        self.cosmology.cosmo_dic['H0'] = self.H0check
        npt.assert_allclose(self.cosmology.cosmo_dic['H0'],
                            self.H0check,
                            err_msg='Cosmology dictionary assignment '
                                    'failed')
        npt.assert_allclose(self.cosmology.cosmo_dic['H'](0.2),
                            self.Hcheck,
                            err_msg='Cosmology dictionary assignment '
                                    'failed')

    def test_cosmo_growth_factor(self):
        D = self.cosmology.growth_factor(0.0, 0.002)
        npt.assert_equal(D, self.Dcheck,
                         err_msg='Error in D_z_k calculation ')

    def test_cosmo_growth_rate(self):
        f = self.cosmology.growth_rate(self.z_win, 0.002)
        npt.assert_allclose(f[0], self.fcheck,
                            rtol=1e-3,
                            err_msg='Error in f_z_k calculation ')

    def test_update_cosmo_dic(self):
        self.cosmology.update_cosmo_dic(self.z_win, 0.002)
        if 'D_z_k' in self.cosmology.cosmo_dic:
            emptflag_D = True
        if 'f_z_k' in self.cosmology.cosmo_dic:
            emptflag_f = True
        npt.assert_equal(emptflag_D, True,
                         err_msg='D_z_k not calculated ')
        npt.assert_equal(emptflag_f, True,
                         err_msg='f_z_k not calculated ')

    def test_cosmo_comov_dist_interp(self):
        self.cosmology.cosmo_dic['z_win'] = self.z_win
        self.cosmology.interp_comoving_dist()
        npt.assert_allclose(self.cosmology.cosmo_dic['comov_dist'][1],
                            self.cosmology.cosmo_dic['r_z_func']
                            (self.z_win[1]),
                            err_msg='Interpolation of comoving distance '
                                    'failed.')

    def test_zwin_exception(self):
        self.cosmology.cosmo_dic['z_win'] = None
        npt.assert_raises(Exception, self.cosmology.interp_comoving_dist)
