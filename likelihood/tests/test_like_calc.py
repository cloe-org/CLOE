"""UNIT TESTS FOR LIKE_CALC

This module contains unit tests for the likelihood calculation module.
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from ..like_calc import euclike
from ..cosmo.cosmology import Cosmology
from likelihood.tests.test_wrapper import CobayaModel
from likelihood.cobaya_interface import EuclidLikelihood


class likecalcTestCase(TestCase):

    def setUp(self):
        cosmo = Cosmology()
        cosmo.cosmo_dic['ombh2'] = 0.022
        cosmo.cosmo_dic['omch2'] = 0.12
        cosmo.cosmo_dic['H0'] = 68.0
        cosmo.cosmo_dic['tau'] = 0.07
        cosmo.cosmo_dic['mnu'] = 0.06
        cosmo.cosmo_dic['nnu'] = 3.046
        cosmo.cosmo_dic['ns'] = 0.9674
        cosmo.cosmo_dic['As'] = 2.1e-9

        self.info = {'params': {
            'ombh2': cosmo.cosmo_dic['ombh2'],
            'omch2': cosmo.cosmo_dic['omch2'],
            'omnuh2': cosmo.cosmo_dic['omnuh2'],
            'H0': cosmo.cosmo_dic['H0'],
            'tau': cosmo.cosmo_dic['tau'],
            'mnu': cosmo.cosmo_dic['mnu'],
            'nnu': cosmo.cosmo_dic['nnu'],
            'ns': cosmo.cosmo_dic['ns'],
            'As': cosmo.cosmo_dic['As'],
            'like_selection': 12},
            'theory': {'camb':
                       {'stop_at_error': True,
                        'extra_args': {'num_massive_neutrinos': 1}}},
            # Likelihood: we load the likelihood as an external function
            'likelihood': {'euclid': EuclidLikelihood}}

        # fiducial params
        cosmo_fiducial = Cosmology()
        self.model_fiducial = CobayaModel(cosmo_fiducial)
        self.model_fiducial.update_cosmo()
        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()
        self.like_tt = euclike.Euclike()

        # (SJ): For now use loglike below, to be updated
        # (SJ): First one without, second one with h and (2pi/h)^3 corrections
        # self.check_loglike = 4.607437e+11
        self.check_loglike = 30568.400834

    def tearDown(self):
        self.check_loglike = None

    def test_loglike(self):
        npt.assert_allclose(self.like_tt.loglike(
            self.model_test.cosmology.cosmo_dic,
            self.model_fiducial.cosmology.cosmo_dic, self.info),
                self.check_loglike, rtol=1e-06, err_msg='Loglike failed')
