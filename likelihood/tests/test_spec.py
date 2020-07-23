"""UNIT TESTS FOR SHEAR

This module contains unit tests for the Spec sub-module of the
spectroscopy survey module.
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
from ..spectroscopic_survey.spec import Spec
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

        # (GCH): define fiducial
        cosmo_fiducial = Cosmology()
        self.model_fiducial = CobayaModel(cosmo_fiducial)
        self.model_fiducial.update_cosmo()

        # (GCH): create wrapper model
        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()

        self.spec = Spec(self.model_test.cosmology.cosmo_dic,
                         self.model_fiducial.cosmology.cosmo_dic)

        # (GCH): Checks
        self.check_multipole_spectra_noap = [6758.118715,
                                             6290.567416,
                                             691.217494]
        self.check_multipole_spectra_m0 = 6710.162801
        self.check_multipole_spectra_m1 = 0.0
        self.check_multipole_spectra_m2 = 6231.481902
        self.check_multipole_spectra_m3 = 0.0
        self.check_multipole_spectra_m4 = 714.627534
        self.check_multipole_spectra_integrand = 1848.17102
        self.check_pkgal_linear = 7885.192895
        self.check_scaling_factor_perp = 0.993
        self.check_scaling_factor_parall = 0.992782
        self.check_get_k = 0.001007
        self.check_get_mu = 1.00

    def tearDown(self):
        self.check_scaling_factor_perp = None
        self.check_scaling_factor_parall = None
        self.check_get_k = None
        self.check_get_mu = None

    def test_multipole_spectra_noap(self):
        npt.assert_allclose(self.spec.multipole_spectra_noap(1.0, 0.1),
                            self.check_multipole_spectra_noap,
                            rtol=1e-06,
                            err_msg='Multipole spectra (no AP) failed')

    def test_multipole_spectra_m0(self):
        npt.assert_allclose(self.spec.multipole_spectra(1.0, 0.1, 0),
                            self.check_multipole_spectra_m0,
                            rtol=1e-06,
                            err_msg='Multipole spectrum m = 0 failed')

    def test_multipole_spectra_m1(self):
        npt.assert_allclose(self.spec.multipole_spectra(1.0, 0.1, 1),
                            self.check_multipole_spectra_m1,
                            atol=1e-06,
                            err_msg='Multipole spectrum m = 1 failed')

    def test_multipole_spectra_m2(self):
        npt.assert_allclose(self.spec.multipole_spectra(1.0, 0.1, 2),
                            self.check_multipole_spectra_m2,
                            rtol=1e-06,
                            err_msg='Multipole spectrum m = 2 failed')

    def test_multipole_spectra_m3(self):
        npt.assert_allclose(self.spec.multipole_spectra(1.0, 0.1, 3),
                            self.check_multipole_spectra_m3,
                            atol=1e-06,
                            err_msg='Multipole spectrum m = 3 failed')

    def test_multipole_spectra_m4(self):
        npt.assert_allclose(self.spec.multipole_spectra(1.0, 0.1, 4),
                            self.check_multipole_spectra_m4,
                            rtol=1e-06,
                            err_msg='Multipole spectrum m = 4 failed')

    def test_multipole_spectra_integrand(self):
        npt.assert_allclose(self.spec.multipole_spectra_integrand(0.7, 1.0,
                                                                  0.1, 2),
                            self.check_multipole_spectra_integrand,
                            rtol=1e-06,
                            err_msg='Multipole spectra integrand failed')

    def test_pkgal_linear(self):
        npt.assert_allclose(self.spec.pkgal_linear(0.7, 1.0, 0.1),
                            self.check_pkgal_linear,
                            rtol=1e-06,
                            err_msg='Linear galaxy spectrum failed')

    def test_scaling_factor_perp(self):
        npt.assert_allclose(self.spec.scaling_factor_perp(0.01),
                            self.check_scaling_factor_perp,
                            rtol=1e-03,
                            err_msg='Scaling Factor Perp failed')

    def test_scaling_factor_parall(self):
        npt.assert_allclose(self.spec.scaling_factor_parall(0.01),
                            self.check_scaling_factor_parall,
                            rtol=1e-03,
                            err_msg='Scaling Factor Parall failed')

    def test_get_k(self):
        npt.assert_allclose(self.spec.get_k(0.001, 1, 0.01),
                            self.check_get_k,
                            rtol=1e-03,
                            err_msg='get_k failed')

    def test_get_mu(self):
        npt.assert_allclose(self.spec.get_mu(1, 0.01),
                            self.check_get_mu,
                            rtol=1e-03,
                            err_msg='get_mu failed')
