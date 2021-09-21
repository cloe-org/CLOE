"""UNIT TESTS FOR COSMO

This module contains unit tests for the cosmo module.
=======

"""

from unittest import TestCase
import numpy.testing as npt
from astropy import constants as const
from likelihood.cosmo.cosmology import Cosmology
from likelihood.tests.test_wrapper import CobayaModel


class cosmoinitTestCase(TestCase):

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
        cosmo.cosmo_dic['nuisance_parameters']['NL_flag'] = 0
        # Create wrapper model
        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()
        # Check values
        self.H0check = 67.0
        self.Dcheck = 1.0
        self.fcheck = 0.525454
        self.Hcheck = 74.349422
        self.Pgg_phot_test = 58392.759202
        self.Pgg_phot_test_interpolation = 58332.02434
        self.Pgdelta_phot_test = 41294.412017
        self.Pgdelta_phot_test_interpolation = 41253.453724
        self.Pgg_spectro_test = 82548.320427
        self.Pgdelta_spectro_test = 59890.445816
        self.Pii_test = 2.417099
        self.Pdeltai_test = -265.680094
        self.Pgi_phot_test = -375.687484
        self.Pgi_spectro_test = -388.28625
        self.MG_mu_test = 1.0
        self.MG_sigma_test = 1.0

    def tearDown(self):
        self.H0check = None
        self.Dcheck = None
        self.fcheck = None
        self.Hcheck = None
        self.Pgg_phot_test = None
        self.Pgdelta_phot_test = None
        self.Pgdelta_phot_test_interpolation = None
        self.Pgg_spectro_test = None
        self.Pgdelta_spectro_test = None
        self.Pii_test = None
        self.Pdeltai_test = None
        self.Pgi_phot_test = None
        self.Pgi_spectro_test = None

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

    def test_pk_source(self):
        source = self.model_test.cosmology.pk_source
        type_check = isinstance(source, Cosmology)
        assert type_check, 'Error in returned data type of pk_source'
        # To test the nonlinear case we simply check if the object returned
        # from pk_source is not a Cosmology object. Otherwise we should
        # import the Nonlinear class, going against the independency among
        # different modules in the unit tests
        self.model_test.cosmology.cosmo_dic[
            'nuisance_parameters']['NL_flag'] = 1
        source = self.model_test.cosmology.pk_source
        type_check = isinstance(source, Cosmology)
        assert not type_check, 'Error in returned data type of pk_source'

    def test_cosmo_growth_factor(self):
        D = self.model_test.cosmology.growth_factor(0.0, 0.002)
        npt.assert_equal(D, self.Dcheck,
                         err_msg='Error in D_z_k calculation ')

    def test_cosmo_growth_factor_interp(self):
        D = self.model_test.cosmology.cosmo_dic['D_z_k_func'](0.0, 0.002)
        npt.assert_allclose(D, self.Dcheck,
                            rtol=1e-3,
                            err_msg='Error in growth factor interpolation')

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
        npt.assert_equal(emptflag_D, True,
                         err_msg='D_z_k not calculated ')

    def test_istf_phot_galbias(self):
        # interpolate a straight-line (b, z) grid to ease the checks
        nuipar = self.model_test.cosmology.cosmo_dic['nuisance_parameters']
        zs_means = [1., 2., 3.]
        nuipar['b1_photo'] = 2.
        nuipar['b2_photo'] = 4.
        nuipar['b3_photo'] = 6.
        self.model_test.cosmology.istf_phot_galbias_interpolator(zs_means)
        # check scalar redshift input
        bi_val_actual = self.model_test.cosmology.istf_phot_galbias(1.5)
        npt.assert_almost_equal(bi_val_actual, desired=3.,
                                decimal=3,
                                err_msg='Error in istf_phot_galbias')
        # check redshift input below zs edges: returns b(z at edge)
        bi_val_actual = \
            self.model_test.cosmology.istf_phot_galbias(0.98, [1., 2.])
        npt.assert_almost_equal(bi_val_actual, desired=2.,
                                decimal=3,
                                err_msg='Error in istf_phot_galbias (z<)')
        # check redshift input above zs edges: returns b(z at edge)
        bi_val_actual = \
            self.model_test.cosmology.istf_phot_galbias(2.04, [1., 2.])
        npt.assert_almost_equal(bi_val_actual, desired=4.,
                                decimal=3,
                                err_msg='Error in istf_phot_galbias (z>)')
        # check vector redshift input: output size and values
        zs_vec = [1.5, 2.5]
        bi_val_actual = self.model_test.cosmology.istf_phot_galbias(zs_vec)
        npt.assert_equal(len(bi_val_actual), len(zs_vec),
                         err_msg='Output size of istf_phot_galbias'
                                 '(vec) does not match with input')
        npt.assert_allclose(bi_val_actual, desired=[3., 5.],
                            rtol=1e-3,
                            err_msg='Array output istf_phot_galbias')

    def test_istf_spectro_galbias(self):
        # assign custom b1_spectro and b2_spectro for maintainability
        nuipar = self.model_test.cosmology.cosmo_dic['nuisance_parameters']
        nuipar['b1_spectro'] = 1.23
        nuipar['b2_spectro'] = 4.56
        # check scalar redshift input: z in 1st bin returns b1_spectro
        bi_val_actual = \
            self.model_test.cosmology.istf_spectro_galbias(1.5, [1., 2.])
        bi_val_desired = nuipar['b1_spectro']
        npt.assert_equal(bi_val_actual, bi_val_desired,
                         err_msg='Error in istf_spectro_galbias: b1_spectro')
        # check redshift outside bounds, with z=0 and z=10
        npt.assert_raises(ValueError,
                          self.model_test.cosmology.istf_spectro_galbias,
                          redshift=0)
        npt.assert_raises(ValueError,
                          self.model_test.cosmology.istf_spectro_galbias,
                          redshift=10)
        # check vector redshift input: output size and values
        zs_vec = [1.5, 2.5]
        z_edge = [1.0, 2.0, 3.0]
        bi_val_actual = \
            self.model_test.cosmology.istf_spectro_galbias(zs_vec, z_edge)
        bi_val_desired = [nuipar['b1_spectro'], nuipar['b2_spectro']]
        npt.assert_equal(len(bi_val_actual), len(zs_vec),
                         err_msg='Output size of istf_spectro_galbias'
                                 '(vec) does not match with input')
        npt.assert_array_equal(bi_val_actual, bi_val_desired,
                               err_msg='Array output istf_spectro_galbias')

    def test_Pgg_phot(self):
        test_p = self.model_test.cosmology.Pgg_phot_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgg_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgg calculation')

    def test_Pgdelta_phot(self):
        test_p = self.model_test.cosmology.Pgdelta_phot_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgdelta_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgdelta calculation')

    def test_Pgg_spectro(self):
        test_p = self.model_test.cosmology.Pgg_spectro_def(1.0, 0.01, 0.5)
        npt.assert_allclose(test_p, self.Pgg_spectro_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spectro Pgg calculation')

    def test_Pgdelta_spectro(self):
        test_p = self.model_test.cosmology.Pgdelta_spectro_def(1.0, 0.01, 0.5)
        npt.assert_allclose(test_p, self.Pgdelta_spectro_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spectro Pgdelta calculation')

    def test_Pgg_phot_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgg_phot'](1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgg_phot_test_interpolation,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgg interpolation')

    def test_Pgdelta_phot_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgdelta_phot'](1.0,
                                                                     0.01)
        npt.assert_allclose(test_p, self.Pgdelta_phot_test_interpolation,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgdelta interpolation')

    def test_Pgg_spectro_interp(self):
        test_p = self.model_test.cosmology.cosmo_dic['Pgg_spectro'](1.0,
                                                                    0.01,
                                                                    0.5)
        npt.assert_allclose(test_p, self.Pgg_spectro_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spectro Pgg (cosmo-dic)')

    def test_Pgdelta_spectro_interp(self):
        pgd = self.model_test.cosmology.cosmo_dic['Pgdelta_spectro'](1.0,
                                                                     0.01,
                                                                     0.5)
        npt.assert_allclose(pgd, self.Pgdelta_spectro_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spectro Pgdelta (cosmo-dic)')

    def test_Pii(self):
        test_p = self.model_test.cosmology.Pii_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pii_test,
                            rtol=1e-3,
                            err_msg='Error in Pii calculation')

    def test_Pdeltai(self):
        test_p = self.model_test.cosmology.Pdeltai_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pdeltai_test,
                            rtol=1e-3,
                            err_msg='Error in Pdeltai calculation')

    def test_Pgi_phot(self):
        test_p = self.model_test.cosmology.Pgi_phot_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgi_phot_test,
                            rtol=1e-3,
                            err_msg='Error in GC-phot Pgi calculation')

    def test_Pgi_spectro(self):
        test_p = self.model_test.cosmology.Pgi_spectro_def(1.0, 0.01)
        npt.assert_allclose(test_p, self.Pgi_spectro_test,
                            rtol=1e-3,
                            err_msg='Error in GC-spectro Pgi calculation')

    def test_MG_mu_def(self):
        test_mu = self.model_test.cosmology.MG_mu_def(1.0, 0.01,
                                                      self.MG_mu_test)
        npt.assert_equal(test_mu, self.MG_mu_test,
                         err_msg='Error in MG mu calculation')

    def test_MG_sigma_def(self):
        test_sigma = self.model_test.cosmology.MG_sigma_def(1.0, 0.01,
                                                            self.MG_sigma_test)
        npt.assert_equal(test_sigma, self.MG_sigma_test,
                         err_msg='Error in MG sigma calculation')

    def test_MG_mu(self):
        test_mu = self.model_test.cosmology.cosmo_dic['MG_mu'](1.0, 0.01)
        npt.assert_equal(test_mu, self.MG_mu_test,
                         err_msg='Error in MG mu calculation')

    def test_MG_sigma(self):
        test_sigma = self.model_test.cosmology.cosmo_dic['MG_sigma'](1.0, 0.01)
        npt.assert_equal(test_sigma, self.MG_sigma_test,
                         err_msg='Error in MG sigma calculation')
