"""UNIT TESTS FOR COSMO

This module contains unit tests for the cosmo module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.cosmo.cosmology import Cosmology
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class cosmoinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Define standard test case
        cls.cosmo = Cosmology()
        cls.cosmo.cosmo_dic = load_test_pickle('cosmo_test_dic.pickle')
        cls.cosmo.nonlinear.theory['redshift_bins'] = \
            cls.cosmo.cosmo_dic['redshift_bins']
        cls.cosmo.nonlinear.set_Pgg_spectro_model()
        # Define test case for negative curvature
        cls.cosmo_curv_neg = Cosmology()
        cls.cosmo_curv_neg.cosmo_dic = (
            load_test_pickle('cosmo_test_curv_neg_dic.pickle')
        )
        # Define test case for positive curvature
        cls.cosmo_curv_pos = Cosmology()
        cls.cosmo_curv_pos.cosmo_dic = (
            load_test_pickle('cosmo_test_curv_pos_dic.pickle')
        )

    def setUp(self) -> None:
        # Check values
        self.H0check = 67.0
        self.Dcheck = 1.0
        self.fcheck = 0.525454
        self.Hcheck = 74.349422
        self.fKcheck = 3415.832866
        self.fK12check = [1454.728042, 2538.164745, 3369.714525]
        self.fK12check_curv_neg = [2897.029977, 1600.91614, 691.969792]
        self.fK12check_curv_pos = 1886.960791
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
        self.matter_density = 0.317763

    def tearDown(self):
        self.H0check = None
        self.Dcheck = None
        self.fcheck = None
        self.Hcheck = None
        self.fKcheck = None
        self.fK12check = None
        self.fK12check_curv_neg = None
        self.fK12check_curv_pos = None
        self.Pgg_phot_test = None
        self.Pgdelta_phot_test = None
        self.Pgdelta_phot_test_interpolation = None
        self.Pgg_spectro_test = None
        self.Pgdelta_spectro_test = None
        self.Pii_test = None
        self.Pdeltai_test = None
        self.Pgi_phot_test = None
        self.Pgi_spectro_test = None
        self.omega_density = None

    def test_cosmo_init(self):
        emptflag = bool(self.cosmo.cosmo_dic)
        npt.assert_equal(
            emptflag,
            True,
            err_msg='Cosmology dictionary not initialised correctly.',
        )

    def test_cosmo_nuisance_init(self):
        emptflag = bool(self.cosmo.cosmo_dic['nuisance_parameters'])
        npt.assert_equal(
            emptflag,
            True,
            err_msg=(
                'nuisance parameters not initialized within cosmology ' +
                'dictionary'
            ),
        )

    def test_cosmo_asign(self):
        npt.assert_allclose(
            self.cosmo.cosmo_dic['H0'],
            self.H0check,
            err_msg='Cosmology dictionary assignment failed',
        )
        npt.assert_allclose(
            self.cosmo.cosmo_dic['H_z_func'](0.2),
            self.Hcheck,
            err_msg='Cosmology dictionary assignment failed',
        )

    def test_pk_source(self):
        source = self.cosmo.pk_source_phot
        type_check = isinstance(source, Cosmology)
        assert type_check, 'Error in returned data type of pk_source'
        # To test the nonlinear case we simply check if the object returned
        # from pk_source is not a Cosmology object. Otherwise we should
        # import the Nonlinear class, going against the independency among
        # different modules in the unit tests
        self.cosmo.cosmo_dic['NL_flag_phot_matter'] = 1
        source = self.cosmo.pk_source_phot
        type_check = isinstance(source, Cosmology)
        assert not type_check, 'Error in returned data type of pk_source'
        self.cosmo.cosmo_dic['NL_flag_phot_matter'] = 0

    def test_cosmo_growth_factor(self):
        npt.assert_equal(
            self.cosmo.growth_factor(0.0, 0.002),
            self.Dcheck,
            err_msg='Error in D_z_k calculation',
        )

    def test_cosmo_growth_factor_interp(self):
        npt.assert_allclose(
            self.cosmo.cosmo_dic['D_z_k_func'](0.0, 0.002),
            self.Dcheck,
            rtol=1e-3,
            err_msg='Error in growth factor interpolation',
        )

    def test_cosmo_growth_rate(self):
        f = self.cosmo.growth_rate(self.cosmo.cosmo_dic['z_win'], 0.002)
        npt.assert_allclose(
            f(0),
            self.fcheck,
            rtol=1e-3,
            err_msg='Error in f_z_k calculation',
        )

    def test_transverse_comoving_dist(self):
        npt.assert_allclose(
            self.cosmo.cosmo_dic['f_K_z_func'](1.0),
            self.fKcheck,
            rtol=1e-3,
            err_msg='Error in transverse comoving distance interpolation',
        )

    def test_transverse_comoving_dist_z12(self):
        f_K = self.cosmo.cosmo_dic['f_K_z12_func'](
            0.5,
            np.array([1.0, 1.5, 2.0]),
        )
        npt.assert_allclose(
            f_K,
            self.fK12check,
            rtol=1e-3,
            err_msg=(
                'Error in transverse comoving distance (z1/z2) ' +
                'interpolation (K=0)'
            )
        )

    def test_transverse_comoving_dist_z12_curv_neg(self):
        f_K = self.cosmo_curv_neg.cosmo_dic['f_K_z12_func'](
            np.array([0.5, 1.0, 1.5]),
            2.0,
        )
        npt.assert_allclose(
            f_K,
            self.fK12check_curv_neg,
            rtol=1e-3,
            err_msg=(
                'Error in transverse comoving distance (z1/z2) ' +
                'interpolation (K<0)'
            ),
        )

    def test_transverse_comoving_dist_z12_curv_pos(self):
        f_K = self.cosmo_curv_pos.cosmo_dic['f_K_z12_func'](0.5, 1.0)
        npt.assert_allclose(
            f_K,
            self.fK12check_curv_pos,
            rtol=1e-3,
            err_msg=(
                'Error in transverse comoving distance (z1/z2) ' +
                'interpolation (K>0)'
            ),
        )

    def test_update_cosmo_dic(self):
        self.cosmo.update_cosmo_dic(self.cosmo.cosmo_dic['z_win'], 0.002)
        keyDfound = 'D_z_k' in self.cosmo.cosmo_dic
        npt.assert_equal(keyDfound, True, err_msg='D_z_k not calculated')

    def test_compute_phot_galbias(self):
        # interpolate a straight-line (b, z) grid to ease the checks
        nuipar = self.cosmo.cosmo_dic['nuisance_parameters']
        nuipar['bias_model'] = 1
        zs_means = [1.0, 2.0, 3.0]
        nuipar['b1_photo'] = 2.0
        nuipar['b2_photo'] = 4.0
        nuipar['b3_photo'] = 6.0
        self.cosmo.create_phot_galbias(zs_means)
        # check scalar redshift input
        bi_val_actual = self.cosmo.compute_phot_galbias(1.5)
        npt.assert_almost_equal(
            bi_val_actual,
            desired=3.0,
            decimal=3,
            err_msg='Error in compute_phot_galbias',
        )
        # check redshift input below zs edges: returns b(z at edge)
        bi_val_actual = \
            self.cosmo.compute_phot_galbias(0.98)
        npt.assert_almost_equal(
            bi_val_actual,
            desired=2.0,
            decimal=3,
            err_msg='Error in compute_phot_galbias (z<)',
        )
        # check redshift input above zs edges: returns b(z at edge)
        bi_val_actual = self.cosmo.compute_phot_galbias(3.04)
        npt.assert_almost_equal(
            bi_val_actual,
            desired=6.0,
            decimal=3,
            err_msg='Error in compute_phot_galbias (z>)',
        )
        # check vector redshift input: output size and values
        zs_vec = [1.5, 2.5]
        bi_val_actual = self.cosmo.compute_phot_galbias(zs_vec)
        npt.assert_equal(
            len(bi_val_actual),
            len(zs_vec),
            err_msg=(
                'Output size of compute_phot_galbias (array) '
                'does not match with input'
            ),
        )
        npt.assert_allclose(
            bi_val_actual,
            desired=[3.0, 5.0],
            rtol=1e-3,
            err_msg='Array output compute_phot_galbias',
        )

    def test_poly_phot_galbias(self):
        nuipar = self.cosmo.cosmo_dic['nuisance_parameters']
        nuipar['b0_poly_photo'] = 1.0
        nuipar['b1_poly_photo'] = 1.0
        nuipar['b2_poly_photo'] = 1.0
        nuipar['b3_poly_photo'] = 1.0
        b_val = self.cosmo.poly_phot_galbias(1.0)
        z_arr = np.array([1.0, 2.0])
        b_val_arr = self.cosmo.poly_phot_galbias(z_arr)
        # check float redshift input
        npt.assert_almost_equal(
            b_val,
            desired=4.0,
            decimal=8,
            err_msg='Error in poly_phot_galbias float redshift case'
        )
        # check array redshift input
        npt.assert_allclose(
            b_val_arr,
            desired=[4.0, 15.0],
            rtol=1e-8,
            err_msg='Error in poly_phot_galbias redshift array case'
        )

    def test_istf_spectro_galbias(self):
        # assign custom b1_spectro and b2_spectro for maintainability
        nuipar = self.cosmo.cosmo_dic['nuisance_parameters']
        nuipar['b1_spectro_bin1'] = 1.4614804
        nuipar['b1_spectro_bin2'] = 1.6060949
        # check scalar redshift input: z in 1st bin returns b1_spectro
        bi_val_actual = self.cosmo.istf_spectro_galbias(1.0)
        bi_val_desired = nuipar['b1_spectro_bin1']
        npt.assert_equal(
            bi_val_actual,
            bi_val_desired,
            err_msg='Error in istf_spectro_galbias: b1_spectro_bin1',
        )
        # check redshift outside bounds, with z=0 and z=10
        npt.assert_raises(
            ValueError,
            self.cosmo.istf_spectro_galbias,
            redshift=0,
        )
        npt.assert_raises(
            ValueError,
            self.cosmo.istf_spectro_galbias,
            redshift=10,
        )
        # check vector redshift input: output size and values
        zs_vec = [1.0, 1.2]
        bi_val_actual = self.cosmo.istf_spectro_galbias(zs_vec)
        bi_val_desired = [nuipar['b1_spectro_bin1'], nuipar['b1_spectro_bin2']]
        npt.assert_equal(
            len(bi_val_actual),
            len(zs_vec),
            err_msg=(
                'Output size of istf_spectro_galbias (vec) does not match ' +
                'with input'
            ),
        )
        npt.assert_array_equal(
            bi_val_actual,
            bi_val_desired,
            err_msg='Array output istf_spectro_galbias',
        )

    def test_Pgg_phot(self):
        test_p = self.cosmo.Pgg_phot_def(1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pgg_phot_test,
            rtol=1e-3,
            err_msg='Error in GC-phot Pgg calculation',
        )

    def test_Pgdelta_phot(self):
        test_p = self.cosmo.Pgdelta_phot_def(1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pgdelta_phot_test,
            rtol=1e-3,
            err_msg='Error in GC-phot Pgdelta calculation',
        )

    def test_Pgg_spectro(self):
        test_p = self.cosmo.Pgg_spectro_def(1.0, 0.01, 0.5)
        npt.assert_allclose(
            test_p,
            self.Pgg_spectro_test,
            rtol=1e-3,
            err_msg='Error in GC-spectro Pgg calculation',
        )

    def test_Pgdelta_spectro(self):
        test_p = self.cosmo.Pgdelta_spectro_def(1.0, 0.01, 0.5)
        npt.assert_allclose(
            test_p,
            self.Pgdelta_spectro_test,
            rtol=1e-3,
            err_msg='Error in GC-spectro Pgdelta calculation',
        )

    def test_Pgg_phot_interp(self):
        test_p = self.cosmo.cosmo_dic['Pgg_phot'](1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pgg_phot_test_interpolation,
            rtol=1e-3,
            err_msg='Error in GC-phot Pgg interpolation',
        )

    def test_Pgdelta_phot_interp(self):
        test_p = self.cosmo.cosmo_dic['Pgdelta_phot'](1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pgdelta_phot_test_interpolation,
            rtol=1e-3,
            err_msg='Error in GC-phot Pgdelta interpolation',
        )

    def test_Pgg_spectro_interp(self):
        test_p = self.cosmo.cosmo_dic['Pgg_spectro'](1.0, 0.01, 0.5)
        npt.assert_allclose(
            test_p,
            self.Pgg_spectro_test,
            rtol=1e-3,
            err_msg='Error in GC-spectro Pgg (cosmo-dic)',
        )

    def test_Pgdelta_spectro_interp(self):
        pgd = self.cosmo.cosmo_dic['Pgdelta_spectro'](1.0, 0.01, 0.5)
        npt.assert_allclose(
            pgd,
            self.Pgdelta_spectro_test,
            rtol=1e-3,
            err_msg='Error in GC-spectro Pgdelta (cosmo-dic)',
        )

    def test_Pii(self):
        test_p = self.cosmo.Pii_def(1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pii_test,
            rtol=1e-3,
            err_msg='Error in Pii calculation',
        )

    def test_Pdeltai(self):
        test_p = self.cosmo.Pdeltai_def(1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pdeltai_test,
            rtol=1e-3,
            err_msg='Error in Pdeltai calculation',
        )

    def test_Pgi_phot(self):
        test_p = self.cosmo.Pgi_phot_def(1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pgi_phot_test,
            rtol=1e-3,
            err_msg='Error in GC-phot Pgi calculation',
        )

    def test_Pgi_spectro(self):
        test_p = self.cosmo.Pgi_spectro_def(1.0, 0.01)
        npt.assert_allclose(
            test_p,
            self.Pgi_spectro_test,
            rtol=1e-3,
            err_msg='Error in GC-spectro Pgi calculation',
        )

    def test_MG_mu_def(self):
        test_mu = self.cosmo.MG_mu_def(1.0, 0.01, self.MG_mu_test)
        npt.assert_equal(
            test_mu,
            self.MG_mu_test,
            err_msg='Error in MG mu calculation',
        )

    def test_MG_sigma_def(self):
        test_sigma = (
            self.cosmo.MG_sigma_def(1.0, 0.01, self.MG_sigma_test)
        )
        npt.assert_equal(
            test_sigma,
            self.MG_sigma_test,
            err_msg='Error in MG sigma calculation',
        )

    def test_MG_mu(self):
        test_mu = self.cosmo.cosmo_dic['MG_mu'](1.0, 0.01)
        npt.assert_equal(
            test_mu,
            self.MG_mu_test,
            err_msg='Error in MG mu calculation',
        )

    def test_MG_sigma(self):
        test_sigma = self.cosmo.cosmo_dic['MG_sigma'](1.0, 0.01)
        npt.assert_equal(
            test_sigma,
            self.MG_sigma_test,
            err_msg='Error in MG sigma calculation',
        )

    def test_matter_density(self):
        test_matter_density = (
            self.cosmo.matter_density(self.cosmo.cosmo_dic['z_win'])
        )
        npt.assert_allclose(
            test_matter_density[0],
            self.matter_density,
            rtol=1e-3,
            err_msg='Error in the omega density calculation',
        )

    def test_growth_rate_MG(self):
        self.cosmo.growth_rate_MG(self.cosmo.cosmo_dic['z_win'])
        npt.assert_allclose(
            self.cosmo.cosmo_dic['f_z'](0),
            self.fcheck,
            rtol=1e-1,
            err_msg='Error in the omega density calculation',
        )

    def test_growth_factor_MG(self):
        self.cosmo.growth_rate_MG(self.cosmo.cosmo_dic['z_win'])
        test_growth_factor_MG = self.cosmo.growth_factor_MG()
        npt.assert_allclose(
            test_growth_factor_MG[0],
            self.Dcheck,
            rtol=1e-1,
            err_msg='Error in the omega density calculation',
        )

    def test_z_of_r_inverse(self):
        # test that z_r_func is really the inverse of r_z_func
        z_in = [0.5, 0.7, 1.]
        r_out = self.cosmo.cosmo_dic['r_z_func'](z_in)
        z_out = self.cosmo.cosmo_dic['z_r_func'](r_out)
        npt.assert_allclose(
            z_out,
            z_in,
            rtol=1e-3,
            err_msg='z_r_func is not the inverse of r_z_func',
        )

    def test_z_win_exception(self):
        z_win = self.cosmo.cosmo_dic['z_win']
        self.cosmo.cosmo_dic['z_win'] = None
        npt.assert_raises(
            Exception,
            self.cosmo.update_cosmo_dic,
            None,
            0.002,
        )
        self.cosmo.cosmo_dic['z_win'] = z_win
