"""UNIT TESTS FOR Galaxy Clusters

This module contains unit tests for the :obj:`CG` sub-module of the
clusters_of_galaxies module.

"""

from unittest.mock import patch
from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.clusters_of_galaxies.CG import CG
from cloe.cobaya_interface import EuclidLikelihood
from cobaya.model import get_model

info = {
    'params': {
        'ombh2': 0.022387993,
        'omch2': 0.120088016,
        'H0': 67.32,
        'tau': 0.0925,
        'mnu': (0.3158 - 0.314378999) * 93.04 * pow(0.6732, 2),
        # 'mnu': 0.0,
        'nnu': 3.046,
        # 'neutrino_cdm': 'cb',
        'As': 2.09175e-9,
        'ns': 0.9661,
        'w': -1,
        'wa': 0,
        'omk': 0.0,
        'omegam': None,
        'omegab': None,
        'omeganu': None,
        'omnuh2': None,
        'omegac': None,
        'N_eff': None,
    },
    'theory': {'camb':
               {'stop_at_error': True,
                'extra_args': {
                    'num_massive_neutrinos': 1,
                    'dark_energy_model': 'ppf'}}},
    'sampler': {'evaluate': None},
    'output': 'chains/my_euclid_experiment',
    'likelihood': {
        'Euclid': {
                'external': EuclidLikelihood,
                'speed': 500,
                'k_max_extrap': 50.0,
                'k_min_extrap': 1e-05,
                'k_samp': 1000,
                'z_min': 0.0,
                'z_max': 4.0,
                'z_samp': 100,
                'solver': 'camb',
                'NL_flag_phot_matter': 0,
                'NL_flag_spectro': 0,
                'NL_flag_phot_baryon': 0,
                'NL_flag_phot_bias': 0,
                'IA_flag': 0,
                'IR_resum': 'DST',
                'Baryon_redshift_model': True,
                'GCsp_z_err': False,
                'bias_model': 1,
                'use_magnification_bias_spectro': 0,
                'use_Weyl': False,
                'magbias_model': 2,
                'use_gamma_MG': False,
                'f_out_z_dep': False,
                'plot_observables_selection': False,
                'data':
                {
                    'sample': 'ExternalBenchmark',
                    'CG':
                    {
                         'file_names_CC': 'data_CG_CC.dat',
                         'file_cov_names_CC': 'data_cov_CG_CC.dat',
                         'file_names_MoR': 'data_CG_MoR.dat',
                         'file_cov_names_MoR': 'data_cov_CG_MoR.dat',
                         'file_names_xi2': 'data_CG_xi2.npy',
                         'file_cov_names_xi2': 'data_cov_CG_xi2.npy'
                    },
                    'photo': {
                        'redshifts': [
                            0.2095, 0.489, 0.619, 0.7335, 0.8445,
                            0.9595, 1.087, 1.2395, 1.45, 2.038],
                        'luminosity_ratio': 'luminosity_ratio.dat',
                        'IA_model': 'zNLA',
                        'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npy',
                        'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npy',
                        'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npy',
                        'cov_model': 'Gauss',
                        'cov_is_num': False,
                        'cov_nsim': 10000,
                        'ndens_GC': 'niTab-EP10-RB00.dat',
                        'ndens_WL': 'niTab-EP10-RB00.dat',
                        'root_GC': 'Cls_{:s}_PosPos.dat',
                        'root_WL': 'Cls_{:s}_ShearShear.dat',
                        'root_XC': 'Cls_{:s}_PosShear.dat',
                    },
                    'spectro':
                    {
                        'redshifts': ['1.', '1.2', '1.4', '1.65'],
                        'edges': [0.9, 1.1, 1.3, 1.5, 1.8],
                        'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
                        'cov_is_num': False,
                        'cov_nsim': 3500,
                    },
                },
                'observables_selection':
                {
                    'CG':
                    {
                        'CG': True,
                    },
                    'WL':
                    {
                        'WL': False,
                        'GCphot': False,
                        'GCspectro': False,
                    },
                    'GCphot':
                    {
                        'GCphot': False,
                        'GCspectro': False,
                    },
                    'GCspectro':
                    {
                        'GCspectro': False,
                    },
                    'add_phot_RSD': False,
                    'matrix_transform_phot': False,
                },
                'observables_specifications':
                {
                    'CG': {
                        'statistics_clusters': 'cluster_counts',
                        'CG_probe': 'CC',
                        'CG_xi2_cov_selection': 'CG_nonanalytic_cov',
                        'neutrino_cdm': 'cb',
                        'external_richness_selection_function': 'non_CG_ESF',
                        'file_richness_selection_function': './../data/' +
                        'ExternalBenchmark/Clusters/int_Plob_ltr_z_Dlob.npy',
                        'effective_area': 10313,
                        'z_obs_edges_CC': np.array([
                            0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6
                        ]),
                        'Lambda_obs_edges_CC': np.array([20, 30, 45, 60, 500]),
                        'halo_profile': 'BMO',
                        'overdensity_type': 'vir',
                        'overdensity': 200.,
                        'A_l': 52.0,
                        'B_l': 0.9,
                        'C_l': 0.5,
                        'sig_A_l': 0.2,
                        'sig_B_l': -0.05,
                        'sig_C_l': 0.001,
                        'M_pivot': 3.e14 / 0.6732,
                        'z_pivot': 0.45,
                        'sig_lambda_norm': 0.9,
                        'sig_lambda_z': 0.1,
                        'sig_lambda_exponent': 0.4,
                        'sig_z_norm': 0.,
                        'sig_z_z': 0.025,
                        'sig_z_lambda': 5.e-6,
                    },
                    'GCphot-GCspectro': None,
                    'WL-GCspectro': None
                },
        },
    },
    'debug': False,
    'timing': False,
    'force': True
}


class CGinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        model = get_model(info)
        logposterior = model.logposterior({})
        like = EuclidLikelihood()

        like.passing_requirements(model, info, **model.provider.params)
        like.cosmo.update_cosmo_dic(like.cosmo.cosmo_dic['z_win'], 0.05)
        like.initialize()

        # Again for some reason
        like.passing_requirements(model, info, **model.provider.params)
        like.cosmo.update_cosmo_dic(like.cosmo.cosmo_dic['z_win'], 0.05)

        cls.cg = CG(like.cosmo.cosmo_dic)

    def setUp(self) -> None:
        self.check_Pk_def = 52.413713
        self.check_dVdzdO = 2.161841e-26
        self.check_rho_crit_0 = 125779111003.34
        self.check_rho_mean_0 = 39542311001.30
        self.check_radius_M = 8.45181108356498
        self.check_Omm_z = 0.3157999616810898
        self.check_delta_c = 1.684307937842588
        self.check_sigma_z_M = 1.0188809669207743
        self.check_nu_z_M = 1.645027349652875
        self.check_window_W = 0.9374456450495531
        self.check_window_dWdx = -0.1528144928443091
        self.check_dlnsdlnR = -0.6283614300433848
        self.check_f_sigma_nu = 0.24956734740428954
        self.check_dndm = 2.066988570730013e-19
        self.check_lnlambda = 2.4205693455649495
        self.check_scatter_lnlambda = 0.27434469171024733
        self.check_Pscaling_relation = 8.148686296932156e-05
        self.check_scatter_Plambda_obs = 3.5082538568245707
        self.check_Plambda_obs = 0.11371534007591107
        self.check_scatter_z_obs = 0.00014999999999999912
        self.check_Pz_obs = 2659.6152026762334
        self.check_Delta = 103.2914180724695
        self.check_halo_bias = 1.6605423679967524
        self.check_cov_window = np.array([5.158245, 4.531227, 3.920613])

    def tearDown(self):
        self.check_Pk_def = None
        self.check_dVdzdO = None
        self.check_rho_crit_0 = None
        self.check_rho_mean_0 = None
        self.check_radius_M = None
        self.check_Omm_z = None
        self.check_delta_c = None
        self.check_sigma_z_M = None
        self.check_nu_z_M = None
        self.check_window_W = None
        self.check_window_dWdx = None
        self.check_dlnsdlnR = None
        self.check_f_sigma_nu = None
        self.check_dndm = None
        self.check_lnlambda = None
        self.check_scatter_lnlambda = None
        self.check_Pscaling_relation = None
        self.check_scatter_Plambda_obs = None
        self.check_Plambda_obs = None
        self.check_scatter_z_obs = None
        self.check_Pz_obs = None
        self.check_Delta = None
        self.check_halo_bias = None
        self.check_cov_window = None

    def test_Pk_def(self):
        npt.assert_allclose(
            self.cg.Pk_def(0.5, 1., 1.),
            self.check_Pk_def,
            rtol=1.e-3,
            err_msg='test_Pk_def failed',
        )

    def test_dVdzdO(self):
        npt.assert_allclose(
            self.cg.dVdzdO(0.0),
            self.check_dVdzdO,
            rtol=1.e-3,
            err_msg='test_dV_dzdO failed',
        )

    def test_rho_crit_0(self):
        npt.assert_allclose(
            self.cg.rho_crit_0(),
            self.check_rho_crit_0,
            rtol=1.e-3,
            err_msg='test_rho_crit_0 failed',
        )

    def test_rho_mean_0(self):
        npt.assert_allclose(
            self.cg.rho_mean_0(),
            self.check_rho_mean_0,
            rtol=1.e-3,
            err_msg='test_rho_mean_0 failed',
        )

    def test_radius_M(self):
        npt.assert_allclose(
            self.cg.radius_M(10.0**14.0),
            self.check_radius_M,
            rtol=1.e-3,
            err_msg='test_radius_M failed',
        )

    def test_Omm_z(self):
        npt.assert_allclose(
            self.cg.Omm_z(0.0, 0.0),
            self.check_Omm_z,
            rtol=1.e-3,
            err_msg='test_Omm_z failed',
        )

    def test_delta_c(self):
        npt.assert_allclose(
            self.cg.delta_c(1.0),
            self.check_delta_c,
            rtol=1e-3,
            err_msg='test_delta_c failed',
        )

    def test_sigma_z_M(self):
        npt.assert_allclose(
            self.cg.sigma_z_M(0.0, 10.0**14.)[0][0],
            self.check_sigma_z_M,
            rtol=1e-3,
            err_msg='test_sigma_z_M failed',
        )

    def test_nu_z_M(self):
        npt.assert_allclose(
            self.cg.nu_z_M(np.array([0.0]), 10.0**14.)[0][0],
            self.check_nu_z_M,
            rtol=1e-3,
            err_msg='test_nu_z_M failed',
        )

    def test_window_W(self):
        npt.assert_allclose(
            self.cg.window(0.1, np.array([8.0]))[0][0][0],
            self.check_window_W,
            rtol=1e-3,
            err_msg='test_window_W failed',
        )

    def test_window_dWdx(self):
        npt.assert_allclose(
            self.cg.window(0.1, np.array([8.0]))[1][0][0],
            self.check_window_dWdx,
            rtol=1e-3,
            err_msg='test_window_dWdx failed',
        )

    def test_dlnsdlnR(self):
        npt.assert_allclose(
            self.cg.dlnsdlnR(np.array([0.0]), np.array([10.0**14.]))[0][0],
            self.check_dlnsdlnR,
            rtol=1e-3,
            err_msg='test_dlns_dlnR failed',
        )

    def test_f_sigma_nu(self):
        npt.assert_allclose(
            self.cg.f_sigma_nu(np.array([0.0]), np.array([10.0**14.]))[0][0],
            self.check_f_sigma_nu,
            rtol=1e-3,
            err_msg='test_f_sigma_nu failed',
        )

    def test_dndm(self):
        npt.assert_allclose(
            self.cg.dndm(np.array([0.0]), np.array([10.0**14.]))[0][0],
            self.check_dndm,
            rtol=1e-3,
            err_msg='test_delta_c failed',
        )

    def test_lnlambda(self):
        npt.assert_allclose(
            self.cg.lnlambda(np.array([0.0]), np.array([10.0**14.]))[0][0],
            self.check_lnlambda,
            rtol=1.e-3,
            err_msg='test_lnlambda failed',
        )

    def test_scatter_lnlambda(self):
        npt.assert_allclose(
            self.cg.scatter_lnlambda(
                np.array([0.0]), np.array([10.0**14.])
            )[0][0],
            self.check_scatter_lnlambda,
            rtol=1e-3,
            err_msg='test_scatter_lnlambda failed',
        )

    def test_Pscaling_relation(self):
        npt.assert_allclose(
            self.cg.Pscaling_relation(
                np.array([0.0]), np.array([10.0**14.]),
                np.array([30.0])
            )[0][0][0],
            self.check_Pscaling_relation,
            rtol=1e-3,
            err_msg='test_Pscaling_relation failed',
        )

    def test_scatter_Plambda_obs(self):
        npt.assert_allclose(
            self.cg.scatter_Plambda_obs(
                np.array([0.0]),
                np.array([30.0])
            )[0][0],
            self.check_scatter_Plambda_obs,
            rtol=1e-3,
            err_msg='test_scatter_Plambda_obs failed',
        )

    def test_Plambda_obs(self):
        npt.assert_allclose(
            self.cg.Plambda_obs(
                np.array([0.0]), np.array([30.0]),
                np.array([30.0])
            )[0][0][0],
            self.check_Plambda_obs,
            rtol=1e-3,
            err_msg='test_Plambda_obs failed',
        )

    def test_scatter_z_obs(self):
        npt.assert_allclose(
            self.cg.scatter_z_obs(30.0, np.array([0.0]))[0],
            self.check_scatter_z_obs,
            rtol=1e-3,
            err_msg='test_scatter_z_obs failed',
        )

    def test_Pz_obs(self):
        npt.assert_allclose(
            self.cg.Pz_obs(np.array([0.0]), 30.0, np.array([0.0]))[0][0],
            self.check_Pz_obs,
            rtol=1.e-3,
            err_msg='test_Pz_obs failed',
        )

    def test_Delta(self):
        npt.assert_allclose(
            self.cg.Delta("vir", 0.0, 0.0, 200),
            self.check_Delta,
            rtol=1.e-3,
            err_msg='test_Delta failed',
        )

    def test_halo_bias(self):
        npt.assert_allclose(
            self.cg.halo_bias(np.array([0.0]), 10.0**14.,
                              self.cg.Delta("vir", 0.0, 0.0, 200))[0][0],
            self.check_halo_bias,
            rtol=1e-3,
            err_msg='test_halo_bias failed',
        )

    def test_cov_window(self):
        self.cg.rint = np.zeros((10, 250, 2))
        self.cg.KL = np.array([1., 2.])
        npt.assert_allclose(
            self.cg.cov_window(0, np.array([0.5, 1.]), 1., 1)[0][0:3],
            self.check_cov_window,
            rtol=1.e-3,
            err_msg='test_cov_window failed',
        )

    def test_number_counts(self):
        input_dir = './data/ExternalBenchmark/Clusters' + \
                    '/truth_tables/numbercounts_truth_tables/'

        # Compute the model with CLOE
        model_theo = self.cg.N_zbin_Lbin_Rbin()
        model_tr = np.genfromtxt(input_dir + "NC_CG_CC_model.dat")
        npt.assert_allclose(
            model_tr[0],
            model_theo[0][0],
            rtol=0.03,
            err_msg='Error in matching NC(z) CG data',
        )
