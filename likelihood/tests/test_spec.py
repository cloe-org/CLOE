"""UNIT TESTS FOR SHEAR

This module contains unit tests for the Spec sub-module of the
spectroscopy survey module.
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import interpolate
from likelihood.spectroscopic_survey.spec import Spec
from astropy import constants as const
from pathlib import Path


def mock_MG_func(z, k):
    """
    Test MG function that simply returns 1.

    Parameters
    ----------
    z: float
        Redshift.
    k: float
        Angular scale.

    Returns
    -------
    float
        Returns 1 for test purposes.
    """
    return 1.0


class mock_P_obj:
    def __init__(self, p_interp):
        self.P = p_interp


class cosmoinitTestCase(TestCase):

    def setUp(self):
        cur_dir = Path(__file__).resolve().parents[0]
        cmov_file = np.loadtxt(str(cur_dir) +
                               '/test_input/ComDist-LCDM-Lin-zNLA.dat')
        zs_r = cmov_file[:, 0]
        rs = cmov_file[:, 1]
        ang_dists = rs / (1.0 + zs_r)

        rz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r, y=rs,
                                                             ext=0)
        dz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r,
                                                             y=ang_dists,
                                                             ext=0)

        Hz_file = np.loadtxt(str(cur_dir) + '/test_input/Hz.dat')
        zs_H = Hz_file[:, 0]
        Hs = Hz_file[:, 1]
        Hs_mpc = Hz_file[:, 1] / const.c.to('km/s').value

        Hz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H, y=Hs,
                                                             ext=0)

        Hmpc_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                               y=Hs_mpc,
                                                               ext=0)

        f_sig_8_arr = np.load(str(cur_dir) +
                              '/test_input/f_sig_8_arr.npy',
                              allow_pickle=True)
        sig_8_arr = np.load(str(cur_dir) +
                            '/test_input/sig_8_arr.npy',
                            allow_pickle=True)
        f_z_arr = np.load(str(cur_dir) +
                          '/test_input/f_z_arr.npy',
                          allow_pickle=True)

        sig_8_interp = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(0.0, 5.0, 50),
            y=sig_8_arr[::-1], ext=0)
        f_sig_8_interp = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(0.0, 5.0, 50),
            y=f_sig_8_arr[::-1], ext=0)
        f_z_interp = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(0.0, 5.0, 50),
            y=f_z_arr[::-1], ext=0)

        MG_interp = mock_MG_func

        spec_zkm = np.load(str(cur_dir) + '/test_input/spec_zkm.npy')

        pdd = np.load(str(cur_dir) + '/test_input/pdd.npy')
        pdi = np.load(str(cur_dir) + '/test_input/pdi.npy')
        pgd = np.load(str(cur_dir) + '/test_input/pgd.npy')
        pgg = np.load(str(cur_dir) + '/test_input/pgg.npy')
        pgi_phot = np.load(str(cur_dir) + '/test_input/pgi_phot.npy')
        pgi_spec = np.load(str(cur_dir) + '/test_input/pgi_spec.npy')
        pgg_spec = np.load(str(cur_dir) + '/test_input/pgg_spec.npy')
        pii = np.load(str(cur_dir) + '/test_input/pii.npy')

        zs_base = np.linspace(0.0, 4.0, 100)
        ks_base = np.logspace(-3.0, 1.0, 100)

        mock_cosmo_dic = {'ombh2': 0.022445, 'omch2': 0.121203, 'H0': 67.0,
                          'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046,
                          'omkh2': 0.0, 'omnuh2': 0.0, 'ns': 0.96,
                          'w': -1.0, 'sigma_8_0': 0.816,
                          'As': 2.115e-9, 'sigma8_z_func': sig_8_interp,
                          'fsigma8_z_func': f_sig_8_interp,
                          'f_z': f_z_interp,
                          'r_z_func': rz_interp, 'd_z_func': dz_interp,
                          'H_z_func_Mpc': Hmpc_interp,
                          'H_z_func': Hz_interp,
                          'z_win': np.linspace(0.0, 4.0, 100),
                          'k_win': np.linspace(0.001, 10.0, 100),
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value}

        # MM: precomputed parameters
        mock_cosmo_dic['H0_Mpc'] = \
            mock_cosmo_dic['H0'] / const.c.to('km/s').value
        mock_cosmo_dic['Omb'] = \
            mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
        mock_cosmo_dic['Omc'] = \
            mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
        mock_cosmo_dic['Omnu'] = \
            mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
        mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                                 mock_cosmo_dic['Omc'] +
                                 mock_cosmo_dic['Omb'])

        p_matter = mock_P_obj(interpolate.interp2d(zs_base, ks_base, pdd.T,
                                                   fill_value=0))
        mock_cosmo_dic['Pk_interpolator'] = p_matter
        mock_cosmo_dic['Pk_delta'] = p_matter
        mock_cosmo_dic['Pgg_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgg.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgdelta_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                              pgd.T,
                                                              fill_value=0.0)
        mock_cosmo_dic['Pii'] = interpolate.interp2d(zs_base, ks_base,
                                                     pii.T,
                                                     fill_value=0.0)
        mock_cosmo_dic['Pdeltai'] = interpolate.interp2d(zs_base, ks_base,
                                                         pdi.T,
                                                         fill_value=0.0)
        mock_cosmo_dic['Pgi_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgi_phot.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgi_spec'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgi_spec.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgg_spec'] = interpolate.LinearNDInterpolator(spec_zkm,
                                                                      pgg_spec)

        fid_H_arr = np.load(str(cur_dir) + '/test_input/spec_fid_HZ.npy')
        fid_d_A_arr = np.load(str(cur_dir) + '/test_input/spec_fid_d_A.npy')

        fid_H_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                                y=fid_H_arr,
                                                                ext=0)
        fid_dA_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                                 y=fid_d_A_arr,
                                                                 ext=0)

        fid_mock_dic = {'H0': 67.5,
                        'omch2': 0.122,
                        'ombh2': 0.022,
                        'omnuh2': 0.00028,
                        'omkh2': 0.0,
                        'w': -1.0,
                        'mnu': 0.06,
                        'tau': 0.07,
                        'nnu': 3.046,
                        'ns': 0.9674,
                        'As': 2.1e-9,
                        'c': const.c.to('km/s').value,
                        'd_z_func': fid_dA_interp,
                        'H_z_func': fid_H_interp,
                        'z_win': np.linspace(0.0, 4.0, 100),
                        'k_win': np.linspace(0.001, 10.0, 100),
                        'MG_sigma': MG_interp}

        self.fiducial_dict = fid_mock_dic
        self.test_dict = mock_cosmo_dic

        self.spec = Spec(self.test_dict,
                         self.fiducial_dict)

        self.check_multipole_spectra_m0 = 12275.473017
        self.check_multipole_spectra_m1 = 2.748343
        self.check_multipole_spectra_m2 = 8408.473137
        self.check_multipole_spectra_m3 = 5.481381
        self.check_multipole_spectra_m4 = 680.137016
        self.check_multipole_spectra_integrand = 3338.69918
        self.check_scaling_factor_perp = 1.007444
        self.check_scaling_factor_parall = 1.007426
        self.check_get_k = 0.000993
        self.check_get_mu = 1.00

    def tearDown(self):
        self.check_scaling_factor_perp = None
        self.check_scaling_factor_parall = None
        self.check_get_k = None
        self.check_get_mu = None

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
