"""UNIT TESTS FOR PHOTO

This module contains unit tests for the Photo sub-module of the
photometric survey module.
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import interpolate
from likelihood.photometric_survey import photo
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


class photoinitTestCase(TestCase):

    def setUp(self):
        cur_dir = Path(__file__).resolve().parents[0]
        cmov_file = np.loadtxt(str(cur_dir) +
                               '/test_inputs/ComDist-LCDM-Lin-zNLA.dat')
        zs_r = cmov_file[:, 0]
        rs = cmov_file[:, 1]
        ang_dists = rs / (1.0 + zs_r)

        rz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r, y=rs,
                                                             ext=0)
        dz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r,
                                                             y=ang_dists,
                                                             ext=0)

        Hz_file = np.loadtxt(str(cur_dir) + '/test_inputs/Hz.dat')
        zs_H = Hz_file[:, 0]
        Hs = Hz_file[:, 1]
        Hs_mpc = Hz_file[:, 1] / const.c.to('km/s').value

        Hz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H, y=Hs,
                                                             ext=0)

        Hmpc_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                               y=Hs_mpc,
                                                               ext=0)

        f_sig_8_arr = np.load(str(cur_dir) +
                              '/test_inputs/f_sig_8_arr.npy',
                              allow_pickle=True)
        sig_8_arr = np.load(str(cur_dir) +
                            '/test_inputs/sig_8_arr.npy',
                            allow_pickle=True)

        sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                       x=np.linspace(0.0, 5.0, 50),
                       y=sig_8_arr[::-1], ext=0)
        f_sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                         x=np.linspace(0.0, 5.0, 50),
                         y=f_sig_8_arr[::-1], ext=0)

        MG_interp = mock_MG_func

        pdd = np.load(str(cur_dir) + '/test_inputs/pdd.npy')
        pdi = np.load(str(cur_dir) + '/test_inputs/pdi.npy')
        pgd = np.load(str(cur_dir) + '/test_inputs/pgd.npy')
        pgg = np.load(str(cur_dir) + '/test_inputs/pgg.npy')
        pgi_phot = np.load(str(cur_dir) + '/test_inputs/pgi_phot.npy')
        pgi_spec = np.load(str(cur_dir) + '/test_inputs/pgi_spec.npy')
        pii = np.load(str(cur_dir) + '/test_inputs/pii.npy')

        zs_base = np.linspace(0.0, 4.0, 100)
        ks_base = np.logspace(0.001, 100.0, 100)

        mock_cosmo_dic = {'ombh2': 0.022, 'omch2': 0.12, 'H0': 68.0,
                          'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046, 'ns': 0.9674,
                          'As': 2.1e-9, 'sigma8_z_func': sig_8_interp,
                          'fsigma8_z_func': f_sig_8_interp,
                          'r_z_func': rz_interp, 'd_z_func': dz_interp,
                          'H_z_func_Mpc': Hmpc_interp,
                          'H_z_func': Hz_interp,
                          'z_win': np.linspace(0.0, 4.0, 100),
                          'k_win': np.linspace(0.001, 10.0, 100),
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value}

        mock_cosmo_dic['omnuh2'] = \
            mock_cosmo_dic['mnu'] / 94.07 * (1. / 3)**0.75
        # MM: precomputed parameters
        mock_cosmo_dic['H0_Mpc'] = \
            mock_cosmo_dic['H0'] / const.c.to('km/s').value
        mock_cosmo_dic['Omb'] = \
            mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.)**2.
        mock_cosmo_dic['Omc'] = \
            mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.)**2.
        mock_cosmo_dic['Omnu'] = \
            mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.)**2.
        mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                                 mock_cosmo_dic['Omc'] +
                                 mock_cosmo_dic['Omb'])

        # (SJ): by setting below to zero, obtain previous non-IA results
        # mock_cosmo_dic['nuisance_parameters']['aia'] = 0
        # mock_cosmo_dic['nuisance_parameters']['bia'] = 0
        # mock_cosmo_dic['nuisance_parameters']['nia'] = 0

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

        nz_dic_WL = np.load(str(cur_dir) +
                            '/test_inputs/nz_dict_WL.npy',
                            allow_pickle=True).item()
        nz_dic_GC = np.load(str(cur_dir) +
                            '/test_inputs/nz_dict_GC_phot.npy',
                            allow_pickle=True).item()
        self.cosmo_dict = mock_cosmo_dic
        self.integrand_check = -0.948932
        self.wbincheck = 7.294055e-06
        self.H0 = 67.0
        self.c = const.c.to('km/s').value
        self.omch2 = 0.12
        self.ombh2 = 0.022
        self.phot = photo.Photo(mock_cosmo_dic,
                                nz_dic_WL, nz_dic_GC)
        self.W_i_Gcheck = 5.241556e-09
        self.W_IA_check = 0.0001049580
        self.cl_integrand_check = 0.000718
        self.cl_WL_check = -5.589712e-13
        self.cl_GC_check = 3.444883e-12
        self.cl_cross_check = 2.531334e-11
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

    # (SJ): wab here refers to the product of the two window functions.
    def test_power_exception(self):
        pow = float("NaN")
        wab = 1.0 * 2.0
        pandw = wab * np.atleast_1d(pow)[0]
        npt.assert_raises(Exception, self.phot.Cl_generic_integrand,
                          10.0, pandw)

    def test_cl_WL(self):
        cl_int = self.phot.Cl_WL(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_WL_check, rtol=1e-05,
                            err_msg='Cl WL test failed')

    def test_cl_GC(self):
        cl_int = self.phot.Cl_GC_phot(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_GC_check, rtol=1e-05,
                            err_msg='Cl GC photometric test failed')

    def test_cl_cross(self):
        cl_int = self.phot.Cl_cross(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_cross_check, rtol=1e-05,
                            err_msg='Cl photometric cross test failed')
