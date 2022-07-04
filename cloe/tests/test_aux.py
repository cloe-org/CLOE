"""UNIT TESTS FOR AUXILIARY

This module contains unit tests for the auxiliary functions module.
=======

"""

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy import constants as const
from unittest import TestCase
from cloe.auxiliary.plotter import Plotter
from pathlib import Path
from cloe.tests.test_input.data import mock_data


# temporary fix, see #767
class mock_CAMB_data:
    def __init__(self, rz_interp):
        self.rz_interp = rz_interp

    def angular_diameter_distance2(self, z1, z2):
        return self.rz_interp(z1) - self.rz_interp(z2)


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


class plotterTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
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

        pdd = np.load(str(cur_dir) + '/test_input/pdd.npy')
        pdi = np.load(str(cur_dir) + '/test_input/pdi.npy')
        pgd = np.load(str(cur_dir) + '/test_input/pgd.npy')
        pgg = np.load(str(cur_dir) + '/test_input/pgg.npy')
        pgi_phot = np.load(str(cur_dir) + '/test_input/pgi_phot.npy')
        pgi_spectro = np.load(str(cur_dir) + '/test_input/pgi_spectro.npy')
        pii = np.load(str(cur_dir) + '/test_input/pii.npy')

        zs_base = np.linspace(0.0, 4.0, 100)
        ks_base = np.logspace(-3.0, 1.0, 100)

        mock_cosmo_dic = {'ombh2': 0.022445, 'omch2': 0.121203, 'H0': 67.0,
                          'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046,
                          'omkh2': 0.0, 'omnuh2': 0.0, 'ns': 0.96,
                          'w': -1.0, 'sigma8_0': 0.816,
                          'As': 2.115e-9, 'sigma8_z_func': sig_8_interp,
                          'fsigma8_z_func': f_sig_8_interp,
                          'f_z': f_z_interp,
                          'r_z_func': rz_interp, 'd_z_func': dz_interp,
                          # pretend that f_K_z_func behaves as r_z_func.
                          # This is not realistic but it is fine for the
                          # purposes of the unit tests
                          'f_K_z_func': rz_interp,
                          'H_z_func_Mpc': Hmpc_interp,
                          'H_z_func': Hz_interp,
                          'z_win': zs_base,
                          'k_win': ks_base,
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value,
                          'NL_flag': 1,
                          'nuisance_parameters': {
                              'b1_photo': 1.0997727037892875,
                              'b2_photo': 1.220245876862528,
                              'b3_photo': 1.2723993083933989,
                              'b4_photo': 1.316624471897739,
                              'b5_photo': 1.35812370570578,
                              'b6_photo': 1.3998214171814918,
                              'b7_photo': 1.4446452851824907,
                              'b8_photo': 1.4964959071110084,
                              'b9_photo': 1.5652475842498528,
                              'b10_photo': 1.7429859437184225,
                              'b1_spectro': 1.4614804,
                              'b2_spectro': 1.6060949,
                              'b3_spectro': 1.7464790,
                              'b4_spectro': 1.8988660,
                              'aia': 1.72,
                              'nia': -0.41,
                              'bia': 0.0,
                              'dz_1_GCphot': 0.0, 'dz_1_WL': 0.0,
                              'dz_2_GCphot': 0.0, 'dz_2_WL': 0.0,
                              'dz_3_GCphot': 0.0, 'dz_3_WL': 0.0,
                              'dz_4_GCphot': 0.0, 'dz_4_WL': 0.0,
                              'dz_5_GCphot': 0.0, 'dz_5_WL': 0.0,
                              'dz_6_GCphot': 0.0, 'dz_6_WL': 0.0,
                              'dz_7_GCphot': 0.0, 'dz_7_WL': 0.0,
                              'dz_8_GCphot': 0.0, 'dz_8_WL': 0.0,
                              'dz_9_GCphot': 0.0, 'dz_9_WL': 0.0,
                              'dz_10_GCphot': 0.0, 'dz_10_WL': 0.0}
                          }

        nuisance_dic = mock_cosmo_dic['nuisance_parameters']
        for i in range(10):
            nuisance_dic[f'multiplicative_bias_{i+1}'] = 0
            nuisance_dic[f'magnification_bias_{i+1}'] = 0
        # precomputed parameters
        mock_cosmo_dic['H0_Mpc'] = \
            mock_cosmo_dic['H0'] / const.c.to('km/s').value
        mock_cosmo_dic['Omb'] = \
            mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.0) ** 2.0
        mock_cosmo_dic['Omc'] = \
            mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.0) ** 2.0
        mock_cosmo_dic['Omnu'] = \
            mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.0) ** 2.0
        mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                                 mock_cosmo_dic['Omc'] +
                                 mock_cosmo_dic['Omb'])
        mock_cosmo_dic['Omk'] = \
            mock_cosmo_dic['omkh2'] / (mock_cosmo_dic['H0'] / 100.0)**2.0

        p_matter = mock_P_obj(interpolate.interp2d(zs_base, ks_base, pdd.T,
                                                   fill_value=0))
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
        mock_cosmo_dic['Pgi_spectro'] = interpolate.interp2d(zs_base, ks_base,
                                                             pgi_spectro.T,
                                                             fill_value=0.0)
        # temporary fix, see #767
        mock_cosmo_dic['CAMBdata'] = mock_CAMB_data(rz_interp)

        fig1 = plt.figure()
        cls.ax1 = fig1.add_subplot(1, 1, 1)

        cls.plot_inst = Plotter(mock_cosmo_dic, mock_data)

    def tearDown(self):
        pass

    def test_plotter_init(self):
        npt.assert_raises(Exception, Plotter)

    def test_plot_phot_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_Cl_phot,
                          np.array([1.0]), 1, 1, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_Cl_phot,
                          np.array([10.0]), 1, 1)

    def test_ext_phot_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_external_Cl_phot, 11,
                          1, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_external_Cl_phot, 10,
                          1)

    def test_plot_XC_phot_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_Cl_XC, 1.0, 1, 1,
                          self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_Cl_XC, 10.0, 1, 1)

    def test_ext_XC_phot_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_external_Cl_XC,
                          11, 1, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_external_Cl_XC,
                          10, 1)

    def test_plot_GC_spectro_multipole_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          0.1, np.array([0.1]), 5, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          0.1, np.array([0.1]), 2)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          0.1, np.array([10.0]), 2, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          3.0, np.array([0.1]), 2, self.ax1)

    def test_ext_GC_spectro_multipole_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          "1.2", 5, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          "1.2", 2)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spectro_multipole,
                          "3.0", 2, self.ax1)
