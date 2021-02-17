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
from likelihood.auxiliary.plotter import Plotter
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


class plotterTestCase(TestCase):

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
        pdi_phot = np.load(str(cur_dir) + '/test_input/pdi.npy')
        pgd_phot = np.load(str(cur_dir) + '/test_input/pgd.npy')
        pgg_phot = np.load(str(cur_dir) + '/test_input/pgg.npy')
        pgi_phot = np.load(str(cur_dir) + '/test_input/pgi_phot.npy')
        pgi_spec = np.load(str(cur_dir) + '/test_input/pgi_spec.npy')
        pgg_spec = np.load(str(cur_dir) + '/test_input/pgg_spec.npy')
        pii_phot = np.load(str(cur_dir) + '/test_input/pii.npy')

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
                          'z_win': zs_base,
                          'k_win': ks_base,
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value}

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
                                                          pgg_phot.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgdelta_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                              pgd_phot.T,
                                                              fill_value=0.0)
        mock_cosmo_dic['Pii'] = interpolate.interp2d(zs_base, ks_base,
                                                     pii_phot.T,
                                                     fill_value=0.0)
        mock_cosmo_dic['Pdeltai'] = interpolate.interp2d(zs_base, ks_base,
                                                         pdi_phot.T,
                                                         fill_value=0.0)
        mock_cosmo_dic['Pgi_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgi_phot.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgi_spec'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgi_spec.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgg_spec'] = interpolate.LinearNDInterpolator(spec_zkm,
                                                                      pgg_spec)

        fig1 = plt.figure()
        self.ax1 = fig1.add_subplot(1, 1, 1)

        self.plot_inst = Plotter(mock_cosmo_dic)

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

    def test_plot_GC_spec_multipole_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          0.1, np.array([0.1]), 5, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          0.1, np.array([0.1]), 2)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          0.1, np.array([10.0]), 2, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          3.0, np.array([0.1]), 2, self.ax1)

    def test_ext_GC_spec_multipole_excep(self):
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          "1.2", 5, self.ax1)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          "1.2", 2)
        npt.assert_raises(Exception, self.plot_inst.plot_GC_spec_multipole,
                          "3.0", 2, self.ax1)
