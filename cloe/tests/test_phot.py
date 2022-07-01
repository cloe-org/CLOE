"""UNIT TESTS FOR PHOTO

This module contains unit tests for the Photo sub-module of the
photometric survey module.
=======

"""

from unittest import TestCase
from unittest.mock import patch
import numpy as np
import numpy.testing as npt
from scipy import interpolate
from cloe.photometric_survey import photo
from astropy import constants as const
from pathlib import Path


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


class photoinitTestCase(TestCase):

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
        zr_interp = interpolate.InterpolatedUnivariateSpline(x=rs, y=zs_r,
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

        sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                       x=np.linspace(0.0, 5.0, 50),
                       y=sig_8_arr[::-1], ext=0)
        f_sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                         x=np.linspace(0.0, 5.0, 50),
                         y=f_sig_8_arr[::-1], ext=0)

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
                          'r_z_func': rz_interp,
                          'z_r_func': zr_interp,
                          'd_z_func': dz_interp,
                          # pretend that f_K_z_func behaves as r_z_func.
                          # This is not realistic but it is fine for the
                          # purposes of the unit tests
                          'f_K_z_func': rz_interp,
                          'H_z_func_Mpc': Hmpc_interp,
                          'H_z_func': Hz_interp,
                          'z_win': np.linspace(0.0, 4.0, 100),
                          'k_win': np.linspace(0.001, 10.0, 100),
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value,
                          'nuisance_parameters': {}}

        # precomputed parameters
        mock_cosmo_dic['H0_Mpc'] = \
            mock_cosmo_dic['H0'] / const.c.to('km/s').value
        mock_cosmo_dic['Omb'] = \
            mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.0)**2.0
        mock_cosmo_dic['Omc'] = \
            mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.0)**2.0
        mock_cosmo_dic['Omnu'] = \
            mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.0)**2.0
        mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                                 mock_cosmo_dic['Omc'] +
                                 mock_cosmo_dic['Omb'])

        nuisance_dic = mock_cosmo_dic['nuisance_parameters']
        # by setting below to zero, obtain previous non-IA results
        # mock_cosmo_dic['nuisance_parameters']['aia'] = 0
        # mock_cosmo_dic['nuisance_parameters']['bia'] = 0
        # mock_cosmo_dic['nuisance_parameters']['nia'] = 0
        for i in range(10):
            nuisance_dic[f'dz_{i+1}_GCphot'] = 0.0
            nuisance_dic[f'dz_{i+1}_WL'] = 0.0
            nuisance_dic[f'multiplicative_bias_{i+1}'] = 0.0
            nuisance_dic[f'magnification_bias_{i+1}'] = 0.0
        mock_cosmo_dic['Pmm_phot'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pdd, kx=1, ky=1)
        mock_cosmo_dic['Pgg_phot'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pgg, kx=1, ky=1)
        mock_cosmo_dic['Pgdelta_phot'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pgd, kx=1, ky=1)
        mock_cosmo_dic['Pii'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pii, kx=1, ky=1)
        mock_cosmo_dic['Pdeltai'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pdi, kx=1, ky=1)
        mock_cosmo_dic['Pgi_phot'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pgi_phot, kx=1, ky=1)
        mock_cosmo_dic['Pgi_spectro'] = \
            interpolate.RectBivariateSpline(zs_base, ks_base,
                                            pgi_spectro, kx=1, ky=1)
        # temporary fix, see #767
        mock_cosmo_dic['CAMBdata'] = mock_CAMB_data(rz_interp)

        nz_dic_WL = np.load(str(cur_dir) +
                            '/test_input/nz_dict_WL.npy',
                            allow_pickle=True).item()
        nz_dic_GC = np.load(str(cur_dir) +
                            '/test_input/nz_dict_GC_phot.npy',
                            allow_pickle=True).item()
        cls.phot = photo.Photo(mock_cosmo_dic, nz_dic_WL, nz_dic_GC)
        cls.flatnz = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 4.6, 20), np.ones(20), ext=2)
        cls.nz_dic_WL = nz_dic_WL
        cls.nz_dic_GC = nz_dic_GC

        ells_WL = range(2, 1000)
        ells_XC = range(2, 2000)
        ells_GC_phot = range(2, 3000)
        cls.phot.set_prefactor(ells_WL=ells_WL,
                               ells_XC=ells_XC,
                               ells_GC_phot=ells_GC_phot)

    def setUp(self) -> None:
        self.win_tol = 1e-03
        self.cl_tol = 1e-03
        self.xi_tol = 1e-03
        self.integrand_check = 1.043825
        self.wbincheck = -1.47437e-06
        self.wbincheck_mag = 0.0

        self.test_prefactor_rtol = 1e-04
        self.test_prefactor_input_ells_WL = range(2, 4)
        self.test_prefactor_input_ells_XC = range(2, 5)
        self.test_prefactor_input_ells_GC_phot = range(2, 6)
        self.test_prefactor_num_check = 7
        self.test_prefactor_len_check = {}
        self.test_prefactor_len_check['shearIA_WL'] = len(
            self.test_prefactor_input_ells_WL)
        self.test_prefactor_len_check['shearIA_XC'] = len(
            self.test_prefactor_input_ells_XC)
        self.test_prefactor_len_check['mag_XC'] = len(
            self.test_prefactor_input_ells_XC)
        self.test_prefactor_len_check['mag_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot)
        self.test_prefactor_len_check['L0_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot)
        self.test_prefactor_len_check['Lplus1_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot)
        self.test_prefactor_len_check['Lminus1_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot)
        # the following prefactors are evaluated for input ell = 3
        self.test_prefactor_input_ell_val = 3
        self.test_prefactor_val_check = {}
        self.test_prefactor_val_check['shearIA_WL'] = 0.799666805
        self.test_prefactor_val_check['shearIA_XC'] = 0.894240910
        self.test_prefactor_val_check['mag_XC'] = 0.979591837
        self.test_prefactor_val_check['mag_GCphot'] = 0.979591837
        self.test_prefactor_val_check['L0_GCphot'] = 0.51111111111
        self.test_prefactor_val_check['Lminus1_GCphot'] = -0.26186146828
        self.test_prefactor_val_check['Lplus1_GCphot'] = -0.253245725465

        self.W_i_Gcheck = 5.241556e-09
        self.W_IA_check = 0.0001049580
        self.cl_WL_check = 2.572589e-08
        self.cl_GC_check = 2.89485e-05
        self.cl_cross_check = -6.379221e-07
        self.prefac_shearia_check = 0.988620523  # expected value for ell=10
        self.prefac_mag_check = 0.997732426  # expected value for ell=10
        self.xi_ssp_check = [2.908933e-06, 1.883681e-06]
        self.xi_ssm_check = 8.993205e-07
        self.xi_sp_check = -6.925195e-05
        self.xi_pp_check = 0.005249

    def tearDown(self):
        self.integrand_check = None
        self.wbincheck = None
        self.wbincheck_mag = None
        self.W_i_Gcheck = None
        self.W_IA_check = None
        self.cl_WL_check = None
        self.cl_GC_check = None
        self.cl_cross_check = None
        self.xi_ssp_check = None
        self.xi_ssm_check = None
        self.xi_sp_check = None
        self.xi_pp_check = None
        self.prefac_shearia_check = None
        self.prefac_mag_check = None
        self.test_prefactor_rtol = None
        self.test_prefactor_input_ells_WL = None
        self.test_prefactor_input_ells_XC = None
        self.test_prefactor_input_ells_GC_phot = None
        self.test_prefactor_num_check = None
        self.test_prefactor_len_check = None
        self.test_prefactor_input_ell_val = None
        self.test_prefactor_val_check = None

    def test_GC_window(self):
        npt.assert_allclose(self.phot.GC_window(0.001, 1),
                            self.W_i_Gcheck, rtol=self.win_tol,
                            err_msg='GC_window failed')

    def test_IA_window(self):
        npt.assert_allclose(self.phot.IA_window(0.1, 1),
                            self.W_IA_check, rtol=self.win_tol,
                            err_msg='IA_window failed')

    def test_w_integrand(self):
        int_comp = self.phot.window_integrand([0.1], 0.2, self.flatnz)
        npt.assert_allclose(int_comp, self.integrand_check, rtol=self.win_tol,
                            err_msg='Integrand of WL kernel failed')

    def test_WL_window(self):
        int_comp = self.phot.WL_window(self.phot.z_winterp, 1)[10]
        npt.assert_allclose(int_comp, self.wbincheck, rtol=self.win_tol,
                            err_msg='WL_window failed')

    def test_magnification_window(self):
        int_comp = self.phot.magnification_window(self.phot.z_winterp, 1)[10]
        npt.assert_allclose(int_comp, self.wbincheck_mag, rtol=self.win_tol,
                            err_msg='magnification_window failed')

    def test_WL_window_slow(self):
        int_comp = self.phot.WL_window_slow(
            z=self.phot.z_winterp[10], bin_i=1, k=0.1)
        npt.assert_allclose(int_comp, self.wbincheck, rtol=self.win_tol,
                            err_msg='WL_window_slow failed')

    # wab here refers to the product of the two window functions.
    def test_power_exception(self):
        pow_ = float("NaN")
        wab = 1.0 * 2.0
        pandw = wab * np.atleast_1d(pow_)[0]
        npt.assert_raises(Exception, self.phot.Cl_generic_integrand,
                          10.0, pandw)

    def test_cl_WL(self):
        cl_int = self.phot.Cl_WL(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_WL_check, rtol=self.cl_tol,
                            err_msg='Cl WL test failed')

    # verify that _eval_prefactor_shearia() is called when the input ell is not
    # available in the precomputed dictionary, and that it is not called if
    # instead the input ell is available in the precomputed dictionary.
    # In any case, _eval_prefactor_mag() should never be called within Cl_WL()
    @patch('cloe.photometric_survey.photo.Photo._eval_prefactor_mag')
    @patch('cloe.photometric_survey.photo.Photo._eval_prefactor_shearia')
    def test_Cl_WL_precomputed(self, shearia_mock, mag_mock):
        # pass a value of ell with non-precomputed prefactor
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_WL(2.5, 1, 1)
        npt.assert_equal(shearia_mock.call_count, 1,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_shearia():'
                         f' {shearia_mock.call_count} instead of 1')
        npt.assert_equal(mag_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_mag(): {mag_mock.call_count}'
                         f' instead of 0')
        # pass a value of ell with precomputed prefactor
        shearia_mock.reset_mock()
        mag_mock.reset_mock()
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_WL(2, 1, 1)
        npt.assert_equal(shearia_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_shearia():'
                         f' {shearia_mock.call_count} instead of 0')
        npt.assert_equal(mag_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_mag(): {mag_mock.call_count}'
                         f' instead of 0')

    # verify that _eval_prefactor_shearia() and _eval_prefactor_mag() are
    # called the expected number of times when the input ell is not available
    # in the precomputed dictionary, and that they are not called if instead
    # the input ell is available in the precomputed dictionary.
    @patch('cloe.photometric_survey.photo.Photo._eval_prefactor_mag')
    @patch('cloe.photometric_survey.photo.Photo._eval_prefactor_shearia')
    def test_Cl_cross_precomputed(self, shearia_mock, mag_mock):
        # pass a value of ell with non-precomputed prefactor
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_cross(2.5, 1, 1)
        npt.assert_equal(shearia_mock.call_count, 1,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_shearia():'
                         f' {shearia_mock.call_count} instead of 1')
        npt.assert_equal(mag_mock.call_count, 1,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_mag(): {mag_mock.call_count}'
                         f' instead of 1')
        # pass a value of ell with precomputed prefactor
        shearia_mock.reset_mock()
        mag_mock.reset_mock()
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_cross(2, 1, 1)
        npt.assert_equal(shearia_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_shearia():'
                         f' {shearia_mock.call_count} instead of 0')
        npt.assert_equal(mag_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_mag(): {mag_mock.call_count}'
                         f' instead of 0')

    # verify that _eval_prefactor_mag() is called the expected number of
    # times when the input ell is not available in the precomputed dictionary,
    # and that it is not called if instead the input ell is available in the
    # precomputed dictionary.
    # In any case, _eval_prefactor_shearia() should never be called within
    # Cl_GC_phot()
    @patch('cloe.photometric_survey.photo.Photo._eval_prefactor_mag')
    @patch('cloe.photometric_survey.photo.Photo._eval_prefactor_shearia')
    def test_Cl_GC_phot_precomputed(self, shearia_mock, mag_mock):
        # pass a value of ell with non-precomputed prefactor
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_GC_phot(2.5, 1, 1)
        npt.assert_equal(shearia_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_shearia():'
                         f' {shearia_mock.call_count} instead of 0')
        npt.assert_equal(mag_mock.call_count, 1,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_mag(): {mag_mock.call_count}'
                         f' instead of 1')
        # pass a value of ell with precomputed prefactor
        shearia_mock.reset_mock()
        mag_mock.reset_mock()
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_GC_phot(2, 1, 1)
        npt.assert_equal(shearia_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_shearia():'
                         f' {shearia_mock.call_count} instead of 0')
        npt.assert_equal(mag_mock.call_count, 0,
                         err_msg=f'unexpected number of calls of '
                         f'_eval_prefactor_mag(): {mag_mock.call_count}'
                         f' instead of 0')

    def test_cl_GC(self):
        cl_int = self.phot.Cl_GC_phot(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_GC_check, rtol=self.cl_tol,
                            err_msg='Cl GC photometric test failed')

    def test_cl_cross(self):
        cl_int = self.phot.Cl_cross(10.0, 1, 1)
        npt.assert_allclose(cl_int, self.cl_cross_check,
                            rtol=self.cl_tol,
                            err_msg='Cl XC cross test failed')

    def test_eval_prefactor_shearia(self):
        prefac = self.phot._eval_prefactor_shearia(10.0)
        npt.assert_allclose(prefac, self.prefac_shearia_check,
                            rtol=self.cl_tol,
                            err_msg='_eval_prefactor_shearia() test failed')

    def test_eval_prefactor_mag(self):
        prefac = self.phot._eval_prefactor_mag(10.0)
        npt.assert_allclose(prefac, self.prefac_mag_check,
                            rtol=self.cl_tol,
                            err_msg='_eval_prefactor_mag() test failed')

    def test_set_prefactor(self):
        self.phot.set_prefactor(ells_WL=self.test_prefactor_input_ells_WL,
                                ells_XC=self.test_prefactor_input_ells_XC,
                                ells_GC_phot=(
                                    self.test_prefactor_input_ells_GC_phot))

        prefac_types = set(key[0] for key in self.phot._prefactor_dict.keys())
        input_ell_val = self.test_prefactor_input_ell_val

        # test that the prefactors exist for seven quantities: shearIA_WL,
        # shearIA_XC, mag_XC, mag_GCphot, L0_GCphot, Lplus1_GCphot and
        # Lminus1GCphot
        npt.assert_equal(len(prefac_types), self.test_prefactor_num_check,
                         err_msg=f'unexpected number of prefactor types,'
                         f' {len(prefac_types)} instead of'
                         f' {self.test_prefactor_num_check}: {prefac_types}')

        # test that, for each prefactor type, the prefactor is evaluated
        # for the correct number of ells, and that the calculation is correct
        # for one specific ell value
        for prefac in prefac_types:
            num_entries = len([key[0]
                               for key in self.phot._prefactor_dict.keys()
                               if key[0] == prefac])
            npt.assert_equal(num_entries,
                             self.test_prefactor_len_check[prefac],
                             err_msg=f'unexpected number of entries for'
                             f' prefactor of type {prefac}: {num_entries}'
                             f' instead of'
                             f' {self.test_prefactor_len_check[prefac]}')
            npt.assert_allclose(
                self.phot._prefactor_dict[prefac, input_ell_val],
                self.test_prefactor_val_check[prefac],
                rtol=self.test_prefactor_rtol,
                err_msg='Unexpected value of prefactor for type={prefac},'
                f' ell={input_ell_val}:'
                f' {self.phot._prefactor_dict[prefac, input_ell_val]} instead'
                f' of {self.test_prefactor_val_check[prefac]}')

    def test_f_K_z_func_is_None(self):
        temp_cosmo_dic = self.phot.theory.copy()
        temp_cosmo_dic['f_K_z_func'] = None
        npt.assert_raises(KeyError,
                          photo.Photo,
                          temp_cosmo_dic,
                          self.nz_dic_WL,
                          self.nz_dic_GC)

    # this function tests a temporary part of code, see #767
    # def test_CAMBdata_is_None(self):
    #    temp_cosmo_dic = self.phot.theory.copy()
    #    temp_cosmo_dic['CAMBdata'] = None
    #    npt.assert_raises(KeyError,
    #                      photo.Photo,
    #                      temp_cosmo_dic,
    #                      self.nz_dic_WL,
    #                      self.nz_dic_GC)

    def test_corr_func_ssp(self):
        xi_ssp = self.phot.corr_func_3x2pt('Shear-Shear_plus', [1.0, 1.5],
                                           1, 1)
        npt.assert_allclose(xi_ssp, self.xi_ssp_check, rtol=self.xi_tol,
                            err_msg='CF Shear-Shear plus test failed')

    def test_corr_func_ssm(self):
        xi_ssm = self.phot.corr_func_3x2pt('Shear-Shear_minus', 1.0, 1, 1)
        npt.assert_allclose(xi_ssm, self.xi_ssm_check, rtol=self.xi_tol,
                            err_msg='CF Shear-Shear minus test failed')

    def test_corr_func_sp(self):
        xi_sp = self.phot.corr_func_3x2pt('Shear-Position', 1.0, 1, 1)
        npt.assert_allclose(xi_sp, self.xi_sp_check, rtol=self.xi_tol,
                            err_msg='CF Shear-Position test failed')

    def test_corr_func_pp(self):
        xi_pp = self.phot.corr_func_3x2pt('Position-Position', 1.0, 1, 1)
        npt.assert_allclose(xi_pp, self.xi_pp_check, rtol=self.xi_tol,
                            err_msg='CF Position-Position plus test failed')

    def test_corr_func_invalid_obs(self):
        npt.assert_raises(ValueError, self.phot.corr_func_3x2pt,
                          'Invalid string', 1.0, 1, 1)

    def test_corr_func_invalid_type(self):
        npt.assert_raises(TypeError, self.phot.corr_func_3x2pt,
                          'Shear-Shear_plus', (1.0, 1.5), 1, 1)

    def test_z_minus1(self):
        ell_test = 2
        r_test = 100.
        r_check = r_test / 5  # (2 ell - 3) / (2 ell + 1)
        z_of_r_check = self.phot.theory['z_r_func'](r_check)
        z_out = self.phot.z_minus1(ell=ell_test, r=r_test)
        npt.assert_allclose(z_out, z_of_r_check, rtol=1e-3,
                            err_msg='z_minus1 test failed')

    def test_z_plus1(self):
        ell_test = 2
        r_test = 100.
        r_check = 9 * r_test / 5  # (2 ell + 5) / (2 ell + 1)
        z_of_r_check = self.phot.theory['z_r_func'](r_check)
        z_out = self.phot.z_plus1(ell=ell_test, r=r_test)
        npt.assert_allclose(z_out, z_of_r_check, rtol=1e-3,
                            err_msg='z_plus1 test failed')

    def test_eval_prefactor_l_0(self):
        npt.assert_allclose(self.phot._eval_prefactor_l_0(ell=1),
                            3 / 5, rtol=1e-6,
                            err_msg='L0 prefactor test failed')

    def test_eval_prefactor_l_minus1(self):
        npt.assert_allclose(self.phot._eval_prefactor_l_minus1(ell=2),
                            -2 / (3 * np.sqrt(5)), rtol=1e-6,
                            err_msg='L-1 prefactor test failed')

    def test_eval_prefactor_l_plus1(self):
        npt.assert_allclose(self.phot._eval_prefactor_l_plus1(ell=2),
                            -12 / (7 * np.sqrt(45)), rtol=1e-6,
                            err_msg='Lplus1 prefactor test failed')
