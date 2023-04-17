"""UNIT TESTS FOR PHOTO

This module contains unit tests for the Photo sub-module of the
photometric survey module.

"""

from unittest import TestCase
from unittest.mock import patch
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from cloe.photometric_survey import photo
from cloe.tests.test_tools import test_data_handler as tdh


class photoinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        mock_cosmo_dic = tdh.load_test_pickle('phot_test_dic.pickle')
        nz_dic_WL = tdh.load_test_npy('nz_dict_WL.npy').item()
        nz_dic_GC = tdh.load_test_npy('nz_dict_GC_phot.npy').item()

        ells_WL = np.arange(2, 1000, 1)
        ells_XC = np.arange(2, 2000, 1)
        ells_GC_phot = np.arange(2, 3000, 1)

        cls.nz_dic_WL = nz_dic_WL
        cls.nz_dic_GC = nz_dic_GC
        cls.flatnz = mock_cosmo_dic['Flat_nz']

        cls.phot = photo.Photo(
            None,
            nz_dic_WL,
            nz_dic_GC,
            add_RSD=False,
        )
        cls.phot.set_prefactor(
            ells_WL=ells_WL,
            ells_XC=ells_XC,
            ells_GC_phot=ells_GC_phot,
        )
        cls.phot.update(mock_cosmo_dic)

        cls.phot_rsd = photo.Photo(None, nz_dic_WL, nz_dic_GC, add_RSD=True)
        cls.phot_rsd.set_prefactor(
            ells_WL=ells_WL,
            ells_XC=ells_XC,
            ells_GC_phot=ells_GC_phot,
        )
        cls.phot_rsd.update(mock_cosmo_dic)

        # Photo module with linear interpolated magnification bias
        cls.phot_magb1 = photo.Photo(None, nz_dic_WL, nz_dic_GC,
                                     add_RSD=False)
        mock_cosmo_dic_magb1 = deepcopy(mock_cosmo_dic)
        mock_cosmo_dic_magb1['magbias_model'] = 1
        cls.phot_magb1.update(mock_cosmo_dic_magb1)

    def setUp(self) -> None:
        self.win_tol = 1e-03
        self.cl_tol = 1e-03
        self.xi_tol = 1e-03
        self.integrand_check = -0.948932
        self.wbincheck = 1.102535e-06
        self.wbincheck_mag = 0.0

        self.test_prefactor_rtol = 1e-04
        self.test_prefactor_input_ells_WL = range(2, 4)
        self.test_prefactor_input_ells_XC = range(2, 5)
        self.test_prefactor_input_ells_GC_phot = range(2, 6)
        self.test_prefactor_num_check = 7
        self.test_prefactor_len_check = {}
        self.test_prefactor_len_check['shearIA_WL'] = len(
            self.test_prefactor_input_ells_WL
        )
        self.test_prefactor_len_check['shearIA_XC'] = len(
            self.test_prefactor_input_ells_XC
        )
        self.test_prefactor_len_check['mag_XC'] = len(
            self.test_prefactor_input_ells_XC
        )
        self.test_prefactor_len_check['mag_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot
        )
        self.test_prefactor_len_check['L0_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot
        )
        self.test_prefactor_len_check['Lplus1_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot
        )
        self.test_prefactor_len_check['Lminus1_GCphot'] = len(
            self.test_prefactor_input_ells_GC_phot
        )
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
        self.W_i_GRSDcheck = np.array([
            [[8.623982e-07], [1.392808e-09], [5.172771e-10]],
            [[1.840731e-10], [1.840731e-10], [1.840731e-10]],
            [[2.069768e-15], [2.141887e-11], [6.343029e-11]]
        ])
        # the following array corresponds to the RSD kernel at the 500th
        # position of the z_winterp array
        self.unpacked_RSD_kernel_check = (
            [-5.389817e-06, -2.110740e-07, -1.601407e-11]
        )
        self.W_IA_check = 0.0001049580
        self.cl_WL_check = 6.908876e-09
        self.cl_GC_check = 2.89485e-05
        self.cl_GC_RSD_check = 3.056115e-05
        self.cl_cross_check = 1.117403e-07
        self.cl_cross_RSD_check = 1.059756e-07
        self.prefac_shearia_check = 0.988620523  # expected value for ell=10
        self.prefac_mag_check = 0.997732426  # expected value for ell=10
        self.xi_ssp_check = [6.326380e-07, 4.395978e-07]
        self.xi_ssm_check = 1.476032e-07
        self.xi_sp_check = -3.455842e-06
        self.xi_pp_check = 0.005249

    def tearDown(self):
        self.integrand_check = None
        self.wbincheck = None
        self.wbincheck_mag = None
        self.W_i_Gcheck = None
        self.W_i_GRSDcheck = None
        self.W_IA_check = None
        self.cl_WL_check = None
        self.cl_GC_check = None
        self.cl_GC_RSD_check = None
        self.unpacked_RSD_kernel_check = None
        self.cl_cross_check = None
        self.cl_cross_RSD = None
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
        npt.assert_allclose(
            self.phot.GC_window(0.001, 1),
            self.W_i_Gcheck,
            rtol=self.win_tol,
            err_msg='GC_window failed',
        )

    def test_GC_window_RSD(self):
        npt.assert_allclose(
            self.phot.GC_window_RSD(1.0, np.array([10.0, 50.0, 100.0]), 1),
            self.W_i_GRSDcheck,
            rtol=self.win_tol,
            err_msg='GC_window_RSD failed',
        )

    def test_IA_window(self):
        npt.assert_allclose(
            self.phot.IA_window(0.1, 1),
            self.W_IA_check,
            rtol=self.win_tol,
            err_msg='IA_window failed',
        )

    def test_w_integrand(self):
        int_comp = self.phot.window_integrand([0.1], 0.2, self.flatnz)
        npt.assert_allclose(
            int_comp,
            self.integrand_check,
            rtol=self.win_tol,
            err_msg='Integrand of WL kernel failed',
        )

    def test_WL_window(self):
        int_comp = self.phot.WL_window(self.phot.z_winterp, 1)[10]
        npt.assert_allclose(
            int_comp,
            self.wbincheck,
            rtol=self.win_tol,
            err_msg='WL_window failed',
        )

    def test_magnification_window(self):
        int_comp = self.phot.magnification_window(self.phot.z_winterp, 1)[10]
        npt.assert_allclose(
            int_comp,
            self.wbincheck_mag,
            rtol=self.win_tol,
            err_msg='magnification_window failed',
        )

    def test_magnification_window_1(self):
        # linear interpolator case
        int_comp =\
            self.phot_magb1.magnification_window(self.phot.z_winterp, 1)[10]
        npt.assert_allclose(
            int_comp,
            self.wbincheck_mag,
            rtol=self.win_tol,
            err_msg='magnification_window failed',
        )
        magbias_type = str(type(self.phot_magb1.magbias))
        npt.assert_string_equal(
            magbias_type,
            "<class 'scipy.interpolate._interpolate.interp1d'>")

    def test_WL_window_slow(self):
        int_comp = self.phot.WL_window_slow(
            z=self.phot.z_winterp[10],
            bin_i=1,
            k=0.1,
        )
        npt.assert_allclose(
            int_comp,
            self.wbincheck,
            rtol=self.win_tol,
            err_msg='WL_window_slow failed',
        )

    # wab here refers to the product of the two window functions.
    def test_power_exception(self):
        pow_ = float("NaN")
        wab = 1.0 * 2.0
        pandw = wab * np.atleast_1d(pow_)[0]
        npt.assert_raises(
            Exception,
            self.phot.Cl_generic_integrand,
            10.0,
            pandw,
        )

    def test_cl_WL(self):
        cl_int = self.phot.Cl_WL(10.0, 1, 1)
        npt.assert_allclose(
            cl_int,
            self.cl_WL_check,
            rtol=self.cl_tol,
            err_msg='Cl WL test failed',
        )

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
        npt.assert_equal(
            shearia_mock.call_count,
            1,
            err_msg=(
                'unexpected number of calls of '
                '_eval_prefactor_shearia():'
                f' {shearia_mock.call_count} instead of 1'
            ),
        )
        npt.assert_equal(
            mag_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                f'_eval_prefactor_mag(): {mag_mock.call_count}' +
                ' instead of 0'
            ),
        )
        # pass a value of ell with precomputed prefactor
        shearia_mock.reset_mock()
        mag_mock.reset_mock()
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_WL(2, 1, 1)
        npt.assert_equal(
            shearia_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                '_eval_prefactor_shearia():' +
                f' {shearia_mock.call_count} instead of 0'
            ),
        )
        npt.assert_equal(
            mag_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                f'_eval_prefactor_mag(): {mag_mock.call_count}' +
                ' instead of 0'
            ),
        )

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
        npt.assert_equal(
            shearia_mock.call_count,
            1,
            err_msg=(
                'unexpected number of calls of ' +
                '_eval_prefactor_shearia():' +
                f' {shearia_mock.call_count} instead of 1'
            ),
        )
        npt.assert_equal(
            mag_mock.call_count,
            1,
            err_msg=(
                'unexpected number of calls of ' +
                f'_eval_prefactor_mag(): {mag_mock.call_count}' +
                ' instead of 1'
            ),
        )
        # pass a value of ell with precomputed prefactor
        shearia_mock.reset_mock()
        mag_mock.reset_mock()
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_cross(2, 1, 1)
        npt.assert_equal(
            shearia_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                '_eval_prefactor_shearia():' +
                f' {shearia_mock.call_count} instead of 0'
            ),
        )
        npt.assert_equal(
            mag_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                f'_eval_prefactor_mag(): {mag_mock.call_count}' +
                ' instead of 0'
            ),
        )

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
        npt.assert_equal(
            shearia_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                '_eval_prefactor_shearia():' +
                f' {shearia_mock.call_count} instead of 0'
            ),
        )
        npt.assert_equal(
            mag_mock.call_count,
            1,
            err_msg=(
                'unexpected number of calls of ' +
                f'_eval_prefactor_mag(): {mag_mock.call_count}' +
                ' instead of 1'
            ),
        )
        # pass a value of ell with precomputed prefactor
        shearia_mock.reset_mock()
        mag_mock.reset_mock()
        shearia_mock.return_value = 1
        mag_mock.return_value = 1
        self.phot.Cl_GC_phot(2, 1, 1)
        npt.assert_equal(
            shearia_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                '_eval_prefactor_shearia():' +
                f' {shearia_mock.call_count} instead of 0'
            ),
        )
        npt.assert_equal(
            mag_mock.call_count,
            0,
            err_msg=(
                'unexpected number of calls of ' +
                f'_eval_prefactor_mag(): {mag_mock.call_count}' +
                f' instead of 0'
            ),
        )

    def test_unpack_RSD_kernel(self):
        rsd_kern = self.phot_rsd._unpack_RSD_kernel(10.0, 1)
        npt.assert_equal(
            rsd_kern.shape,
            (1, 1000),
            err_msg='unpack RSD kernel failed',
        )
        rsd_kern = self.phot_rsd._unpack_RSD_kernel(10.0, 1, 2, 3)
        npt.assert_equal(
            rsd_kern.shape,
            (3, 1000),
            err_msg='unpack RSD kernel failed',
        )
        npt.assert_allclose(
            rsd_kern[:, 500],
            self.unpacked_RSD_kernel_check,
            rtol=self.cl_tol,
            err_msg='unpack RSD kernel failed',
        )

    def test_cl_GC(self):
        cl_int = self.phot.Cl_GC_phot(10.0, 1, 1)
        npt.assert_allclose(
            cl_int,
            self.cl_GC_check,
            rtol=self.cl_tol,
            err_msg='Cl GC photometric test failed',
        )

    def test_cl_GC_RSD(self):
        cl_int = self.phot_rsd.Cl_GC_phot(10.0, 1, 1)
        npt.assert_allclose(
            cl_int,
            self.cl_GC_RSD_check,
            rtol=self.cl_tol,
            err_msg='Cl GC RSD photometric test failed',
        )

    def test_cl_cross(self):
        cl_int = self.phot.Cl_cross(10.0, 1, 1)
        npt.assert_allclose(
            cl_int,
            self.cl_cross_check,
            rtol=self.cl_tol,
            err_msg='Cl XC cross test failed',
        )

    def test_cl_cross_RSD(self):
        cl_int = self.phot_rsd.Cl_cross(10.0, 1, 1)
        npt.assert_allclose(
            cl_int,
            self.cl_cross_RSD_check,
            rtol=self.cl_tol,
            err_msg='Cl XC cross RSD test failed',
        )

    def test_eval_prefactor_shearia(self):
        prefac = self.phot._eval_prefactor_shearia(10.0)
        npt.assert_allclose(
            prefac,
            self.prefac_shearia_check,
            rtol=self.cl_tol,
            err_msg='_eval_prefactor_shearia() test failed',
        )

    def test_eval_prefactor_mag(self):
        prefac = self.phot._eval_prefactor_mag(10.0)
        npt.assert_allclose(
            prefac,
            self.prefac_mag_check,
            rtol=self.cl_tol,
            err_msg='_eval_prefactor_mag() test failed',
        )

    def test_set_prefactor(self):
        self.phot.set_prefactor(
            ells_WL=self.test_prefactor_input_ells_WL,
            ells_XC=self.test_prefactor_input_ells_XC,
            ells_GC_phot=self.test_prefactor_input_ells_GC_phot,
        )

        prefac_types = set(key[0] for key in self.phot._prefactor_dict.keys())
        input_ell_val = self.test_prefactor_input_ell_val

        # test that the prefactors exist for seven quantities: shearIA_WL,
        # shearIA_XC, mag_XC, mag_GCphot, L0_GCphot, Lplus1_GCphot and
        # Lminus1GCphot
        npt.assert_equal(
            len(prefac_types),
            self.test_prefactor_num_check,
            err_msg=(
                'unexpected number of prefactor types,' +
                f' {len(prefac_types)} instead of' +
                f' {self.test_prefactor_num_check}: {prefac_types}'
            ),
        )

        # test that, for each prefactor type, the prefactor is evaluated
        # for the correct number of ells, and that the calculation is correct
        # for one specific ell value
        for prefac in prefac_types:
            num_entries = len([
                key[0] for key in self.phot._prefactor_dict.keys()
                if key[0] == prefac
            ])
            npt.assert_equal(
                num_entries,
                self.test_prefactor_len_check[prefac],
                err_msg=(
                    'unexpected number of entries for' +
                    f' prefactor of type {prefac}: {num_entries}' +
                    f' instead of' +
                    f' {self.test_prefactor_len_check[prefac]}'
                ),
            )
            npt.assert_allclose(
                self.phot._prefactor_dict[prefac, input_ell_val],
                self.test_prefactor_val_check[prefac],
                rtol=self.test_prefactor_rtol,
                err_msg='Unexpected value of prefactor for type={prefac},'
                f' ell={input_ell_val}:'
                f' {self.phot._prefactor_dict[prefac, input_ell_val]} instead'
                f' of {self.test_prefactor_val_check[prefac]}'
            )

    def test_f_K_z_func_is_None(self):
        temp_cosmo_dic = self.phot.theory.copy()
        temp_cosmo_dic['f_K_z_func'] = None
        npt.assert_raises(
            KeyError,
            photo.Photo,
            temp_cosmo_dic,
            self.nz_dic_WL,
            self.nz_dic_GC,
        )

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
        xi_ssp = self.phot.corr_func_3x2pt(
            'Shear-Shear_plus',
            [1.0, 1.5],
            1,
            1,
        )
        npt.assert_allclose(
            xi_ssp,
            self.xi_ssp_check,
            rtol=self.xi_tol,
            err_msg='CF Shear-Shear plus test failed',
        )

    def test_corr_func_ssm(self):
        xi_ssm = self.phot.corr_func_3x2pt('Shear-Shear_minus', 1.0, 1, 1)
        npt.assert_allclose(
            xi_ssm,
            self.xi_ssm_check,
            rtol=self.xi_tol,
            err_msg='CF Shear-Shear minus test failed',
        )

    def test_corr_func_sp(self):
        xi_sp = self.phot.corr_func_3x2pt('Shear-Position', 1.0, 1, 1)
        npt.assert_allclose(
            xi_sp,
            self.xi_sp_check,
            rtol=self.xi_tol,
            err_msg='CF Shear-Position test failed',
        )

    def test_corr_func_pp(self):
        xi_pp = self.phot.corr_func_3x2pt('Position-Position', 1.0, 1, 1)
        npt.assert_allclose(
            xi_pp,
            self.xi_pp_check,
            rtol=self.xi_tol,
            err_msg='CF Position-Position plus test failed',
        )

    def test_corr_func_invalid_obs(self):
        npt.assert_raises(
            ValueError,
            self.phot.corr_func_3x2pt,
            'Invalid string',
            1.0,
            1,
            1,
        )

    def test_corr_func_invalid_type(self):
        npt.assert_raises(
            TypeError,
            self.phot.corr_func_3x2pt,
            'Shear-Shear_plus',
            (1.0, 1.5),
            1,
            1,
        )

    def test_z_minus1(self):
        ell_test = 2
        r_test = 100.
        r_check = r_test / 5  # (2 ell - 3) / (2 ell + 1)
        z_of_r_check = self.phot.theory['z_r_func'](r_check)
        z_out = self.phot.z_minus1(ell=ell_test, r=r_test)
        npt.assert_allclose(
            z_out,
            z_of_r_check,
            rtol=1e-3,
            err_msg='z_minus1 test failed',
        )

    def test_z_plus1(self):
        ell_test = 2
        r_test = 100.
        r_check = 9 * r_test / 5  # (2 ell + 5) / (2 ell + 1)
        z_of_r_check = self.phot.theory['z_r_func'](r_check)
        z_out = self.phot.z_plus1(ell=ell_test, r=r_test)
        npt.assert_allclose(
            z_out,
            z_of_r_check,
            rtol=1e-3,
            err_msg='z_plus1 test failed',
        )

    def test_eval_prefactor_l_0(self):
        npt.assert_allclose(
            self.phot._eval_prefactor_l_0(ell=1),
            3 / 5, rtol=1e-6,
            err_msg='L0 prefactor test failed',
        )

    def test_eval_prefactor_l_minus1(self):
        npt.assert_allclose(
            self.phot._eval_prefactor_l_minus1(ell=2),
            -2 / (3 * np.sqrt(5)),
            rtol=1e-6,
            err_msg='L-1 prefactor test failed',
        )

    def test_eval_prefactor_l_plus1(self):
        npt.assert_allclose(
            self.phot._eval_prefactor_l_plus1(ell=2),
            -12 / (7 * np.sqrt(45)),
            rtol=1e-6,
            err_msg='Lplus1 prefactor test failed'
        )
