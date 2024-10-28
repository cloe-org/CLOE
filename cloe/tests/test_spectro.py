"""UNIT TESTS FOR SPECTRO

This module contains unit tests for the :obj:`spectro` sub-module of the
spectroscopy survey module.

"""

from unittest.mock import patch
from unittest import TestCase
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from cloe.spectroscopic_survey.spectro import Spectro
from cloe.tests.test_tools.spectro_test_handler import SpectroTestParent
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class specinitTestCase(TestCase, SpectroTestParent):

    @classmethod
    def setUpClass(cls) -> None:

        cls.test_dict = load_test_pickle('spectro_test_dic.pickle')
        cls.test_dict_1 = deepcopy(cls.test_dict)
        cls.test_dict_1['GCsp_z_err'] = True
        cls.test_dict_2 = deepcopy(cls.test_dict)
        cls.test_dict_2['f_out'] = 1.0
        cls.test_dict_3 = deepcopy(cls.test_dict)
        cls.test_dict_3['f_out_z_dep'] = True
        cls.test_dict_3['f_out_1'] = 1.0
        cls.mixing_matrix_dict_spectro = \
            load_test_pickle('mixmat_spectro.pickle')

        cls.spectro = Spectro(cls.test_dict, ['1.', '1.2', '1.4', '1.65'],
                              cls.mixing_matrix_dict_spectro)
        cls.spectro_1 = Spectro(cls.test_dict_1, ['1.', '1.2', '1.4', '1.65'],
                                cls.mixing_matrix_dict_spectro)
        cls.spectro_2 = Spectro(cls.test_dict_2, ['1.', '1.2', '1.4', '1.65'],
                                cls.mixing_matrix_dict_spectro)
        cls.spectro_3 = Spectro(cls.test_dict_3, ['1.', '1.2', '1.4', '1.65'],
                                cls.mixing_matrix_dict_spectro)

    def setUp(self):
        self.test_dict['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        self.test_dict['noise_Pgg_spectro'] = \
            np.vectorize(self.noise_Pgg_spectro)
        self.test_dict_1['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        self.test_dict_1['noise_Pgg_spectro'] = \
            np.vectorize(self.noise_Pgg_spectro)
        self.test_dict_2['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        self.test_dict_2['noise_Pgg_spectro'] = \
            np.vectorize(self.noise_Pgg_spectro)
        self.test_dict_3['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        self.test_dict_3['noise_Pgg_spectro'] = \
            np.vectorize(self.noise_Pgg_spectro)

        self.check_multipole_spectra_m0 = 12292.776078
        self.check_multipole_spectra_m1 = 0.0
        self.check_multipole_spectra_m2 = 8409.284572
        self.check_multipole_spectra_m3 = 0.0
        self.check_multipole_spectra_m4 = 678.085174
        self.check_multipole_spectra_integrand_no_z_err = 3343.949991
        self.check_multipole_spectra_integrand = 2969.445812
        self.check_scaling_factor_perp = 1.007444
        self.check_scaling_factor_parall = 1.007426
        self.check_get_k = 0.000993
        self.check_get_mu = 1.00
        self.check_gal_redshift_scatter = 0.999923

    def tearDown(self):
        self.check_scaling_factor_perp = None
        self.check_scaling_factor_parall = None
        self.check_get_k = None
        self.check_get_mu = None
        self.check_gal_redshift_scatter = None
        self.check_multipole_spectra_integrand_no_z_err = None
        self.check_multipole_spectra_integrand = None
        self.check_multipole_spectra_m0 = None
        self.check_multipole_spectra_m1 = None
        self.check_multipole_spectra_m2 = None
        self.check_multipole_spectra_m3 = None
        self.check_multipole_spectra_m4 = None

    def istf_spectro_galbias(self, redshift, bin_edge_list=None):
        """
        Updates galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift, according to default recipe.

        Note: for redshifts above the final bin (z > 1.80), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.90), we use the bias of the first bin.

        Attention: this will change in the future

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of redshift bin edges for spectroscopic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        bi_val: float
            Value of spectroscopic galaxy bias at input redshift
        """
        if bin_edge_list is None:
            bin_edge_list = [0.90, 1.10, 1.30, 1.50, 1.80]

        nuisance_dic = self.test_dict['nuisance_parameters']
        istf_bias_list = [
            nuisance_dic['b1_spectro_bin1'],
            nuisance_dic['b1_spectro_bin2'],
            nuisance_dic['b1_spectro_bin3'],
            nuisance_dic['b1_spectro_bin4']
        ]

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for index in range(len(bin_edge_list) - 1):
                if bin_edge_list[index] <= redshift < bin_edge_list[index + 1]:
                    bi_val = istf_bias_list[index]
        elif redshift >= bin_edge_list[-1]:
            bi_val = istf_bias_list[-1]
        elif redshift < bin_edge_list[0]:
            bi_val = istf_bias_list[0]
        return bi_val

    def test_multipole_spectra_m0(self):
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[0]),
            self.check_multipole_spectra_m0,
            rtol=1e-05,
            err_msg='Multipole spectrum m = 0 failed',
        )

    def test_multipole_spectra_m1(self):
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[1]),
            self.check_multipole_spectra_m1,
            atol=1e-10,
            err_msg='Multipole spectrum m = 1 failed',
        )

    def test_multipole_spectra_m2(self):
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[2]),
            self.check_multipole_spectra_m2,
            rtol=2e-04,
            err_msg='Multipole spectrum m = 2 failed',
        )

    def test_multipole_spectra_m3(self):
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[3]),
            self.check_multipole_spectra_m3,
            atol=1e-10,
            err_msg='Multipole spectrum m = 3 failed',
        )

    def test_multipole_spectra_m4(self):
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[4]),
            self.check_multipole_spectra_m4,
            rtol=1e-05,
            err_msg='Multipole spectrum m = 4 failed',
        )

    def test_multipole_spectra_m100(self):
        npt.assert_raises(
            KeyError,
            self.spectro.multipole_spectra,
            1.0,
            0.1,
            ms=[100],
        )

    @patch('cloe.spectroscopic_survey.spectro.Spectro.multipole_spectra')
    def test_convolved_multipole_spectra(self, mock_mul_spe):
        array_shape = np.zeros((3, 98))
        convolved_mps = self.spectro.convolved_power_spectrum_multipoles(1.0)
        # test that the shape of the returned array matches the expectations
        npt.assert_equal(array_shape.shape, np.array(convolved_mps).shape)
        # test that the convolved_power_spectrum_multipoles() function
        # was called
        mock_mul_spe.assert_called()

    def test_multipole_spectra_integrand(self):
        mu_grid = np.linspace(-1, 1, 2001)
        integrand = (
            self.spectro_1.multipole_spectra_integrand(mu_grid, 1.0, 0.1, [2])
        )
        idx = np.where(mu_grid == 0.7)
        value = integrand[0][idx]
        npt.assert_allclose(
            value,
            self.check_multipole_spectra_integrand,
            rtol=1e-05,
            err_msg='Multipole spectra integrand failed',
        )

    def test_multipole_spectra_integrand_no_z_err(self):
        mu_grid = np.linspace(-1, 1, 2001)
        integrand = (
            self.spectro.multipole_spectra_integrand(mu_grid, 1.0, 0.1, [2])
        )
        idx = np.where(mu_grid == 0.7)
        value = integrand[0][idx]
        npt.assert_allclose(
            value,
            self.check_multipole_spectra_integrand_no_z_err,
            rtol=1e-05,
            err_msg='Multipole spectra integrand no z err failed',
        )

    def test_scaling_factor_perp(self):
        npt.assert_allclose(
            self.spectro.scaling_factor_perp(0.01),
            self.check_scaling_factor_perp,
            rtol=1e-03,
            err_msg='Scaling Factor Perp failed',
        )

    def test_scaling_factor_parall(self):
        npt.assert_allclose(
            self.spectro.scaling_factor_parall(0.01),
            self.check_scaling_factor_parall,
            rtol=1e-03,
            err_msg='Scaling Factor Parall failed',
        )

    def test_get_k(self):
        npt.assert_allclose(
            self.spectro.get_k(0.001, 1, 0.01),
            self.check_get_k,
            rtol=1e-03,
            err_msg='get_k failed',
        )

    def test_get_mu(self):
        npt.assert_allclose(
            self.spectro.get_mu(1, 0.01),
            self.check_get_mu,
            rtol=1e-03,
            err_msg='get_mu failed',
        )

    def test_gal_redshift_scatter(self):
        npt.assert_allclose(
            self.spectro.gal_redshift_scatter(self.check_get_k,
                                              self.check_get_mu,
                                              0.01),
            self.check_gal_redshift_scatter,
            rtol=1e-05,
            err_msg='Gal Redshift Scatter failed',
        )

    @patch('cloe.fftlog.fftlog.fftlog.fftlog')
    @patch('cloe.spectroscopic_survey.spectro.Spectro.multipole_spectra')
    def test_multipole_correlation_function(self, mock_mul_spe, mock_fftlog):
        s_array_lin = np.linspace(1, 10, 10)
        s_array_log = np.logspace(np.log(0.1), np.log(100), 10)
        pk_array_log = np.logspace(np.log(0.1), np.log(100), 10)

        # test a call with a single value of ell and the default
        # k_grid parametrization
        mock_mul_spe.return_value = [0]
        mock_fftlog.return_value = s_array_log, pk_array_log
        corr_fun_array = \
            self.spectro.multipole_correlation_function(s_array_lin, 1.0, 0)

        # test that the shape of the returned array matches the expectations,
        # i.e. n arrays with the same size of the input s array, where n is
        # the number of ell values provided as input (1 in this case)
        npt.assert_equal((1, len(s_array_lin)), corr_fun_array.shape)
        # test that the multipole_spectra() function was called
        mock_mul_spe.assert_called()
        # test that the fftlog() function was called once
        mock_fftlog.assert_called_once()

        # now use non-default k_grid parametrization, and multiple ell values
        mock_mul_spe.reset_mock()
        mock_fftlog.reset_mock()
        ell_arr = [0, 2, 4]
        k_min = 5e-5
        k_max = 50
        k_num_points = 2**8  # this is non-default
        mock_mul_spe.return_value = [0, 0, 0]  # must be same size as ell_arr
        mock_fftlog.return_value = s_array_log, pk_array_log
        corr_fun_array = \
            self.spectro.multipole_correlation_function(s_array_lin, 1.0,
                                                        ell_arr,
                                                        k_min, k_max,
                                                        k_num_points)
        # test that the shape of the returned array matches the expectations,
        # i.e. n arrays with the same size of the input s array, where n is
        # the number of ell values provided as input (1 in this case)
        npt.assert_equal((len(ell_arr), len(s_array_lin)),
                         corr_fun_array.shape)
        # verify that multipole_spectra() was called the expected number of
        # times (i.e. k_num_points)
        npt.assert_equal(len(mock_mul_spe.call_args_list), k_num_points,
                         err_msg='Unexpected number of calls to'
                         f' multipole_spectra()')
        # verify that fftlog() was called the expected number of times
        # (i.e. len(ell_arr))
        npt.assert_equal(len(mock_fftlog.call_args_list), len(ell_arr),
                         err_msg='Unexpected number of calls to fftlog()')

    @patch('cloe.spectroscopic_survey.spectro.Spectro.'
           'multipole_correlation_function_mag_mag')
    def test_multipole_correlation_function_mag_mag(self, mock_mag_mag):
        s_array_lin = np.linspace(1, 10, 10)

        # test a call to the function
        mock_mag_mag.return_value = np.zeros(10)
        corr_fun_mag_mag_array = \
            self.spectro.multipole_correlation_function_mag_mag(s_array_lin,
                                                                1.0, 0)

        # test that the shape of the returned array matches the expectations,
        # i.e. same size as the input s array (10 in this case)
        npt.assert_equal(len(s_array_lin), len(corr_fun_mag_mag_array))
        # test that the function was called
        mock_mag_mag.assert_called()

        # verify that the function was called the expected number of
        # times (i.e. len(s_array))
        npt.assert_equal(len(mock_mag_mag.call_args_list), 1,
                         err_msg='Unexpected number of calls to'
                         f' multipole_correlation_function_mag_mag()')

    @patch('cloe.spectroscopic_survey.spectro.Spectro.'
           'multipole_correlation_function_dens_mag')
    def test_multipole_correlation_function_dens_mag(self, mock_dens_mag):
        s_array_lin = np.linspace(1, 10, 10)

        # test a call to the function
        mock_dens_mag.return_value = np.zeros(10)
        corr_fun_dens_mag_array = \
            self.spectro.multipole_correlation_function_dens_mag(s_array_lin,
                                                                 1.0, 0)
        # test that the shape of the returned array matches the expectations,
        # i.e. same size of the input s array (10 in this case)
        npt.assert_equal(len(s_array_lin), len(corr_fun_dens_mag_array))
        # test that the function was called
        mock_dens_mag.assert_called()

        # verify that function was called the expected number of
        # times (i.e. lens(s_array))
        npt.assert_equal(len(mock_dens_mag.call_args_list), 1,
                         err_msg='Unexpected number of calls to'
                         f' multipole_correlation_function_dens_mag()')

    def test_f_out(self):
        npt.assert_allclose(
            self.spectro_2.multipole_spectra(1.0, 0.1, ms=[1]),
            0.0,
            atol=1e-10,
            err_msg='Test redshift independent f_out failed',
        )

    def test_f_out_z_dep(self):
        npt.assert_allclose(
            self.spectro_3.multipole_spectra(1.0, 0.1, ms=[1]),
            0.0,
            atol=1e-10,
            err_msg='Test redshift dependent f_out failed',
        )

    def test_f_out_z_exception(self):
        npt.assert_raises(
            Exception,
            self.spectro_3.multipole_spectra,
            20.0,
            0.1,
            ms=[1],
        )
