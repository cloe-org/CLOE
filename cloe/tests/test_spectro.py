"""UNIT TESTS FOR SPECTRO

This module contains unit tests for the Spectro sub-module of the
spectroscopy survey module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.spectroscopic_survey.spectro import Spectro
from cloe.tests.test_tools.spectro_test_handler import SpectroTestParent
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class specinitTestCase(TestCase, SpectroTestParent):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dict = load_test_pickle('spectro_test_dic.pickle')
        cls.spectro = Spectro(cls.test_dict, ['1.', '1.2', '1.4', '1.65'])

    def setUp(self):
        self.test_dict['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        self.check_multipole_spectra_m0 = 12292.778742
        self.check_multipole_spectra_m1 = 0.0
        self.check_multipole_spectra_m2 = 8408.473137
        self.check_multipole_spectra_m3 = 0.0
        self.check_multipole_spectra_m4 = 678.085174
        self.check_multipole_spectra_integrand = 3343.949991
        self.check_scaling_factor_perp = 1.007444
        self.check_scaling_factor_parall = 1.007426
        self.check_get_k = 0.000993
        self.check_get_mu = 1.00

    def tearDown(self):
        self.check_scaling_factor_perp = None
        self.check_scaling_factor_parall = None
        self.check_get_k = None
        self.check_get_mu = None

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
            nuisance_dic['b1_spectro'],
            nuisance_dic['b2_spectro'],
            nuisance_dic['b3_spectro'],
            nuisance_dic['b4_spectro']
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

    def test_multipole_spectra_integrand(self):
        mu_grid = np.linspace(-1, 1, 2001)
        integrand = (
            self.spectro.multipole_spectra_integrand(mu_grid, 1.0, 0.1, [2])
        )
        idx = np.where(mu_grid == 0.7)
        value = integrand[0][idx]
        npt.assert_allclose(
            value,
            self.check_multipole_spectra_integrand,
            rtol=1e-06,
            err_msg='Multipole spectra integrand failed',
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

    def test_f_out(self):
        self.test_dict['nuisance_parameters']['f_out'] = 1.0
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[1]),
            0.0,
            atol=1e-10,
            err_msg='Test redshift independent f_out failed',
        )
        self.test_dict['nuisance_parameters']['f_out'] = 0.0

    def test_f_out_z_dep(self):
        self.test_dict['f_out_z_dep'] = True
        self.test_dict['nuisance_parameters']['f_out_1'] = 1.0
        npt.assert_allclose(
            self.spectro.multipole_spectra(1.0, 0.1, ms=[1]),
            0.0,
            atol=1e-10,
            err_msg='Test redshift dependent f_out failed',
        )
        self.test_dict['f_out_z_dep'] = False
        self.test_dict['nuisance_parameters']['f_out_1'] = 0.0

    def test_f_out_z_exception(self):
        self.test_dict['f_out_z_dep'] = True
        npt.assert_raises(
            Exception,
            self.spectro.multipole_spectra,
            20.0,
            0.1,
            ms=[1],
        )
        self.test_dict['f_out_z_dep'] = False
