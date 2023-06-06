"""UNIT TESTS FOR LIKE_CALC

This module contains unit tests for the likelihood calculation module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
import copy
from cloe.like_calc.euclike import Euclike
from cloe.tests.test_input.data import mock_data
from cloe.tests.test_input.mock_observables import build_mock_observables
from cloe.tests.test_tools.spectro_test_handler import SpectroTestParent
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class likecalcTestCase(TestCase, SpectroTestParent):

    def setUp(self):

        mock_cosmo_dic = load_test_pickle('like_calc_test_dic.pickle')
        mock_cosmo_dic['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        # self.fiducial_dict = fid_mock_dic
        self.test_dict = mock_cosmo_dic
        # init Euclike
        mock_obs = build_mock_observables()
        self.like_tt = Euclike(mock_data, mock_obs)
        self.like_tt.fiducial_cosmo_quantities_dic.update(
            self.test_dict)
        self.like_tt.get_masked_data()
        # The correct check value, using the h scaling for the h from
        # supplied external file for all the probes together is:
        self.check_loglike = -1721.244143

    def tearDown(self):
        self.check_loglike = None

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

        istf_bias_list = [
            self.test_dict['nuisance_parameters']['b1_spectro_bin1'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin2'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin3'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin4']
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

    def test_loglike(self):
        npt.assert_allclose(
            self.like_tt.loglike(self.test_dict, 1),
            self.check_loglike,
            rtol=1e-06,
            err_msg='Loglike failed',
        )


class likecalcBNT_TestCase(TestCase, SpectroTestParent):

    def setUp(self):

        mock_data['cov_model'] = 'Gauss'
        mock_cosmo_dic = load_test_pickle('like_calc_test_dic.pickle')
        mock_cosmo_dic['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)
        # self.fiducial_dict = fid_mock_dic
        self.test_dict = mock_cosmo_dic
        # init Euclike
        mock_obs = build_mock_observables()
        mock_obs['selection']['matrix_transform_phot'] = 'BNT-test'
        self.like_tt = Euclike(mock_data, mock_obs)
        self.like_tt.fiducial_cosmo_quantities_dic.update(
            self.test_dict)
        self.like_tt.get_masked_data()
        # The correct check value, using the h scaling for the h from
        # supplied external file for all the probes together is:
        self.check_loglike = -1721.244143

    def tearDown(self):
        self.check_loglike = None

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

        istf_bias_list = [
            self.test_dict['nuisance_parameters']['b1_spectro_bin1'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin2'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin3'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin4']
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

    def test_loglike_BNT(self):
        npt.assert_allclose(
            self.like_tt.loglike(self.test_dict, 1),
            self.check_loglike,
            rtol=1e-06,
            err_msg='Loglike failed',
        )


class likecalcngTestCase(TestCase, SpectroTestParent):

    def setUp(self):

        mock_cosmo_dic = load_test_pickle('like_calc_test_dic.pickle')
        mock_cosmo_dic['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)

        # self.fiducial_dict = fid_mock_dic
        self.test_dict = mock_cosmo_dic
        # init Euclike
        mock_obs = build_mock_observables()
        self.mock_data_ng = copy.deepcopy(mock_data)
        self.mock_data_ng['spectro']['cov_is_num'] = True
        self.mock_data_ng['spectro']['cov_nsim'] = 3500
        self.like_tt_ng = Euclike(self.mock_data_ng, mock_obs)
        self.like_tt_ng.fiducial_cosmo_quantities_dic.update(
            self.test_dict)
        self.like_tt_ng.get_masked_data()

        # The correct check value, using the h scaling for the h from
        # supplied external file for all the probes together is:
        self.check_loglike_ng = -1682.686194

    def tearDown(self):
        self.check_loglike_ng = None

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

        istf_bias_list = [
            self.test_dict['nuisance_parameters']['b1_spectro_bin1'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin2'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin3'],
            self.test_dict['nuisance_parameters']['b1_spectro_bin4']
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

    def test_loglike_ng(self):
        npt.assert_allclose(
            self.like_tt_ng.loglike(self.test_dict, 10),
            self.check_loglike_ng,
            rtol=1e-06,
            err_msg='Non-gaussian loglike failed',
        )

    def test_spectro_cov_nsim(self):
        self.mock_data_ng['spectro']['cov_nsim'] = 1
        npt.assert_raises(ValueError, self.like_tt_ng.get_masked_data)
        self.mock_data_ng['spectro']['cov_nsim'] = 3500
