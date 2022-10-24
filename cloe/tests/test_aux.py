"""UNIT TESTS FOR AUXILIARY

This module contains unit tests for the auxiliary functions module.

"""

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from unittest import TestCase
from cloe.auxiliary.plotter import Plotter
from cloe.tests.test_input.data import mock_data
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class plotterTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        mock_cosmo_dic = load_test_pickle('spectro_test_dic.pickle')

        fig1 = plt.figure()
        cls.ax1 = fig1.add_subplot(1, 1, 1)

        cls.plot_inst = Plotter(mock_cosmo_dic, mock_data)

    def test_plotter_init(self):
        npt.assert_raises(Exception, Plotter)

    def test_plot_phot_excep(self):
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_Cl_phot,
            np.array([1.0]),
            1,
            1,
            self.ax1,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_Cl_phot,
            np.array([10.0]),
            1,
            1,
        )

    def test_ext_phot_excep(self):
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_external_Cl_phot,
            11,
            1,
            self.ax1,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_external_Cl_phot,
            10,
            1,
        )

    def test_plot_XC_phot_excep(self):
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_Cl_XC,
            1.0,
            1,
            1,
            self.ax1,
        )
        npt.assert_raises(Exception, self.plot_inst.plot_Cl_XC, 10.0, 1, 1)

    def test_ext_XC_phot_excep(self):
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_external_Cl_XC,
            11,
            1,
            self.ax1,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_external_Cl_XC,
            10,
            1,
        )

    def test_plot_GC_spectro_multipole_excep(self):
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            0.1,
            np.array([0.1]),
            5,
            self.ax1,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            0.1,
            np.array([0.1]),
            2,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            0.1,
            np.array([10.0]),
            2,
            self.ax1,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            3.0,
            np.array([0.1]),
            2,
            self.ax1,
        )

    def test_ext_GC_spectro_multipole_excep(self):
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            '1.2',
            5,
            self.ax1,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            '1.2',
            2,
        )
        npt.assert_raises(
            Exception,
            self.plot_inst.plot_GC_spectro_multipole,
            '3.0',
            2,
            self.ax1,
        )
