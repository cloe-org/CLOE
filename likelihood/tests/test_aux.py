"""UNIT TESTS FOR AUXILIARY

This module contains unit tests for the auxiliary functions module.
=======

"""

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from unittest import TestCase
from ..cosmo.cosmology import Cosmology
from likelihood.tests.test_wrapper import CobayaModel
from likelihood.auxiliary.plotter import Plotter


class plotterTestCase(TestCase):

    def setUp(self):
        cosmo = Cosmology()
        cosmo.cosmo_dic['ombh2'] = 0.022
        cosmo.cosmo_dic['omch2'] = 0.12
        cosmo.cosmo_dic['H0'] = 68.0
        cosmo.cosmo_dic['tau'] = 0.07
        cosmo.cosmo_dic['mnu'] = 0.06
        cosmo.cosmo_dic['nnu'] = 3.046
        cosmo.cosmo_dic['ns'] = 0.9674
        cosmo.cosmo_dic['As'] = 2.1e-9

        self.model_test = CobayaModel(cosmo)
        self.model_test.update_cosmo()
        fig1 = plt.figure()
        self.ax1 = fig1.add_subplot(1, 1, 1)

        self.plot_inst = Plotter(self.model_test.cosmology.cosmo_dic)

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
