"""UNIT TESTS FOR EFT MODULE

This module contains unit tests for the eft module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.non_linear.eft import EFTofLSS
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class eftinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # Load cosmology dictionary for tests
        cosmo_dic = load_test_pickle('cosmo_test_NLspectro1_dic.pickle')
        # Create instance of EFTofLSS class
        cls.eft = EFTofLSS(cosmo_dic)
        cls.PEH = cls.eft.CallEH_NW()

    def setUp(self) -> None:
        # Check values
        self.PEH_test = 18366.38013
        self.Pnw_test = 17711.220032
        self.Pw_test = -1060.852868
        self.Sigma2_test = 60.664153
        self.dSigma2_test = 22.530151
        self.Sig2mu_test = 87.258008
        self.RSDdamp_test = 0.642037
        self.Pb1b1_test = [-1284.521497, 427.9779]
        self.Pb1b2_test = [2982.8153, 9.249978]
        self.Pb2b2_test = [-1306.100102, -4.209333]
        self.Pmu2fb1_test = [-4052.274427, 854.6795]
        self.Pmu2fb2_test = [808.474363, 23.71016]
        self.Pmu2f2b12_test = [-3746.652715, 322.0014]
        self.Pmu2fb12_test = [-1085.81156, 857.2321]
        self.Pmu2fb1b2_test = [5157.156238, -5.210208]
        self.Pmu2f2b1_test = [-1297.827504, -1.116758]
        self.Pmu2f2b2_test = [-1902.54832, 12.65266]
        self.Pmu4f4_test = [460.493797, -3.279820]
        self.Pmu4f3v_test = [-596.12267, -6.114579]
        self.Pmu4f3b1_test = [-9492.837769, 647.883888]
        self.Pmu4f2b2_test = [3254.607917, 7.442455]
        self.Pmu4f2b1_test = [-5754.63044, 1717.782974]
        self.Pmu4f2b12_test = [4346.336843, 104.3968957]
        self.Pmu4f2_test = [-2366.77874, 423.845718]
        self.Pmu6f4_test = [-5439.189189, 323.695929]
        self.Pmu6f3_test = [-3967.114046, 855.553038]
        self.Pmu6f3b1_test = [6693.141348, 212.674854]
        self.Pmu8f4_test = [2807.298302, 104.998138]
        self.PZ1b1_test = [1782.733917, -109.2234]
        self.PZ1mu2f_test = [-1609.621346, 99.245109]
        self.PZ1mu2fb1_test = [6957.823096, -426.915309]
        self.PZ1mu2f2_test = [-3816.399671, 234.527073]
        self.PZ1mu4f2_test = [3141.423425, -192.388236]
        self.Pgg_kmu_test = [[137641.029656, 210569.609045],
                             [52551.209503, 79154.241991],
                             [21559.529345, 34160.972728]]
        self.nuis = {'b1': 2.0, 'b2': 0.5,
                     'c0': 1.0, 'c2': 1.0, 'c4': 1.0,
                     'aP': 1.0, 'Psn': 1000.0}
        self.index = 500
        self.lamb = 0.25
        self.rtol = 1e-3
        self.z1 = 1.0
        self.k1 = 0.01
        self.k2 = 0.05
        self.k3 = 0.1
        self.mu1 = 0.5
        self.mu2 = 1.0
        self.f = 0.7
        self.D = 0.6

    def tearDown(self):
        self.PEH = None

    def test_CallEH_NW(self):
        npt.assert_allclose(
            self.eft.CallEH_NW()[self.index],
            self.PEH_test,
            rtol=self.rtol,
            err_msg='Error in value returned by CallEH_NW',
        )

    def test_gaussianFiltering(self):
        npt.assert_allclose(
            self.eft._gaussianFiltering(self.lamb)[self.index],
            self.Pnw_test,
            rtol=self.rtol,
            err_msg='Error in value returned by _gaussianFiltering'
        )

    def test_IRresum(self):
        Pnw, Pw, Sigma2, dSigma2 = self.eft.IRresum(self.PEH)
        npt.assert_allclose(
            [Pnw[self.index], Pw[self.index], Sigma2, dSigma2],
            [self.Pnw_test, self.Pw_test, self.Sigma2_test, self.dSigma2_test],
            rtol=self.rtol,
            err_msg='Error in value returned by IRresum'
        )

    def test_muxdamp(self):
        Sig2mu, RSDdamp = self.eft._muxdamp(self.mu1, self.Sigma2_test,
                                            self.dSigma2_test, self.f)
        npt.assert_allclose(
            [Sig2mu, RSDdamp[self.index, 0]],
            [self.Sig2mu_test, self.RSDdamp_test],
            rtol=self.rtol, err_msg='Error in value returned by _muxdamp'
        )

    def test_Pgg_kmu_terms(self):
        self.eft._Pgg_kmu_terms()
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop22_nw],
            [self.Pb1b1_test[0], self.Pb1b2_test[0], self.Pb2b2_test[0],
             self.Pmu2fb1_test[0], self.Pmu2fb2_test[0],
             self.Pmu2f2b12_test[0], self.Pmu2fb12_test[0],
             self.Pmu2fb1b2_test[0], self.Pmu2f2b1_test[0],
             self.Pmu2f2b2_test[0], self.Pmu4f4_test[0], self.Pmu4f3v_test[0],
             self.Pmu4f3b1_test[0], self.Pmu4f2b2_test[0],
             self.Pmu4f2b1_test[0], self.Pmu4f2b12_test[0],
             self.Pmu4f2_test[0], self.Pmu6f4_test[0], self.Pmu6f3_test[0],
             self.Pmu6f3b1_test[0], self.Pmu8f4_test[0]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: nw 22 '
        )
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop22_w],
            [self.Pb1b1_test[1], self.Pb1b2_test[1], self.Pb2b2_test[1],
             self.Pmu2fb1_test[1], self.Pmu2fb2_test[1],
             self.Pmu2f2b12_test[1], self.Pmu2fb12_test[1],
             self.Pmu2fb1b2_test[1], self.Pmu2f2b1_test[1],
             self.Pmu2f2b2_test[1], self.Pmu4f4_test[1], self.Pmu4f3v_test[1],
             self.Pmu4f3b1_test[1], self.Pmu4f2b2_test[1],
             self.Pmu4f2b1_test[1], self.Pmu4f2b12_test[1],
             self.Pmu4f2_test[1], self.Pmu6f4_test[1], self.Pmu6f3_test[1],
             self.Pmu6f3b1_test[1], self.Pmu8f4_test[1]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: w 22 '
        )
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop13_nw],
            [self.PZ1b1_test[0], self.PZ1mu2f_test[0], self.PZ1mu2fb1_test[0],
             self.PZ1mu2f2_test[0], self.PZ1mu4f2_test[0]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: nw 13 '
        )
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop13_w],
            [self.PZ1b1_test[1], self.PZ1mu2f_test[1], self.PZ1mu2fb1_test[1],
             self.PZ1mu2f2_test[1], self.PZ1mu4f2_test[1]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: w 13 '
        )

    def test_P_kmu_z(self):
        self.eft._Pgg_kmu_terms()
        interp = self.eft.P_kmu_z(self.f, self.D, **self.nuis)
        npt.assert_allclose(
            interp([self.k1, self.k2, self.k3], [self.mu1, self.mu2]),
            self.Pgg_kmu_test,
            rtol=self.rtol, err_msg='Error in value returned by '
                                    'P_kmu_z'
        )
