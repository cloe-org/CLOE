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
        self.Pnw_test = 17568.73279
        self.Pw_test = -1055.290097
        self.Sigma2_test = 60.210358
        self.dSigma2_test = 22.375376
        self.Sig2mu_test = 86.604014
        self.RSDdamp_test = 0.644173
        self.Pb1b1_test = [-1259.130012, 421.5848]
        self.Pb1b2_test = [2939.746131, 7.978035]
        self.Pb2b2_test = [-1289.037198, -3.694734]
        self.Pmu2fb1_test = [-3979.713294, 842.4677]
        self.Pmu2fb2_test = [799.206919, 22.91875]
        self.Pmu2f2b12_test = [-3686.998443, 318.0360]
        self.Pmu2fb12_test = [-1056.806753, 843.8716]
        self.Pmu2fb1b2_test = [5080.285343, -6.962682]
        self.Pmu2f2b1_test = [-1278.771612, -0.6142177]
        self.Pmu2f2b2_test = [-1872.97181, 13.07313]
        self.Pmu4f4_test = [453.491255, -3.392519]
        self.Pmu4f3v_test = [-587.737319, -5.783770]
        self.Pmu4f3b1_test = [-9343.802791, 640.730291]
        self.Pmu4f2b2_test = [3207.313533, 6.107573]
        self.Pmu4f2b1_test = [-5643.591866, 1692.57167]
        self.Pmu4f2b12_test = [4284.198441, 101.303677]
        self.Pmu4f2_test = [-2325.706544, 417.989582]
        self.Pmu6f4_test = [-5354.476845, 320.381771]
        self.Pmu6f3_test = [-3895.75082, 843.413647]
        self.Pmu6f3b1_test = [6598.590977, 207.163871]
        self.Pmu8f4_test = [2767.883791, 102.467745]
        self.PZ1b1_test = [1754.336536, -108.001865]
        self.PZ1mu2f_test = [-1583.162817, 98.023244]
        self.PZ1mu2fb1_test = [6846.172425, -422.028837]
        self.PZ1mu2f2_test = [-3754.686772, 231.778247]
        self.PZ1mu4f2_test = [3091.485652, -190.250591]
        self.Pgg_kmu_test = [[137092.131545, 209729.006641],
                             [52162.804141, 78573.175201],
                             [21385.385007, 33866.47418]]
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
