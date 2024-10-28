"""UNIT TESTS FOR EFT MODULE

This module contains unit tests for the :obj:`eft` module.

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
        cls.eft.PL = cosmo_dic['Pk_cb'].P(0.0, cls.eft.ks)
        cls.PEH = cls.eft.CallEH_NW(0.0)

    def setUp(self) -> None:
        # Check values
        self.PEH_test = 18366.38013
        self.Pnw_GS1D_test = 17711.220032
        self.Pnw_test = 17513.85
        self.Pw_test = -863.4851
        self.Sigma2_test = 64.03505
        self.dSigma2_test = 17.48366
        self.Sig2mu_test = 92.6853
        self.RSDdamp_test = 0.624584
        self.Pb1b1_test = [-1205.491500, 348.947914]
        self.Pb1b2_test = [2987.454504, 4.610774]
        self.Pb1bG2_test = [-5190.131347, -5.645702]
        self.Pb2b2_test = [-1309.525568, -0.783869]
        self.Pb2bG2_test = [-7569.420898, 9.838267]
        self.PbG2bG2_test = [4891.545302, -14.596208]
        self.Pmu2fb1_test = [-3893.877670, 696.282769]
        self.Pmu2fb2_test = [824.762819, 7.421707]
        self.Pmu2fbG2_test = [-2394.962603, -13.986393]
        self.Pmu2f2b12_test = [-3691.475360, 266.824056]
        self.Pmu2fb12_test = [-928.088329, 699.508885]
        self.Pmu2fb1b2_test = [5150.146189, 1.799840]
        self.Pmu2fb1bG2_test = [-7985.300091, 2.694988]
        self.Pmu2f2b1_test = [-1297.532837, -1.411426]
        self.Pmu2f2b2_test = [-1892.355224, 2.459567]
        self.Pmu2f2bG2_test = [2445.772651, -7.298104]
        self.Pmu4f4_test = [458.582372, -1.368395]
        self.Pmu4f3_test = [-598.740651, -3.496598]
        self.Pmu4f3b1_test = [-9379.275741, 534.321860]
        self.Pmu4f2b2_test = [3257.790965, 4.259407]
        self.Pmu4f2bG2_test = [-5539.527440, -4.603116]
        self.Pmu4f2b1_test = [-5435.223806, 1398.376341]
        self.Pmu4f2b12_test = [4368.188351, 82.545388]
        self.Pmu4f2_test = [-2289.076350, 346.143329]
        self.Pmu6f4_test = [-5382.078801, 266.585540]
        self.Pmu6f3_test = [-3808.343291, 696.782283]
        self.Pmu6f3b1_test = [6740.051679, 165.764522]
        self.Pmu8f4_test = [2830.445700, 81.850740]
        self.PZ1b1_test = [1760.516264, -87.005747]
        self.PZ1bG3_test = [-10046.954554, 495.294292]
        self.PZ1bG2_test = [-25117.386384, 1238.235731]
        self.PZ1mu2f_test = [-1588.468587, 78.092350]
        self.PZ1mu2fb1_test = [6870.017380, -339.109592]
        self.PZ1mu2f2_test = [-3767.607958, 185.735360]
        self.PZ1mu4f2_test = [3102.409421, -153.374232]
        self.Pgg_kmu_test = [[144746.768191, 242578.11107],
                             [56833.028809, 93288.221193],
                             [24680.05168, 43745.442421]]
        self.nuis = {'b1': 2.0, 'b2': 0.5, 'bG2': -0.7, 'bG3': 1.2,
                     'c0': 1.0, 'c2': 1.0, 'c4': 1.0, 'ck4': 10.,
                     'aP': 1.0, 'e0k2': -1.2, 'e2k2': -5.3, 'Psn': 1000.0}
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

        self.Pb1b2_phot_test = 5.319397444374871
        self.Pb1bG2_phot_test = -16.653791713284814
        self.Pb2b2_phot_test = 928.6939554015156
        self.Pb2bG2_phot_test = -127.59701366040896
        self.PbG2bG2_phot_test = 32.2833850269415
        self.PZ1bG3_phot_test = -262.90858279662723
        self.PZ1bG2_phot_test = -657.2714569915681

    def tearDown(self):
        self.PEH = None

    def test_CallEH_NW(self):
        npt.assert_allclose(
            self.eft.CallEH_NW(0.0)[self.index],
            self.PEH_test,
            rtol=self.rtol,
            err_msg='Error in value returned by CallEH_NW',
        )

    def test_gaussianFiltering(self):
        npt.assert_allclose(
            self.eft._gaussianFiltering(self.lamb, 0.0)[self.index],
            self.Pnw_GS1D_test,
            rtol=self.rtol,
            err_msg='Error in value returned by _gaussianFiltering'
        )

    def test_IRresum(self):
        Pnw, Pw, Sigma2, dSigma2 = self.eft.IRresum(IRres='DST')
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
        self.eft._Pgg_kmu_terms(redshift=0.0)
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop22_nw],
            [self.Pb1b1_test[0], self.Pb1b2_test[0], self.Pb1bG2_test[0],
             self.Pb2b2_test[0], self.Pb2bG2_test[0], self.PbG2bG2_test[0],
             self.Pmu2fb1_test[0], self.Pmu2fb2_test[0], self.Pmu2fbG2_test[0],
             self.Pmu2f2b12_test[0], self.Pmu2fb12_test[0],
             self.Pmu2fb1b2_test[0], self.Pmu2fb1bG2_test[0],
             self.Pmu2f2b1_test[0], self.Pmu2f2b2_test[0],
             self.Pmu2f2bG2_test[0], self.Pmu4f4_test[0], self.Pmu4f3_test[0],
             self.Pmu4f3b1_test[0], self.Pmu4f2b2_test[0],
             self.Pmu4f2bG2_test[0], self.Pmu4f2b1_test[0],
             self.Pmu4f2b12_test[0], self.Pmu4f2_test[0], self.Pmu6f4_test[0],
             self.Pmu6f3_test[0], self.Pmu6f3b1_test[0], self.Pmu8f4_test[0]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: nw 22 '
        )
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop22_w],
            [self.Pb1b1_test[1], self.Pb1b2_test[1], self.Pb1bG2_test[1],
             self.Pb2b2_test[1], self.Pb2bG2_test[1], self.PbG2bG2_test[1],
             self.Pmu2fb1_test[1], self.Pmu2fb2_test[1], self.Pmu2fbG2_test[1],
             self.Pmu2f2b12_test[1], self.Pmu2fb12_test[1],
             self.Pmu2fb1b2_test[1], self.Pmu2fb1bG2_test[1],
             self.Pmu2f2b1_test[1], self.Pmu2f2b2_test[1],
             self.Pmu2f2bG2_test[1], self.Pmu4f4_test[1], self.Pmu4f3_test[1],
             self.Pmu4f3b1_test[1], self.Pmu4f2b2_test[1],
             self.Pmu4f2bG2_test[1], self.Pmu4f2b1_test[1],
             self.Pmu4f2b12_test[1], self.Pmu4f2_test[1], self.Pmu6f4_test[1],
             self.Pmu6f3_test[1], self.Pmu6f3b1_test[1], self.Pmu8f4_test[1]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: w 22 '
        )
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop13_nw],
            [self.PZ1b1_test[0], self.PZ1bG3_test[0], self.PZ1bG2_test[0],
             self.PZ1mu2f_test[0], self.PZ1mu2fb1_test[0],
             self.PZ1mu2f2_test[0], self.PZ1mu4f2_test[0]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: nw 13 '
        )
        npt.assert_allclose(
            [term[self.index] for term in self.eft.loop13_w],
            [self.PZ1b1_test[1], self.PZ1bG3_test[1], self.PZ1bG2_test[1],
             self.PZ1mu2f_test[1], self.PZ1mu2fb1_test[1],
             self.PZ1mu2f2_test[1], self.PZ1mu4f2_test[1]],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    '_Pgg_kmu_terms: w 13 '
        )

    def test_P_kmu_z(self):
        interp = self.eft.P_kmu_z(self.z1, True, **self.nuis)
        npt.assert_allclose(
            interp([self.k1, self.k2, self.k3], [self.mu1, self.mu2]),
            self.Pgg_kmu_test,
            rtol=self.rtol, err_msg='Error in value returned by '
                                    'P_kmu_z'
        )

    def test_P_realspace_terms_kz(self):
        self.eft._Pgg_k_terms_L()
        terms = self.eft.P_realspace_terms_kz()
        npt.assert_allclose(
            [t(self.z1, self.k1)[0, 0] for t in terms],
            [self.Pb1b2_phot_test, self.Pb1bG2_phot_test,
             self.Pb2b2_phot_test, self.Pb2bG2_phot_test,
             self.PbG2bG2_phot_test, self.PZ1bG3_phot_test,
             self.PZ1bG2_phot_test],
            rtol=self.rtol, err_msg='Error in value returned by '
                                    'P_realspace_terms_kz '
        )
