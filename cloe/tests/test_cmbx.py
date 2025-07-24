"""UNIT TESTS FOR CMBX code

This module contains unit tests for the code written by the CMBX lensing team.
==============================================================================

The pickles are computed using the notebook
"Gererate test spectro, phot and like_calc pickles.ipynb"



"""

from scipy import interpolate

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.cosmo.cosmology import Cosmology

from cloe.photometric_survey import photo
from cloe.cmbx_p.cmbx import CMBX
from cloe.tests.test_tools import test_data_handler as tdh


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


class mock_camb_data:
    def __init__(self):
        self.tau_maxvis = 280.39892358828456

    def conformal_time(self, x):
        return 14164.044004163536

    # def redshift_at_comoving_radial_distance(self, chi):
    #     return 1088.6794458695929


class cmbxinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        mock_cosmo_dic = tdh.load_test_pickle("cmbx_test_dic.pickle")
        mock_cosmo_dic["CAMBdata"] = mock_camb_data()
        mock_cosmo_dic['use_Weyl'] = False
        mock_cosmo_dic['chistar'] = \
            mock_cosmo_dic['CAMBdata'].conformal_time(0) - \
            mock_cosmo_dic["CAMBdata"].tau_maxvis

        nz_dic_WL = tdh.load_test_npy("nz_dict_WL.npy").item()
        nz_dic_GC = tdh.load_test_npy("nz_dict_GC_phot.npy").item()

        cls.ells_WL = np.arange(2, 10, 1)
        cls.ells_XC = np.arange(2, 15, 1)
        cls.ells_GC_phot = np.arange(2, 20, 1)
        cls.ells_kCMB = np.arange(2, 10, 1)
        cls.ells_kCMB_X_GC_phot = np.arange(2, 10, 1)
        cls.ells_kCMB_X_WL = np.arange(2, 10, 1)
        cls.ells_ISW_X_GC = np.arange(2, 10, 1)

        cls.nz_dic_WL = nz_dic_WL
        cls.nz_dic_GC = nz_dic_GC
        cls.flatnz = mock_cosmo_dic["Flat_nz"]

        cls.phot = photo.Photo(
            None,
            nz_dic_WL,
            nz_dic_GC,
            add_RSD=False,
        )
        cls.phot.set_prefactor(
            ells_WL=cls.ells_WL,
            ells_XC=cls.ells_XC,
            ells_GC_phot=cls.ells_GC_phot,
        )
        cls.phot.update(mock_cosmo_dic)
        cls.cmbx = CMBX(cls.phot)
        cls.cmbx.cmbx_update(cls.phot)
        cls.cmbx.cmbx_set_prefactor(
            ells_kCMB_X_WL=cls.ells_kCMB_X_WL,
            ells_kCMB_X_GC_phot=cls.ells_kCMB_X_GC_phot,
            ells_ISW_X_GC=cls.ells_ISW_X_GC,
        )

    def setUp(self) -> None:
        self.win_tol = 1e-03
        self.cl_tol = 2e-03
        self.win_kCMBcheck = 2.3604456e-4  # 2.350741e-04
        self.cl_kCMBcheck = 1.71913002e-07  # 1.733371e-07
        self.cl_kCMB_X_GCcheck = 1.40492352e-06  # 1.462705e-06
        self.cl_kCMB_X_WLcheck = 1.04289059e-08  # 1.077229e-08

    def tearDown(self):
        self.win_kCMBcheck = None
        self.cl_kCMB_check = None
        self.cl_kCMB_X_GCcheck = None
        self.cl_kCMB_X_GCcheck = None

    def test_kCMB_window(self):
        print(self.cmbx.kCMB_window(2.0))
        npt.assert_allclose(
            self.cmbx.kCMB_window(2.0),
            self.win_kCMBcheck,
            rtol=self.win_tol,
            err_msg="kCMB_window failed",
        )

    # TODO: Cl kCMB not working becasue needed to add the Pk Weyl into
    # the test pickle, see update_dict_w_mock in
    # tests/test_tools/update_dict_w_mock
    # used to run the notebook Gererate test spectro,
    # phot and like_calc pickles
    # def test_cl_kCMB(self):
    #     ells = np.array([10.0])
    #     npt.assert_allclose(
    #         self.cmbx.Cl_kCMB(ells),
    #         self.cl_kCMBcheck,
    #         rtol=self.cl_tol,
    #         err_msg="Cl kCMB test failed",
    #     )

    def test_cl_kCMB_X_GC(self):
        ells = np.array([10.0])
        npt.assert_allclose(
            self.cmbx.Cl_kCMB_X_GC_phot(ells, 1),
            self.cl_kCMB_X_GCcheck,
            rtol=self.cl_tol,
            err_msg="Cl kCMBxGC test failed",
        )

    def test_cl_kCMB_X_WL(self):
        ells = np.array([10.0])
        npt.assert_allclose(
            self.cmbx.Cl_kCMB_X_WL(ells, 1),
            self.cl_kCMB_X_WLcheck,
            rtol=self.cl_tol,
            err_msg="Cl kCMBxWL test failed",
        )
