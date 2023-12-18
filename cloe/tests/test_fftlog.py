"""UNIT TESTS FOR FFTLog

This module contains unit tests for the :obj:`FFTLog` module.


"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.fftlog.hankel import hankel
from cloe.fftlog.fftlog import fftlog
from cloe.tests.test_tools import test_data_handler as tdh


def f(k, a=1):
    return np.exp(-k**2.0 * a**2 / 2)


def F(r, a=1):
    return np.exp(- r**2.0 / (2. * a**2))


class fftloginitTestCase(TestCase):

    def setUp(self) -> None:
        # Check values
        self.a = 1
        self.k = np.logspace(-5., 1., 2**10)
        self.fk = f(self.k)
        self.rtol = 1e-6
        self.atol = 1e-3
        self.check_bl_Fr1 = tdh.load_test_npy('check_bl_Fr1.npy')
        self.check_bl_r1 = tdh.load_test_npy('check_bl_r1.npy')
        self.check_bl_Fr2 = tdh.load_test_npy('check_bl_Fr2.npy')
        self.check_bl_r2 = tdh.load_test_npy('check_bl_r2.npy')
        self.Pk_test = tdh.load_test_npy('Pk_test.npy')

    def tearDown(self):
        self.a = None
        self.k = None
        self.fk = None
        self.atol = None
        self.rtol = None
        self.check_bl_Fr1 = None
        self.check_bl_r1 = None
        self.check_bl_Fr2 = None
        self.check_bl_r2 = None
        self.Pk_test = None

    def test_hankel(self):
        myhankel = hankel(self.k, self.fk, nu=1.01, N_extrap_begin=1500,
                          N_extrap_end=1500, c_window_width=0.25, N_pad=500)
        r, Fr = myhankel.hankel(0)
        real_Fr = F(r)
        npt.assert_allclose(real_Fr, Fr,
                            atol=self.atol,
                            err_msg='FFTLog test failed')

    def test_fftlog_dj(self):
        myfftlog = fftlog(self.Pk_test[:, 0], self.Pk_test[:, 1], nu=1.01,
                          N_extrap_begin=1500, N_extrap_end=1500,
                          c_window_width=0.25, N_pad=2000)
        r, Fr_dj = myfftlog.fftlog_first_derivative(1)

        npt.assert_allclose(self.check_bl_Fr1, Fr_dj,
                            rtol=self.rtol,
                            err_msg='FFTLog_dj test failed')

    def test_fftlog_ddj(self):
        myfftlog = fftlog(self.Pk_test[:, 0], self.Pk_test[:, 1], nu=1.01,
                          N_extrap_begin=1500, N_extrap_end=1500,
                          c_window_width=0.25, N_pad=2000)
        r, Fr_dj = myfftlog.fftlog_second_derivative(1)

        npt.assert_allclose(self.check_bl_Fr2, Fr_dj,
                            rtol=self.rtol,
                            err_msg='FFTLog_ddj test failed')
