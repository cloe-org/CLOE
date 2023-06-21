"""UNIT TESTS FOR FFTLog

This module contains unit tests for the :obj:`FFTLog` module.


"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.fftlog.hankel import hankel


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
        self.atol = 1e-03

    def tearDown(self):
        self.a = None
        self.k = None
        self.fk = None
        self.atol = None

    def test_hankel(self):
        myhankel = hankel(self.k, self.fk, nu=1.01, N_extrap_begin=1500,
                          N_extrap_end=1500, c_window_width=0.25, N_pad=500)
        r, Fr = myhankel.hankel(0)
        real_Fr = F(r)
        npt.assert_allclose(real_Fr, Fr,
                            atol=self.atol,
                            err_msg='FFTLog test failed')
