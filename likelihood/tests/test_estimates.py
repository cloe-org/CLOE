"""UNIT TESTS FOR THEORY

This module contains unit tests for the estimates module in likelihoods
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import integrate

from likelihood.general_specs import estimates


class WLnzTestCase(TestCase):

    def setUp(self):
        self.nRBchecks = [0.023200287747794302, 0.3444277243561545,
                          0.037586840798216704]
        self.nfinchecks = [1.7212476944896058, 3.258728496592645,
                           3.193587520650502]
        self.concheck = 70.71084813162472

    def tearDown(self):
        self.nRBchecks = None
        self.nfinchecks = None
        self.concheck = None

    def test_true_rb_nz(self):
        galdist = estimates.Galdist(1)
        rb1 = galdist.n_istf_int(0.1)
        rb2 = galdist.n_istf_int(1.0)
        rb3 = galdist.n_istf_int(2.0)
        npt.assert_allclose([rb1, rb2, rb3],
                            self.nRBchecks,
                            err_msg='True unormalised n(z) from Euclid '
                                    'RedBook is incorrect.')

    def test_normalisation(self):
        galdist = estimates.Galdist(1)
        proptest = (galdist.n_istf(z=0.1, n_gal=30.0) /
                    galdist.n_istf_int(z=0.1))
        npt.assert_allclose(proptest, self.concheck,
                            err_msg='n(z) proportionality constant not'
                                    'calculating correctly.')

    def test_phot_p(self):
        galdist = estimates.Galdist(1)
        zp_list = np.linspace(0.0, 1, 200)
        ilist = []
        for zp in zp_list:
            ilist.append(galdist.p_phot(z_photo=zp, z=0.5))
        npt.assert_allclose(integrate.trapz(ilist, zp_list), 1.0,
                            err_msg='Photo-z PDF not correctly normalised')

    def test_fin_nz(self):
        galdist = estimates.Galdist(1)
        nz1 = galdist.n_i(0.2)
        nz2 = galdist.n_i(0.3)
        nz3 = galdist.n_i(0.4)
        npt.assert_allclose([nz1, nz2, nz3],
                            self.nfinchecks,
                            err_msg='ISTF n(z) producing incorrect values')

    def test_custom_nz_exp(self):
        npt.assert_raises(Exception, estimates.Galdist,
                          1, 'custom')

    def test_custom_no_bcols(self):
        npt.assert_raises(Exception, estimates.Galdist,
                          1, 'custom',
                          'test_fname')

    def test_custom_wrong_bcols(self):
        npt.assert_raises(ValueError, estimates.Galdist,
                          0, 'custom',
                          'test_fname')

    def test_nz_choice(self):
        npt.assert_raises(Exception, estimates.Galdist,
                          1, 'yeet')

    def test_nz_value_err(self):
        galdist = estimates.Galdist(1)
        npt.assert_raises(ValueError, galdist.n_i, 4.2)
