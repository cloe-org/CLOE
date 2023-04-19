"""UNIT TESTS FOR NONLINEAR

This module contains unit tests for the nonlinear module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from cloe.non_linear.nonlinear import Nonlinear
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class nonlinearinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # Load cosmology dictionary for tests with NL_flag_phot_matter=1
        cosmo_dic_1 = load_test_pickle('cosmo_test_NLphot1_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl1 = Nonlinear(cosmo_dic_1)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=1
        cosmo_dic_2 = load_test_pickle('cosmo_test_NLphot2_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl2 = Nonlinear(cosmo_dic_2)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=3
        cosmo_dic_3 = load_test_pickle('cosmo_test_NLphot3_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl3 = Nonlinear(cosmo_dic_3)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=4
        cosmo_dic_4 = load_test_pickle('cosmo_test_NLphot4_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl4 = Nonlinear(cosmo_dic_4)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode)
        cosmo_dic_5 = cosmo_dic_3.copy()
        cosmo_dic_5['NL_flag_phot_matter'] = 5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl5 = Nonlinear(cosmo_dic_5)
        cls.nl5.set_Pgg_spectro_model()
        cls.nl5.update_dic(cosmo_dic_5)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode)
        cosmo_dic_6 = cosmo_dic_3.copy()
        cosmo_dic_6['NL_flag_phot_matter'] = 6
        # Create instance of Nonlinear class and update its dictionary
        cls.nl6 = Nonlinear(cosmo_dic_6)
        cls.nl6.set_Pgg_spectro_model()
        cls.nl6.update_dic(cosmo_dic_6)

        # Load cosmology dictionary for cosmology extrapolation tests
        cosmo_dic_extra = \
            load_test_pickle('cosmo_test_NLphot3_extra_dic.pickle')
        cosmo_dic_extra['NL_flag_phot_matter'] = 5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl5_extra = Nonlinear(cosmo_dic_extra)
        cls.nl5_extra.set_Pgg_spectro_model()
        cls.nl5_extra.update_dic(cosmo_dic_extra)

        cosmo_dic_extra['NL_flag_phot_matter'] = 6
        # Create instance of Nonlinear class and update its dictionary
        cls.nl6_extra = Nonlinear(cosmo_dic_extra)
        cls.nl6_extra.set_Pgg_spectro_model()
        cls.nl6_extra.update_dic(cosmo_dic_extra)

        # Load cosmology dictionary for tests
        cosmo_dic_7 = load_test_pickle('cosmo_test_NLspectro1_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl7 = Nonlinear(cosmo_dic_7)
        cls.nl7.set_Pgg_spectro_model()
        nonlinear_dic = \
            load_test_pickle('cosmo_test_NLspectro1_nonlin_dic.pickle')
        cls.nl7.Pgg_spectro_model.nonlinear_dic = nonlinear_dic

    def setUp(self) -> None:
        # Check values
        self.Pgi_spectro_test = -388.28625
        self.Pgg_spectro_test = 82595.002101
        self.Pgdelta_spectro_test = 59890.445816

        self.Pgg_phot_test_NL1 = 58175.37128267319
        self.Pgdelta_phot_test_NL1 = 41140.678806465
        self.Pii_test_NL1 = np.array([2.40814584, 0.89845222, 0.33413697])
        self.Pdeltai_test_NL1 = -264.69349205263046
        self.Pgi_phot_test_NL1 = -374.2923700580575

        self.Pgg_phot_test_NL2 = 58265.169583
        self.Pgdelta_phot_test_NL2 = 41204.182708
        self.Pii_test_NL2 = np.array([2.411818, 0.897909, 0.334765])
        self.Pdeltai_test_NL2 = -265.099577
        self.Pgi_phot_test_NL2 = -374.866598

        self.Pgg_phot_test_NL3 = 58385.00518538
        self.Pgdelta_phot_test_NL3 = 41290.92152593
        self.Pii_test_NL3 = np.array([2.41684859, 0.89712765, 0.33484379])
        self.Pdeltai_test_NL3 = -265.66180498
        self.Pgi_phot_test_NL3 = -375.64305879

        self.Pgg_phot_test_NL4 = 58361.236575
        self.Pgdelta_phot_test_NL4 = 41272.119726
        self.Pii_test_NL4 = np.array([2.415794, 0.896139, 0.333824])
        self.Pdeltai_test_NL4 = -265.53667
        self.Pgi_phot_test_NL4 = -375.484674

        self.Pgg_phot_test_NL5 = 58057.58175505
        self.Pgdelta_phot_test_NL5 = 41059.35948637
        self.Pii_test_NL5 = np.array([2.40329487, 0.89325885, 0.33133042])
        self.Pdeltai_test_NL5 = -264.17195626
        self.Pgi_phot_test_NL5 = -373.53644919

        self.Pgg_phot_test_NL6 = 58273.69731319
        self.Pgdelta_phot_test_NL6 = 41212.25488618
        self.Pii_test_NL6 = np.array([2.41173, 0.890966, 0.330161])
        self.Pdeltai_test_NL6 = -265.15569155
        self.Pgi_phot_test_NL6 = -374.92694385

        self.Pmm_phot_test_NL5_extra_k = 3.4178353
        self.Pmm_phot_test_NL6_extra_kz = 1.3728688

        self.Pmm_phot_test_extra_cosmo = 3722.815436285109

        self.Pmm_phot_test_extra_k_const = 1.75174
        self.Pmm_phot_test_extra_k_power_law = 4.39334023
        self.Pmm_phot_test_extra_k_hm_simple = 3.41310076

        self.Pmm_phot_test_extra_z_const = 13776.70636963
        self.Pmm_phot_test_extra_z_power_law = 13729.49552857

        self.rtol = 1e-3
        self.redshift1 = 1.0
        self.redshift2 = 0.5
        self.redshift3 = 2.0
        self.wavenumber1 = 0.01
        self.wavenumber2 = 0.05
        self.wavenumber3 = 0.1
        self.wavenumber_extra = 10.0
        self.mu = 0.5
        self.arrsize = 3

        self.D_test = 0.6061337447181263
        self.fia_test = -0.00909382071134854
        self.bspec = 1.46148
        self.bphot = 1.41406

    def tearDown(self):

        self.Pgg_spectro_test = None
        self.Pgdelta_spectro_test = None
        self.Pgi_spectro_test = None

        self.Pgg_phot_test_NL1 = None
        self.Pgdelta_phot_test_NL1 = None
        self.Pii_test_NL1 = None
        self.Pdeltai_test_NL1 = None
        self.Pgi_phot_test_NL1 = None

        self.Pgg_phot_test_NL2 = None
        self.Pgdelta_phot_test_NL2 = None
        self.Pii_test_NL2 = None
        self.Pdeltai_test_NL2 = None
        self.Pgi_phot_test_NL2 = None

        self.Pgg_phot_test_NL3 = None
        self.Pgdelta_phot_test_NL3 = None
        self.Pii_test_NL3 = None
        self.Pdeltai_test_NL3 = None
        self.Pgi_phot_test_NL3 = None

        self.Pgg_phot_test_NL4 = None
        self.Pgdelta_phot_test_NL4 = None
        self.Pii_test_NL4 = None
        self.Pdeltai_test_NL4 = None
        self.Pgi_phot_test_NL4 = None

        self.Pgg_phot_test_NL5 = None
        self.Pgdelta_phot_test_NL5 = None
        self.Pii_test_NL5 = None
        self.Pdeltai_test_NL5 = None
        self.Pgi_phot_test_NL5 = None

        self.Pgg_phot_test_NL6 = None
        self.Pgdelta_phot_test_NL6 = None
        self.Pii_test_NL6 = None
        self.Pdeltai_test_NL6 = None
        self.Pgi_phot_test_NL6 = None

    def test_Pgg_phot_def(self):
        npt.assert_allclose(
            self.nl1.Pgg_phot_def(self.redshift1, self.wavenumber1),
            self.Pgg_phot_test_NL1,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_matter=1',
        )

        npt.assert_allclose(
            self.nl2.Pgg_phot_def(self.redshift1, self.wavenumber1),
            self.Pgg_phot_test_NL2,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_matter=2',
        )

        npt.assert_allclose(
            self.nl3.Pgg_phot_def(self.redshift1, self.wavenumber1),
            self.Pgg_phot_test_NL3,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_matter=3',
        )

        npt.assert_allclose(
            self.nl4.Pgg_phot_def(self.redshift1, self.wavenumber1),
            self.Pgg_phot_test_NL4,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_matter=4',
        )

        npt.assert_allclose(
            self.nl5.Pgg_phot_def(self.redshift1, self.wavenumber1),
            self.Pgg_phot_test_NL5,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_matter=5',
        )

        npt.assert_allclose(
            self.nl6.Pgg_phot_def(self.redshift1, self.wavenumber1),
            self.Pgg_phot_test_NL6,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_matter=6',
        )

    def test_Pgdelta_phot_def(self):
        npt.assert_allclose(
            self.nl1.Pgdelta_phot_def(self.redshift1, self.wavenumber1),
            self.Pgdelta_phot_test_NL1,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_matter=1'
        )

        npt.assert_allclose(
            self.nl2.Pgdelta_phot_def(self.redshift1, self.wavenumber1),
            self.Pgdelta_phot_test_NL2,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_matter=2'
        )

        npt.assert_allclose(
            self.nl3.Pgdelta_phot_def(self.redshift1, self.wavenumber1),
            self.Pgdelta_phot_test_NL3,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_matter=3',
        )

        npt.assert_allclose(
            self.nl4.Pgdelta_phot_def(self.redshift1, self.wavenumber1),
            self.Pgdelta_phot_test_NL4,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_matter=4'
        )

        npt.assert_allclose(
            self.nl5.Pgdelta_phot_def(self.redshift1, self.wavenumber1),
            self.Pgdelta_phot_test_NL5,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_matter=5',
        )

        npt.assert_allclose(
            self.nl6.Pgdelta_phot_def(self.redshift1, self.wavenumber1),
            self.Pgdelta_phot_test_NL6,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_matter=6',
        )

    def test_Pgg_spectro_def(self):
        npt.assert_allclose(
            self.nl7.Pgg_spectro_def(self.redshift1, self.wavenumber1,
                                     self.mu),
            self.Pgg_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_spectro_def '
                    'for NL_flag_spectro=1',
        )

    def test_Pgdelta_spectro_def(self):
        npt.assert_allclose(
            self.nl7.Pgdelta_spectro_def(self.redshift1, self.wavenumber1,
                                         self.mu),
            self.Pgdelta_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_spectro_def '
                    'for NL_flag_spectro=1',
        )

    def test_Pii_def(self):
        test_p1 = self.nl1.Pii_def(self.redshift1,
                                   [self.wavenumber1,
                                    self.wavenumber2,
                                    self.wavenumber3])

        type_check = isinstance(test_p1, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=1'

        assert test_p1.size == self.arrsize, (
            'Error in size of array returned by Pii_def for '
            'NL_flag_phot_matter=1'
        )

        npt.assert_allclose(
            test_p1,
            self.Pii_test_NL1,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=1',
        )

        test_p2 = self.nl2.Pii_def(self.redshift1,
                                   [self.wavenumber1,
                                    self.wavenumber2,
                                    self.wavenumber3])

        type_check = isinstance(test_p2, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=2'

        assert test_p2.size == self.arrsize, (
            'Error in size of array returned by Pii_def for '
            'NL_flag_phot_matter=2'
        )

        npt.assert_allclose(
            test_p2,
            self.Pii_test_NL2,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=2',
        )

        test_p3 = self.nl3.Pii_def(self.redshift1,
                                   [self.wavenumber1,
                                    self.wavenumber2,
                                    self.wavenumber3])

        type_check = isinstance(test_p3, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=3'

        assert test_p3.size == self.arrsize, 'Error in size of array ' \
                                             'returned by Pii_def ' \
                                             'for NL_flag_phot_matter=3'

        npt.assert_allclose(
            test_p3,
            self.Pii_test_NL3,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=3')

        test_p4 = self.nl4.Pii_def(self.redshift1,
                                   [self.wavenumber1,
                                    self.wavenumber2,
                                    self.wavenumber3])

        type_check = isinstance(test_p4, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=4'

        assert test_p4.size == self.arrsize, (
            'Error in size of array returned by Pii_def for '
            'NL_flag_phot_matter=4'
        )

        npt.assert_allclose(
            test_p4,
            self.Pii_test_NL4,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=4',
        )

        test_p5 = self.nl5.Pii_def(self.redshift1,
                                   [self.wavenumber1,
                                    self.wavenumber2,
                                    self.wavenumber3])

        type_check = isinstance(test_p5, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=5'

        assert test_p5.size == self.arrsize, 'Error in size of array ' \
                                             'returned by Pii_def ' \
                                             'for NL_flag_phot_matter=5'

        npt.assert_allclose(
            test_p5,
            self.Pii_test_NL5,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=5')

        test_p6 = self.nl6.Pii_def(self.redshift1,
                                   [self.wavenumber1,
                                    self.wavenumber2,
                                    self.wavenumber3])

        type_check = isinstance(test_p6, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=6'

        assert test_p6.size == self.arrsize, 'Error in size of array ' \
                                             'returned by Pii_def ' \
                                             'for NL_flag_phot_matter=6'

        npt.assert_allclose(
            test_p6,
            self.Pii_test_NL6,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=6')

    def test_Pdeltai_def(self):
        npt.assert_allclose(
            self.nl1.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL1,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=1',
        )

        npt.assert_allclose(
            self.nl2.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL2,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=2',
        )

        npt.assert_allclose(
            self.nl3.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL3,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=3',
        )

        npt.assert_allclose(
            self.nl4.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL4,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=4',
        )

        npt.assert_allclose(
            self.nl5.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL5,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=5',
        )

        npt.assert_allclose(
            self.nl6.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL6,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=6',
        )

    def test_Pgi_phot_def(self):
        npt.assert_allclose(
            self.nl1.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL1,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=1',
        )

        npt.assert_allclose(
            self.nl2.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL2,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=2',
        )

        npt.assert_allclose(
            self.nl3.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL3,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=3',
        )

        npt.assert_allclose(
            self.nl4.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL4,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=4',
        )

        npt.assert_allclose(
            self.nl5.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL5,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=5',
        )

        npt.assert_allclose(
            self.nl6.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL6,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=6',
        )

    def test_Pgi_spectro_def(self):
        npt.assert_allclose(
            self.nl1.Pgi_spectro_def(self.redshift1, self.wavenumber1),
            self.Pgi_spectro_test,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_spectro_def',
        )

    def test_fia(self):
        npt.assert_allclose(
            self.nl1.misc.fia(self.redshift1, self.wavenumber1),
            self.fia_test,
            rtol=self.rtol,
            err_msg='Error in value returned by fia',
        )

    def test_fia_k_array(self):
        wavenumber_array = np.array([self.wavenumber1, self.wavenumber2,
                                     self.wavenumber3])
        fia = self.nl1.misc.fia(self.redshift1, wavenumber_array)
        npt.assert_equal(
            isinstance(fia, np.ndarray),
            True,
            err_msg=(
                'Error in type returned by fia when redshift is scalar and ' +
                'wavenumber is array'
            ),
        )
        npt.assert_equal(
            np.size(fia),
            np.size(wavenumber_array),
            err_msg=(
                'Error in size returned by fia when redshift is scalar and ' +
                'wavenumber is array'
            ),
        )

    def test_fia_z_array(self):
        redshift_array = np.array([self.redshift1, self.redshift1])
        fia = self.nl1.misc.fia(redshift_array, self.wavenumber1)
        npt.assert_equal(
            isinstance(fia, np.ndarray),
            True,
            err_msg=(
                'Error in type returned by fia when redshift is array and ' +
                'wavenumber is scalar'
            ),
        )
        npt.assert_equal(
            np.size(fia),
            np.size(redshift_array),
            err_msg=(
                'Error in size returned by fia when redshift is array and ' +
                'wavenumber is scalar'
            ),
        )

    def test_fia_zk_array(self):
        redshift_array = np.array([self.redshift1, self.redshift1])
        wavenumber_array = np.array([self.wavenumber1, self.wavenumber2,
                                     self.wavenumber3])
        fia = self.nl1.misc.fia(redshift_array, wavenumber_array)
        npt.assert_equal(
            isinstance(fia, np.ndarray),
            True,
            err_msg=(
                'Error in type returned by fia when redshift is array and ' +
                'wavenumber is array'
            ),
        )
        npt.assert_equal(
            np.size(fia),
            np.size(redshift_array) * np.size(wavenumber_array),
            err_msg=(
                'Error in size returned by fia when redshift is array and ' +
                'wavenumber is array'
            ),
        )
        npt.assert_array_equal(
            np.shape(fia),
            (2, 3),
            err_msg=(
                'Error in size returned by fia when redshift is array and ' +
                'wavenumber is array'
            ),
        )

    def test_istf_spectro_galbias(self):
        npt.assert_allclose(
            self.nl1.misc.istf_spectro_galbias(self.redshift1),
            self.bspec,
            rtol=self.rtol,
            err_msg='Error in istf_spectro_galbias',
        )
        self.assertRaises(
            ValueError,
            self.nl1.misc.istf_spectro_galbias,
            self.redshift2,
        )
        self.assertRaises(
            ValueError,
            self.nl1.misc.istf_spectro_galbias,
            self.redshift3,
        )

    def test_istf_phot_galbias(self):
        npt.assert_allclose(
            self.nl1.misc.istf_phot_galbias(self.redshift1),
            self.bphot,
            rtol=self.rtol,
            err_msg='Error in istf_phot_galbias',
        )

    def test_extrapolation_Pmm_phot_def(self):
        # Test for wavenumber extrapolation with EE2
        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_NL5_extra_k,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def for EE2',
        )

        # Test for wavenumber and redshift extrapolation with BACCO
        npt.assert_allclose(
            self.nl6.Pmm_phot_def(self.redshift3, self.wavenumber_extra),
            self.Pmm_phot_test_NL6_extra_kz,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber and redshift ' +
                    'extrapolation of Pmm_phot_def for BACCO',
        )

        # Test for cosmology extrapolation with EE2
        npt.assert_allclose(
            self.nl5_extra.Pmm_phot_def(self.redshift1, self.wavenumber3),
            self.Pmm_phot_test_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for EE2',
        )

        # Test for cosmology extrapolation with BACCO
        npt.assert_allclose(
            self.nl6_extra.Pmm_phot_def(self.redshift1, self.wavenumber3),
            self.Pmm_phot_test_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for BACCO',
        )

    def test_extrapolation_options(self):

        # Test alternative wavenumber extrapolation options
        self.nl6.nonlinear_dic['option_extra_wavenumber'] = "const"
        self.nl6.update_dic(self.nl6.theory)

        npt.assert_allclose(
            self.nl6.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_extra_k_const,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def with constant for BACCO',
        )

        self.nl6.nonlinear_dic['option_extra_wavenumber'] = "power_law"
        self.nl6.update_dic(self.nl6.theory)

        npt.assert_allclose(
            self.nl6.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_extra_k_power_law,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def with power law for BACCO',
        )

        self.nl6.nonlinear_dic['option_extra_wavenumber'] = "hm_simple"
        self.nl6.update_dic(self.nl6.theory)

        npt.assert_allclose(
            self.nl6.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_extra_k_hm_simple,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def with HM code for BACCO',
        )

        # Test alternative redshift extrapolation options
        self.nl6.nonlinear_dic['option_extra_redshift'] = "const"
        self.nl6.update_dic(self.nl6.theory)

        npt.assert_allclose(
            self.nl6.Pmm_phot_def(self.redshift3, self.wavenumber1),
            self.Pmm_phot_test_extra_z_const,
            rtol=self.rtol,
            err_msg='Error in value returned by redshift extrapolation of ' +
                    'Pmm_phot_def with constant for BACCO',
        )

        self.nl6.nonlinear_dic['option_extra_redshift'] = "power_law"
        self.nl6.update_dic(self.nl6.theory)

        npt.assert_allclose(
            self.nl6.Pmm_phot_def(self.redshift3, self.wavenumber1),
            self.Pmm_phot_test_extra_z_power_law,
            rtol=self.rtol,
            err_msg='Error in value returned by redshift extrapolation of ' +
                    'Pmm_phot_def with power law for BACCO',
        )
