"""UNIT TESTS FOR NONLINEAR

This module contains unit tests for the
:obj:`non_linear` module.

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from cloe.non_linear.nonlinear import Nonlinear
from cloe.tests.test_tools.test_data_handler import load_test_pickle


class nonlinearinitTestCase(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # Load cosmology dictionary for tests with NL_flag_phot_matter=1
        cosmo_dic_1 = load_test_pickle('cosmo_test_NLphot1_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl1 = Nonlinear(cosmo_dic_1)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=2
        cosmo_dic_2 = load_test_pickle('cosmo_test_NLphot2_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl2 = Nonlinear(cosmo_dic_2)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=3
        cosmo_dic_3 = load_test_pickle('cosmo_test_NLphot3_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl3 = Nonlinear(cosmo_dic_3)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode2020)
        cosmo_dic_4 = cosmo_dic_3.copy()
        cosmo_dic_4['NL_flag_phot_matter'] = 4
        # Create instance of Nonlinear class and update its dictionary
        cls.nl4 = Nonlinear(cosmo_dic_4)
        cls.nl4.set_Pgg_spectro_model()
        cls.nl4.update_dic(cosmo_dic_4)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode2020)
        cosmo_dic_5 = cosmo_dic_3.copy()
        cosmo_dic_5['NL_flag_phot_matter'] = 5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl5 = Nonlinear(cosmo_dic_5)
        cls.nl5.set_Pgg_spectro_model()
        cls.nl5.update_dic(cosmo_dic_5)

        # Load cosmology dictionary for cosmology extrapolation tests
        cosmo_dic_extra = \
            load_test_pickle('cosmo_test_NLphot3_extra_dic.pickle')
        cosmo_dic_extra['NL_flag_phot_matter'] = 4
        # Create instance of Nonlinear class and update its dictionary
        cls.nl4_extra = Nonlinear(cosmo_dic_extra)
        cls.nl4_extra.set_Pgg_spectro_model()
        cls.nl4_extra.update_dic(cosmo_dic_extra)

        cosmo_dic_extra['NL_flag_phot_matter'] = 5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl5_extra = Nonlinear(cosmo_dic_extra)
        cls.nl5_extra.set_Pgg_spectro_model()
        cls.nl5_extra.update_dic(cosmo_dic_extra)

        # Load cosmology dictionary for tests
        cosmo_dic_7 = load_test_pickle('cosmo_test_NLspectro1_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl7 = Nonlinear(cosmo_dic_7)
        cls.nl7.set_Pgg_spectro_model()
        nonlinear_dic = \
            load_test_pickle('cosmo_test_NLspectro1_nonlin_dic.pickle')
        cls.nl7.Pgg_spectro_model.nonlinear_dic = nonlinear_dic

        cosmo_dic_8 = deepcopy(cosmo_dic_7)
        cosmo_dic_8['nuisance_parameters']['Psn_spectro_bin1'] = 1000.0
        cosmo_dic_8['nuisance_parameters']['aP_spectro_bin1'] = 1.0
        cosmo_dic_8['nuisance_parameters']['e0k2_spectro_bin1'] = 1.0
        cosmo_dic_8['nuisance_parameters']['e2k2_spectro_bin1'] = 1.0
        cls.nl8 = Nonlinear(cosmo_dic_8)
        cls.nl8.set_Pgg_spectro_model()
        cls.nl8.Pgg_spectro_model.nonlinear_dic = nonlinear_dic

        # Load cosmology dictionary for tests with NL_flag_phot_baryon=1
        cosmo_dic_2_bar1 = load_test_pickle('cosmo_test_NLBar1_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl2_b1 = Nonlinear(cosmo_dic_2_bar1)

        # Load cosmology dictionary for tests with NL_flag_phot_baryon=2
        cosmo_dic_3_bar2 = load_test_pickle('cosmo_test_NLBar2_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl3_b2 = Nonlinear(cosmo_dic_3_bar2)

        cosmo_dic_2_bar2 = cosmo_dic_3_bar2.copy()

        cosmo_dic_2_bar2['NL_flag_phot_matter'] = 2
        # Create instance of Nonlinear class and update its dictionary
        cls.nl2_b2 = Nonlinear(cosmo_dic_2_bar2)
        cls.nl2_b2.set_Pgg_spectro_model()
        cls.nl2_b2.update_dic(cosmo_dic_2_bar2)

        cosmo_dic_4_bar2 = cosmo_dic_3_bar2.copy()

        cosmo_dic_4_bar2['NL_flag_phot_matter'] = 4
        # Create instance of Nonlinear class and update its dictionary
        cls.nl4_b2 = Nonlinear(cosmo_dic_4_bar2)
        cls.nl4_b2.set_Pgg_spectro_model()
        cls.nl4_b2.update_dic(cosmo_dic_4_bar2)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode2020)
        cosmo_dic_3_bar = cosmo_dic_3.copy()
        cosmo_dic_3_bar['Baryon_redshift_model'] = False

        cosmo_dic_3_bar['NL_flag_phot_baryon'] = 3
        # Create instance of Nonlinear class and update its dictionary
        cls.nl3_b3 = Nonlinear(cosmo_dic_3_bar)
        cls.nl3_b3.set_Pgg_spectro_model()
        cls.nl3_b3.update_dic(cosmo_dic_3_bar)

        cosmo_dic_3_bar['NL_flag_phot_baryon'] = 4
        # Create instance of Nonlinear class and update its dictionary
        cls.nl3_b4 = Nonlinear(cosmo_dic_3_bar)
        cls.nl3_b4.set_Pgg_spectro_model()
        cls.nl3_b4.update_dic(cosmo_dic_3_bar)

        # Test also case with HMcode baryons but not HMcode matter
        # Copy cosmo_dic from NL_flag_phot_matter=1 (to have Halofit)
        cosmo_dic_1_bar = cosmo_dic_1.copy()
        cosmo_dic_1_bar['NL_flag_phot_matter'] = 1

        cosmo_dic_1_bar['NL_flag_phot_baryon'] = 2
        # Create instance of Nonlinear class and update its dictionary
        cls.nl1_b2 = Nonlinear(cosmo_dic_1_bar)
        cls.nl1_b2.set_Pgg_spectro_model()
        cls.nl1_b2.update_dic(cosmo_dic_1_bar)

        cosmo_dic_1_bar['NL_flag_phot_baryon'] = 1
        cosmo_dic_1_bar['nuisance_parameters']['HMCode_A_baryon'] = 3.01
        cosmo_dic_1_bar['nuisance_parameters']['HMCode_eta_baryon'] = 0.70
        # Create instance of Nonlinear class and update its dictionary
        cls.nl1_b1 = Nonlinear(cosmo_dic_1_bar)
        cls.nl1_b1.set_Pgg_spectro_model()
        cls.nl1_b1.update_dic(cosmo_dic_1_bar)

        # Now we activate the model with a power law in redshift to test that
        cosmo_dic_3_bar['Baryon_redshift_model'] = True

        cosmo_dic_3_bar['NL_flag_phot_baryon'] = 3
        # Create instance of Nonlinear class and update its dictionary
        cls.nl3_b3_z = Nonlinear(cosmo_dic_3_bar)
        cls.nl3_b3_z.set_Pgg_spectro_model()
        cls.nl3_b3_z.update_dic(cosmo_dic_3_bar)

        cosmo_dic_3_bar['NL_flag_phot_baryon'] = 4
        # Create instance of Nonlinear class and update its dictionary
        cls.nl3_b4_z = Nonlinear(cosmo_dic_3_bar)
        cls.nl3_b4_z.set_Pgg_spectro_model()
        cls.nl3_b4_z.update_dic(cosmo_dic_3_bar)

        # Load cosmology dictionary for cosmology extrapolation tests
        cosmo_dic_bar3_extra = \
            load_test_pickle('cosmo_test_NLphot3_extra_dic.pickle')
        cosmo_dic_bar3_extra['NL_flag_phot_matter'] = 3
        cosmo_dic_bar3_extra['Baryon_redshift_model'] = False
        cosmo_dic_bar3_extra['NL_flag_phot_baryon'] = 3
        cosmo_dic_bar3_extra['nuisance_parameters']['log10Mc_bcemu_bin1'] = \
            15.5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl3_extra_b3 = Nonlinear(cosmo_dic_bar3_extra)
        cls.nl3_extra_b3.set_Pgg_spectro_model()
        cls.nl3_extra_b3.update_dic(cosmo_dic_bar3_extra)

        # Load cosmology dictionary for cosmology extrapolation tests
        cosmo_dic_bar4_extra = \
            load_test_pickle('cosmo_test_NLphot3_extra_dic.pickle')
        cosmo_dic_bar4_extra['NL_flag_phot_matter'] = 3
        cosmo_dic_bar4_extra['Baryon_redshift_model'] = False
        cosmo_dic_bar4_extra['NL_flag_phot_baryon'] = 4
        cosmo_dic_bar4_extra['nuisance_parameters']['M_c_bacco_bin1'] = 15.5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl3_extra_b4 = Nonlinear(cosmo_dic_bar4_extra)
        cls.nl3_extra_b4.set_Pgg_spectro_model()
        cls.nl3_extra_b4.update_dic(cosmo_dic_bar4_extra)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=1
        # and IA_flag = 1
        cosmo_dic_1_tatt = \
            load_test_pickle('cosmo_test_NLphot1_tatt_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl1_tatt = Nonlinear(cosmo_dic_1_tatt)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=2
        # and IA_flag = 1
        cosmo_dic_2_tatt = \
            load_test_pickle('cosmo_test_NLphot2_tatt_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl2_tatt = Nonlinear(cosmo_dic_2_tatt)

        # Load cosmology dictionary for tests with NL_flag_phot_matter=3
        # and IA_flag = 1
        cosmo_dic_3_tatt = \
            load_test_pickle('cosmo_test_NLphot3_tatt_dic.pickle')
        # Create instance of Nonlinear class
        cls.nl3_tatt = Nonlinear(cosmo_dic_3_tatt)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode2020)
        # and IA_flag = 1
        cosmo_dic_4_tatt = cosmo_dic_3_tatt.copy()
        cosmo_dic_4_tatt['NL_flag_phot_matter'] = 4
        # Create instance of Nonlinear class and update its dictionary
        cls.nl4_tatt = Nonlinear(cosmo_dic_4_tatt)
        cls.nl4_tatt.set_Pgg_spectro_model()
        cls.nl4_tatt.update_dic(cosmo_dic_4_tatt)

        # Copy cosmo_dic from NL_flag_phot_matter=3 (to have HMcode2020)
        # and IA_flag = 1
        cosmo_dic_5_tatt = cosmo_dic_3_tatt.copy()
        cosmo_dic_5_tatt['NL_flag_phot_matter'] = 5
        # Create instance of Nonlinear class and update its dictionary
        cls.nl5_tatt = Nonlinear(cosmo_dic_5_tatt)
        cls.nl5_tatt.set_Pgg_spectro_model()
        cls.nl5_tatt.update_dic(cosmo_dic_5_tatt)

        # Copy cosmo_dic from NL_flag_phot_matter=1
        # (to test Pgg_phot_halo_NLbias)
        cosmo_dic_NLbias_halo = cosmo_dic_1.copy()
        cosmo_dic_NLbias_halo['NL_flag_phot_bias'] = 1
        # Set values for NL bias parameters
        nuis = cosmo_dic_NLbias_halo['nuisance_parameters']
        for b in ['1', '2', 'G2', 'G3']:
            for i in range(4):
                nuis[f'b{b}_{i}_poly_photo'] = 1.0
        for b in ['2', 'G2', 'G3']:
            for i in range(10):
                nuis[f'b{b}_photo_bin{i+1}'] = nuis[f'b1_photo_bin{i+1}']
        # Create instance of Nonlinear class and update its dictionary
        cls.nl1_nlbias_halo = Nonlinear(cosmo_dic_NLbias_halo)
        cls.nl1_nlbias_halo.set_Pgg_spectro_model()
        cls.nl1_nlbias_halo.update_dic(cosmo_dic_NLbias_halo)

        # Copy cosmo_dic from NL_flag_phot_matter=4
        # (to test Pgg_phot_emu_NLbias)
        cosmo_dic_NLbias_emu = cosmo_dic_4.copy()
        cosmo_dic_NLbias_emu['NL_flag_phot_bias'] = 1
        # Set values for NL bias parameters
        nuis = cosmo_dic_NLbias_emu['nuisance_parameters']
        for b in ['1', '2', 'G2', 'G3']:
            for i in range(4):
                nuis[f'b{b}_{i}_poly_photo'] = 1.0
        for b in ['2', 'G2', 'G3']:
            for i in range(10):
                nuis[f'b{b}_photo_bin{i+1}'] = nuis[f'b1_photo_bin{i+1}']
        # Create instance of Nonlinear class and update its dictionary
        cls.nl1_nlbias_emu = Nonlinear(cosmo_dic_NLbias_emu)
        cls.nl1_nlbias_emu.set_Pgg_spectro_model()
        cls.nl1_nlbias_emu.update_dic(cosmo_dic_NLbias_emu)

    def setUp(self) -> None:
        # Check values
        self.Pgi_spectro_test = -388.28625
        self.Pgg_spectro_test = 82780.067618
        self.Pgdelta_spectro_test = 59890.445816
        self.noise_Pgg_spectro = 2000.125

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

        self.Pgg_phot_test_NL4 = 58057.58175505
        self.Pgdelta_phot_test_NL4 = 41059.35948637
        self.Pii_test_NL4 = np.array([2.40329487, 0.89325885, 0.33133042])
        self.Pdeltai_test_NL4 = -264.17195626
        self.Pgi_phot_test_NL4 = -373.53644919

        self.Pgg_phot_test_NL5 = 58273.69731319
        self.Pgdelta_phot_test_NL5 = 41212.25488618
        self.Pii_test_NL5 = np.array([2.41430454, 0.89225537, 0.32953085])
        self.Pdeltai_test_NL5 = -265.15569155
        self.Pgi_phot_test_NL5 = -374.92694385

        self.Pgg_phot_test_NLbias_halo = 57979.4343265
        self.Pgdelta_phot_test_NLbias_halo = 40482.2743869

        self.Pgg_phot_test_NLbias_emu = 57860.2730410
        self.Pgdelta_phot_test_NLbias_emu = 40398.0054589

        self.Pmm_phot_test_NL4_extra_k = 3.4178353
        self.Pmm_phot_test_NL5_extra_kz = 0.61634334

        self.Pii_test_NL1_tatt = \
            np.array([7.11749416, 6.52936089, 5.24846405])
        self.Pdeltai_test_NL1_tatt = -277.24173292
        self.Pgi_phot_test_NL1_tatt = -392.01731165

        self.Pii_test_NL2_tatt = \
            np.array([7.12113817, 6.52871957, 5.24909852])
        self.Pdeltai_test_NL2_tatt = -277.64232328
        self.Pgi_phot_test_NL2_tatt = -392.58369074

        self.Pii_test_NL3_tatt = \
            np.array([7.12611501, 6.5281627, 5.24907609])
        self.Pdeltai_test_NL3_tatt = -278.18941671
        self.Pgi_phot_test_NL3_tatt = -393.35722993

        self.Pii_test_NL4_tatt = \
            np.array([7.11287427, 6.52525458, 5.24379997])
        self.Pdeltai_test_NL4_tatt = -276.7730395
        self.Pgi_phot_test_NL4_tatt = -391.37356994

        self.Pii_test_NL5_tatt = \
            np.array([7.12395548, 6.52429, 5.24201532])
        self.Pdeltai_test_NL5_tatt = -277.99105237
        self.Pgi_phot_test_NL5_tatt = -393.09591271

        self.Pmm_phot_test_extra_cosmo = 4300.03416143

        self.Pmm_phot_test_extra_k_const = 2.75083865
        self.Pmm_phot_test_extra_k_power_law = 3.46471815
        self.Pmm_phot_test_extra_k_hm_simple = 3.41277902

        self.Pmm_phot_test_extra_z_const = 7898.45870884
        self.Pmm_phot_test_extra_z_power_law = 7902.06722255

        self.Pmm_phot_test_Bar4 = 58.89835816
        self.Pmm_phot_test_Bar3 = 57.67567192
        self.Pmm_phot_test_Bar2 = 59.71375045
        self.Pmm_phot_test_Bar1 = 61.57124527

        self.Pmm_phot_test_NL1_Bar1 = 62.99958622
        self.Pmm_phot_test_NL1_Bar2 = 60.99212985
        self.Pmm_phot_test_NL2_Bar2 = 59.60063105
        self.Pmm_phot_test_NL4_Bar2 = 61.65879822

        self.Pmm_phot_test_Bar4_z_model = 53.97835655
        self.Pmm_phot_test_Bar3_z_model = 55.16310211

        self.Pmm_phot_test_Bar4_extra_k = 2.62388035
        self.Pmm_phot_test_Bar3_extra_kz = 0.57554706

        self.Pmm_phot_test_Bar4_extra_cosmo = 79.78729328
        self.Pmm_phot_test_Bar3_extra_cosmo = 79.23000146

        self.Pmm_phot_test_Bar4_extra_cosmo_const = 78.83848354
        self.Pmm_phot_test_Bar3_extra_cosmo_const = 78.50482874

        self.rtol = 1e-3
        self.redshift1 = 1.0
        self.redshift2 = 0.5
        self.redshift3 = 2.0
        self.wavenumber1 = 0.01
        self.wavenumber2 = 0.05
        self.wavenumber3 = 0.1
        self.redshift_extra = 3.0
        self.wavenumber_extra = 10.0
        self.mu = 0.5
        self.arrsize = 3
        self.wavenumber_b = 2.0

        self.D_test = 0.6061337447181263
        self.fia_test = -0.00909382071134854
        self.bspec = 1.46148
        self.bphot = 1.41406
        self.bphot_low = 1.09977
        self.bphot_high = 1.74299

    def tearDown(self):

        self.Pgg_spectro_test = None
        self.Pgdelta_spectro_test = None
        self.Pgi_spectro_test = None
        self.noise_Pgg_spectro = None

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

        self.Pii_test_NL1_tatt = None
        self.Pdeltai_test_NL1_tatt = None
        self.Pgi_phot_test_NL1_tatt = None

        self.Pii_test_NL2_tatt = None
        self.Pdeltai_test_NL2_tatt = None
        self.Pgi_phot_test_NL2_tatt = None

        self.Pii_test_NL3_tatt = None
        self.Pdeltai_test_NL3_tatt = None
        self.Pgi_phot_test_NL3_tatt = None

        self.Pii_test_NL4_tatt = None
        self.Pdeltai_test_NL4_tatt = None
        self.Pgi_phot_test_NL4_tatt = None

        self.Pii_test_NL5_tatt = None
        self.Pdeltai_test_NL5_tatt = None
        self.Pgi_phot_test_NL5_tatt = None

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
            self.nl1_nlbias_halo.Pgg_phot_def(self.redshift1,
                                              self.wavenumber1),
            self.Pgg_phot_test_NLbias_halo,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_bias=1 and NL_flag_phot_matter=1',
        )

        npt.assert_allclose(
            self.nl1_nlbias_emu.Pgg_phot_def(self.redshift1,
                                             self.wavenumber1),
            self.Pgg_phot_test_NLbias_emu,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgg_phot_def '
                    'for NL_flag_phot_bias=1 and NL_flag_phot_matter=4',
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
            self.nl1_nlbias_halo.Pgdelta_phot_def(self.redshift1,
                                                  self.wavenumber1),
            self.Pgdelta_phot_test_NLbias_halo,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_bias=1 and NL_flag_phot_matter=1',
        )

        npt.assert_allclose(
            self.nl1_nlbias_emu.Pgdelta_phot_def(self.redshift1,
                                                 self.wavenumber1),
            self.Pgdelta_phot_test_NLbias_emu,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgdelta_phot_def '
                    'for NL_flag_phot_bias=1 and NL_flag_phot_matter=4',
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

    def test_noise_Pgg_spectro(self):
        npt.assert_allclose(
            self.nl8.noise_Pgg_spectro(self.redshift1, self.wavenumber1,
                                       self.mu),
            self.noise_Pgg_spectro, rtol=self.rtol,
            err_msg='Error in value returned by noise_Pgg_spectro'
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

        test_p1_tatt = self.nl1_tatt.Pii_def(self.redshift1,
                                             [self.wavenumber1,
                                              self.wavenumber2,
                                              self.wavenumber3])

        type_check = isinstance(test_p1_tatt, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=1 and IA_flag=1'

        assert test_p1_tatt.size == self.arrsize, (
            'Error in size of array returned by Pii_def for '
            'NL_flag_phot_matter=1 and IA_flag=1'
        )

        npt.assert_allclose(
            test_p1_tatt,
            self.Pii_test_NL1_tatt,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=1 and IA_flag=1',
        )

        test_p2_tatt = self.nl2_tatt.Pii_def(self.redshift1,
                                             [self.wavenumber1,
                                              self.wavenumber2,
                                              self.wavenumber3])

        type_check = isinstance(test_p2_tatt, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=2 and IA_flag=1'

        assert test_p2_tatt.size == self.arrsize, (
            'Error in size of array returned by Pii_def for '
            'NL_flag_phot_matter=2 and IA_flag=1'
        )

        npt.assert_allclose(
            test_p2_tatt,
            self.Pii_test_NL2_tatt,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=2 and IA_flag=1',
        )

        test_p3_tatt = self.nl3_tatt.Pii_def(self.redshift1,
                                             [self.wavenumber1,
                                              self.wavenumber2,
                                              self.wavenumber3])

        type_check = isinstance(test_p3_tatt, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=3 and IA_flag=1'

        assert test_p3_tatt.size == self.arrsize, 'Error in size of array ' \
                                                  'returned by Pii_def ' \
                                                  'for NL_flag_phot_matter=3' \
                                                  'and IA_flag=1'

        npt.assert_allclose(
            test_p3_tatt,
            self.Pii_test_NL3_tatt,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=3 and IA_flag=1')

        test_p4_tatt = self.nl4_tatt.Pii_def(self.redshift1,
                                             [self.wavenumber1,
                                              self.wavenumber2,
                                              self.wavenumber3])

        type_check = isinstance(test_p4_tatt, np.ndarray)

        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=4 and IA_flag=1'

        assert test_p4_tatt.size == self.arrsize, (
            'Error in size of array returned by Pii_def for '
            'NL_flag_phot_matter=4 and IA_flag=1'
        )

        npt.assert_allclose(
            test_p4_tatt,
            self.Pii_test_NL4_tatt,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=4 and IA_flag=1',
        )

        test_p5_tatt = self.nl5_tatt.Pii_def(self.redshift1,
                                             [self.wavenumber1,
                                              self.wavenumber2,
                                              self.wavenumber3])

        type_check = isinstance(test_p5_tatt, np.ndarray)
        assert type_check, 'Error in returned data type of Pii_def ' \
                           'for NL_flag_phot_matter=5 and IA_flag=1'

        assert test_p5_tatt.size == self.arrsize, 'Error in size of array ' \
                                                  'returned by Pii_def ' \
                                                  'for NL_flag_phot_matter=5' \
                                                  'and IA_flag=1'

        npt.assert_allclose(
            test_p5_tatt,
            self.Pii_test_NL5_tatt,
            rtol=self.rtol,
            err_msg='Error in values returned by Pii_def for '
                    'NL_flag_phot_matter=5 and IA_flag=1')

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
            self.nl1_tatt.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL1_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=1 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl2_tatt.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL2_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=2 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl3_tatt.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL3_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=3 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl4_tatt.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL4_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=4 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl5_tatt.Pdeltai_def(self.redshift1, self.wavenumber1),
            self.Pdeltai_test_NL5_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pdeltai_def '
                    'for NL_flag_phot_matter=5 and IA_flag=1',
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
            self.nl1_tatt.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL1_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=1 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl2_tatt.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL2_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=2 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl3_tatt.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL3_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=3 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl4_tatt.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL4_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=4 and IA_flag=1',
        )

        npt.assert_allclose(
            self.nl5_tatt.Pgi_phot_def(self.redshift1, self.wavenumber1),
            self.Pgi_phot_test_NL5_tatt,
            rtol=self.rtol,
            err_msg='Error in value returned by Pgi_phot_def '
                    'for NL_flag_phot_matter=5 and IA_flag=1',
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

    def test_create_phot_galbias_nl(self):
        self.nl1_nlbias_halo.create_phot_galbias_nl(model=1)
        bi_vals = [
            self.nl1_nlbias_halo.theory['b1_inter'](self.redshift1),
            self.nl1_nlbias_halo.theory['b2_inter'](self.redshift1),
            self.nl1_nlbias_halo.theory['bG2_inter'](self.redshift1),
            self.nl1_nlbias_halo.theory['bG3_inter'](self.redshift1)]
        for bi in bi_vals:
            npt.assert_allclose(
                bi,
                self.bphot,
                rtol=self.rtol,
                err_msg='Error in create_phot_galbias_nl with bias_model=1',
            )
        # check redshift input below zs edges: returns b(z) at edge
        bi_vals = [
            self.nl1_nlbias_halo.theory['b1_inter'](0.1),
            self.nl1_nlbias_halo.theory['b2_inter'](0.1),
            self.nl1_nlbias_halo.theory['bG2_inter'](0.1),
            self.nl1_nlbias_halo.theory['bG3_inter'](0.1)]
        for bi in bi_vals:
            npt.assert_allclose(
                bi,
                self.bphot_low,
                rtol=self.rtol,
                err_msg='Error in create_phot_galbias_nl with bias_model=1',
            )
        # check redshift input above zs edges: returns b(z) at edge
        bi_vals = [
            self.nl1_nlbias_halo.theory['b1_inter'](3.0),
            self.nl1_nlbias_halo.theory['b2_inter'](3.0),
            self.nl1_nlbias_halo.theory['bG2_inter'](3.0),
            self.nl1_nlbias_halo.theory['bG3_inter'](3.0)]
        for bi in bi_vals:
            npt.assert_allclose(
                bi,
                self.bphot_high,
                rtol=self.rtol,
                err_msg='Error in create_phot_galbias_nl with bias_model=1',
            )
        # check vector redshift input: output size and values
        zs_vec = [self.redshift1, self.redshift1]
        bi_vals = [
            self.nl1_nlbias_halo.theory['b1_inter'](zs_vec),
            self.nl1_nlbias_halo.theory['b2_inter'](zs_vec),
            self.nl1_nlbias_halo.theory['bG2_inter'](zs_vec),
            self.nl1_nlbias_halo.theory['bG3_inter'](zs_vec)]
        for bi in bi_vals:
            npt.assert_equal(
                len(bi),
                len(zs_vec),
                err_msg=(
                    'Output size of create_phot_galbias_nl'
                    'interpolators (array) does not match'
                    'with input'
                ),
            )
            npt.assert_allclose(
                bi,
                desired=[self.bphot, self.bphot],
                rtol=1e-3,
                err_msg='Array output create_phot_galbias_nl interpolators',
            )

    def test_poly_phot_galbias(self):
        for b in ['1', '2', 'G2', 'G3']:
            b_val =\
                self.nl1_nlbias_halo.poly_phot_galbias_nl(f'b{b}')(1.0)
            z_arr = np.array([1.0, 2.0])
            b_val_arr =\
                self.nl1_nlbias_halo.poly_phot_galbias_nl(f'b{b}')(z_arr)
            # check float redshift input
            npt.assert_almost_equal(
                b_val,
                desired=4.0,
                decimal=8,
                err_msg='Error in poly_phot_galbias_nl float redshift case'
            )
            # check array redshift input
            npt.assert_allclose(
                b_val_arr,
                desired=[4.0, 15.0],
                rtol=1e-8,
                err_msg='Error in poly_phot_galbias_nl redshift array case'
            )

    def test_extrapolation_Pmm_phot_def(self):
        # Test for wavenumber extrapolation with EE2
        npt.assert_allclose(
            self.nl4.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_NL4_extra_k,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def for EE2',
        )

        # Test for wavenumber and redshift extrapolation with BACCO
        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift_extra, self.wavenumber_extra),
            self.Pmm_phot_test_NL5_extra_kz,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber and redshift ' +
                    'extrapolation of Pmm_phot_def for BACCO',
        )

        # Test for cosmology extrapolation with EE2
        npt.assert_allclose(
            self.nl4_extra.Pmm_phot_def(self.redshift1, self.wavenumber3),
            self.Pmm_phot_test_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for EE2',
        )

        # Test for cosmology extrapolation with BACCO
        npt.assert_allclose(
            self.nl5_extra.Pmm_phot_def(self.redshift1, self.wavenumber3),
            self.Pmm_phot_test_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for BACCO',
        )

    def test_extrapolation_options(self):

        # Test alternative wavenumber extrapolation options
        self.nl5.nonlinear_dic['option_extra_wavenumber'] = "const"
        self.nl5.update_dic(self.nl5.theory)

        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_extra_k_const,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def with constant for BACCO',
        )

        self.nl5.nonlinear_dic['option_extra_wavenumber'] = "power_law"
        self.nl5.update_dic(self.nl5.theory)

        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_extra_k_power_law,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def with power law for BACCO',
        )

        self.nl5.nonlinear_dic['option_extra_wavenumber'] = "hm_simple"
        self.nl5.update_dic(self.nl5.theory)

        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_extra_k_hm_simple,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def with HM code for BACCO',
        )

        # Test alternative redshift extrapolation options
        self.nl5.nonlinear_dic['option_extra_redshift'] = "const"
        self.nl5.update_dic(self.nl5.theory)

        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift_extra, self.wavenumber1),
            self.Pmm_phot_test_extra_z_const,
            rtol=self.rtol,
            err_msg='Error in value returned by redshift extrapolation of ' +
                    'Pmm_phot_def with constant for BACCO',
        )

        self.nl5.nonlinear_dic['option_extra_redshift'] = "power_law"
        self.nl5.update_dic(self.nl5.theory)

        npt.assert_allclose(
            self.nl5.Pmm_phot_def(self.redshift_extra, self.wavenumber1),
            self.Pmm_phot_test_extra_z_power_law,
            rtol=self.rtol,
            err_msg='Error in value returned by redshift extrapolation of ' +
                    'Pmm_phot_def with power law for BACCO',
        )

        # Test alternative cosmology extrapolation with BACCO
        self.nl5_extra.nonlinear_dic['option_extra_cosmo'] = "hm_simple"
        self.nl5_extra.update_dic(self.nl5_extra.theory)

        npt.assert_allclose(
            self.nl5_extra.Pmm_phot_def(self.redshift1, self.wavenumber3),
            self.Pmm_phot_test_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for BACCO with hm_simple',
        )

    def test_baryons_Pmm_phot_def(self):
        # Test for baryon model 4 (BACCO)
        npt.assert_allclose(
            self.nl3_b4.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar4,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for BACCO baryons',
        )

        # Test for baryon model 3 (BCemu)
        npt.assert_allclose(
            self.nl3_b3.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar3,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for BCemu baryons',
        )

        # Test for baryon model 2 (HMcode2020_feedback)
        npt.assert_allclose(
            self.nl3_b2.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar2,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for HMcode2020 baryons',
        )

        # Test for baryon model 1 (HMcode2016) with same matter model
        npt.assert_allclose(
            self.nl2_b1.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar1,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for HMcode2016 baryons',
        )

        # Test for baryon model 1 (HMcode2016) with halofit
        npt.assert_allclose(
            self.nl1_b1.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_NL1_Bar1,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for HMcode2016 baryons with Halofit matter',
        )

        # Test for baryon model 2 (HMcode2020_feedback) with halofit
        npt.assert_allclose(
            self.nl1_b2.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_NL1_Bar2,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for HMcode2020 baryons with Halofit matter',
        )

        # Test for baryon model 2 (HMcode2020_feedback) with HMcode2016 matter
        npt.assert_allclose(
            self.nl2_b2.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_NL2_Bar2,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for HMcode2020 baryons with HMcode2016' +
                    'matter',
        )

        # Test for baryon model 2 (HMcode2020_feedback) with EE2 matter
        npt.assert_allclose(
            self.nl4_b2.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_NL4_Bar2,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for HMcode2020 baryons with EE2' +
                    'matter',
        )

        # Test for baryon model 1 (BACCO) with z model
        npt.assert_allclose(
            self.nl3_b4_z.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar4_z_model,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for BACCO baryons with z model',
        )

        # Test for baryon model 2 (BCemu) with z model
        npt.assert_allclose(
            self.nl3_b3_z.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar3_z_model,
            rtol=self.rtol,
            err_msg='Error in value returned by ' +
                    'Pmm_phot_def for BCemu baryons with z model',
        )

    def test_baryon_extrapolation_Pmm_phot_def(self):
        # Test for wavenumber extrapolation with Bacco baryons
        npt.assert_allclose(
            self.nl3_b4.Pmm_phot_def(self.redshift1, self.wavenumber_extra),
            self.Pmm_phot_test_Bar4_extra_k,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber extrapolation of ' +
                    'Pmm_phot_def for BACCO baryons',
        )

        # Test for wavenumber and redshift extrapolation with BCemu
        npt.assert_allclose(
            self.nl3_b3.Pmm_phot_def(self.redshift_extra,
                                     self.wavenumber_extra),
            self.Pmm_phot_test_Bar3_extra_kz,
            rtol=self.rtol,
            err_msg='Error in value returned by wavenumber and redshift ' +
                    'extrapolation of Pmm_phot_def for BCemu baryons',
        )

        # Test for cosmology extrapolation for BACCO baryons
        npt.assert_allclose(
            self.nl3_extra_b4.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar4_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for BACCO baryons',
        )

        # Test for cosmology extrapolation with BCemu baryons
        npt.assert_allclose(
            self.nl3_extra_b3.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar3_extra_cosmo,
            rtol=self.rtol,
            err_msg='Error in value returned by cosmo extrapolation of ' +
                    'Pmm_phot_def for BCemu baryons',
        )

    def test_baryon_extrapolation_options(self):

        # Test alternative baryon extrapolation options
        self.nl3_extra_b4.nonlinear_dic['option_extra_bar'] = "const"
        self.nl3_extra_b4.update_dic(self.nl3_extra_b4.theory)

        npt.assert_allclose(
            self.nl3_extra_b4.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar4_extra_cosmo_const,
            rtol=self.rtol,
            err_msg='Error in value returned by baryon extrapolation of ' +
                    'Pmm_phot_def with constant for BACCO baryons',
        )

        self.nl3_extra_b3.nonlinear_dic['option_extra_bar'] = "const"
        self.nl3_extra_b3.update_dic(self.nl3_extra_b3.theory)

        npt.assert_allclose(
            self.nl3_extra_b3.Pmm_phot_def(self.redshift1, self.wavenumber_b),
            self.Pmm_phot_test_Bar3_extra_cosmo_const,
            rtol=self.rtol,
            err_msg='Error in value returned by baryon extrapolation of ' +
                    'Pmm_phot_def with constant for BCemu baryons',
        )
