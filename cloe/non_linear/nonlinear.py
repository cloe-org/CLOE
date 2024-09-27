"""NONLINEAR

This module to computes the nonlinear recipes for the
photometric and spectroscopic probes.
"""
import numpy as np
from scipy import interpolate
import cloe.auxiliary.redshift_bins as rb
from cloe.non_linear.miscellanous import Misc
from cloe.non_linear.pgg_spectro import Pgg_spectro_model
from cloe.non_linear.pgg_phot import Pgg_phot_model
from cloe.non_linear.pgL_phot import PgL_phot_model
from cloe.non_linear.pLL_phot import PLL_phot_model
from cloe.non_linear.eft import EFTofLSS
import pyhmcode as hmcode
import copy
from pathlib import Path
import sys

parent_path = str(Path(Path(__file__).resolve().parents[0]))
sys.path.insert(0, parent_path)


class Nonlinear:
    """
    Class to compute nonlinear recipes.

    The current available options, depending on the value of the nonlinear
    flags are:

    NL_flag_phot_matter
    0: linear-only (from Cosmology class)
    1: Takahashi
    2: Mead2016
    3: Mead2020 (w/o baryon corrections)
    4: Euclid Emulator 2
    5: Bacco (matter)

    NL_flag_phot_baryon
    0: no baryons
    1: Mead2016
    2: Mead2020_feedback
    3: BCemu
    4: Bacco (baryons)

    NL_flag_phot_bias
    0: linear bias only (b_1)
    1: non-linear bias with loop calculations from linear power spectrum

    NL_flag_spectro
    0: linear-only (from Cosmology class)
    1: EFT
    """

    def __init__(self, cosmo_dic):
        """Class constructor.

        Initialises class and nonlinear code.

        Parameters
        ----------
        cosmo_dic: dict
            External dictionary from Cosmology class
        redshift_bins: list or numpy.ndarray
            Spectroscopic redshift bins to determine which bin the input
            redshift corresponds to
        """
        self.theory = cosmo_dic
        self.nuis = cosmo_dic['nuisance_parameters']

        # Add Omb
        self.theory['Omb'] = \
            self.theory['ombh2'] / (self.theory['H0'] / 100) ** 2

        self.misc = Misc(cosmo_dic)

        # Nonlinear dictionary, to store intermediate quantities
        # not necessary to the rest of the code
        self.nonlinear_dic = {'NL_boost': None,
                              'P_NL_extra': None,
                              'option_extra_wavenumber': 'hm_smooth',
                              'option_extra_redshift': 'hm_simple',
                              'Pk_mu': None,
                              'Pb1b2_kz': None,
                              'Pb1bG2_kz': None,
                              'Pb2b2_kz': None,
                              'Pb2bG2_kz': None,
                              'PbG2bG2_kz': None,
                              'PZ1bG3_kz': None,
                              'PZ1bG2_kz': None,
                              'option_extra_cosmo': 'hm_smooth',
                              'wavenumber_tanh_slope': 10.0,
                              'wavenumber_tanh_scale': 1.15,
                              'Pk_mu': None,
                              'Bar_boost': self.linear_boost,
                              'bcemu_par_inter': None,
                              'bacco_par_inter': None,
                              'option_extra_bar': 'lin'}

        # Empty variables for emulator class instances
        self.ee2 = None
        self.bemu = None
        self.bcemu = None

        # Matrix values that depend on matter and baryonic feedback flags
        # Matrix values for choosing the nonlinear model from cobaya (relevant
        # for extrapolations). This is the same matrix used in the function
        # set_halofit_version in likelihood_yaml_handler.py. See the
        # documentation for that function for more details.
        self.NL_Bar_matrix = np.array([[0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1],
                                       [2, 2, 4, 2, 2],
                                       [3, 2, 4, 3, 3],
                                       [3, 2, 4, 3, 3],
                                       [3, 2, 4, 3, 3]])

        # Matrix values for choosing between the halo and emu versions of
        # spectra for the observables. For each value of the NL and baryonic
        # feedback flags, this matrix determines which function to be used for
        # the observables, whether the _halo version (based on having a full
        # nonlinear spectrum from cobaya) or the _emu version (based on having
        # a boost instead of a full spectrum). The use of a boost for two
        # non-emulator combinations is due to the need to still compute a boost
        # when the HMcode versions for matter and baryon feedback do not match.
        self.halo_emu_matrix = np.array([[0, 0, 0, 0, 0],
                                         [1, 1, 1, 1, 1],
                                         [1, 1, 2, 1, 1],
                                         [1, 2, 1, 1, 1],
                                         [2, 2, 2, 2, 2],
                                         [2, 2, 2, 2, 2]])

        # Instances of classes to compute power spectra interpolators
        # for the 3x2pt statistics
        self.Pgg_phot_model = Pgg_phot_model(cosmo_dic, self.nonlinear_dic,
                                             self.misc)
        self.PgL_phot_model = PgL_phot_model(cosmo_dic, self.nonlinear_dic,
                                             self.misc)
        self.PLL_phot_model = PLL_phot_model(cosmo_dic, self.nonlinear_dic,
                                             self.misc)

        self.eftobj = None

    def set_Pgg_spectro_model(self):
        """Sets GCspectro redshift bins for ``Pgg_spectro_model``.

        Reads the values of the redshift bin edges from the :obj:`theory`
        class attribute and computes the centers of the bins that are
        later used for the GCspectro recipe. Also, creates an instance of
        the :obj:`Pgg_spectro_model class`, used to return the recipe for
        GCspectro.

        Raises
        ------
        KeyError
            If GCspectro redshift bins cannot be found in the cosmo
            dictionary
        """
        if 'redshift_bins_means_spectro' not in self.theory.keys():
            raise KeyError('Attempting to set Pgg_spectro_model class '
                           'without having specified the GCspectro '
                           'redshift bins in the cosmo dictionary.')
        zbins = self.theory['redshift_bins_means_spectro']
        if not isinstance(zbins, np.ndarray):
            self.zbins = np.array(zbins)
        else:
            self.zbins = zbins
        self.zmeans = (self.zbins[1:] + self.zbins[:-1]) / 2.0

        # Instance of class to compute power spectra interpolators
        # for the GCspectro statistics
        self.Pgg_spectro_model = Pgg_spectro_model(self.theory,
                                                   self.nonlinear_dic,
                                                   self.misc, self.zbins)

    def update_dic(self, cosmo_dic):
        """Updates Dic.

        Calls all routines updating the cosmo dictionary, and recomputes
        the various intermediate nonlinear blocks needed for the different
        recipes (according to the value of the nonlinear flag).

        Parameters
        ----------
        cosmo_dic: dict
            External dictionary from Cosmology class

        Returns
        -------
        theory: dict
            Updated dictionary of the :obj:`Nonlinear` class
        """
        self.theory = cosmo_dic
        self.nuis = cosmo_dic['nuisance_parameters']
        self.misc.update_dic(cosmo_dic)

        # Add Omb
        self.theory['Omb'] = \
            self.theory['ombh2'] / (self.theory['H0'] / 100) ** 2

        # Creation of EE2 instance on the first request of EE2
        if (self.theory['NL_flag_phot_matter'] == 4 and self.ee2 is None):

            import euclidemu2

            self.ee2 = euclidemu2.PyEuclidEmulator()

        # Creation of BACCO emulator instance on the first request of BACCO
        if ((self.theory['NL_flag_phot_matter'] == 5 or
             self.theory['NL_flag_phot_baryon'] == 4) and self.bemu is None):

            import baccoemu

            baccoemu_file = parent_path + \
                '/NN_emulator_PCA6_0.99_400_400n_paired'

            self.bemu = \
                baccoemu.Matter_powerspectrum(
                    verbose=False,
                    nonlinear_emu_path=baccoemu_file,
                    nonlinear_emu_details='details.pickle')

        if self.theory['NL_flag_phot_baryon'] == 3:

            if self.bcemu is None:

                import BCemu

                self.bcemu = BCemu.BCM_7param(Ob=self.theory['Omb'],
                                              Om=self.theory['Omm'])

            self.bcemu_params_interpolator(
                                       self.theory['redshift_bins_means_phot'])

        if self.theory['NL_flag_phot_baryon'] == 4:

            self.bacco_params_interpolator(
                                       self.theory['redshift_bins_means_phot'])

        # In case we need to extrapolate with HMcode and HMcode is not
        # available from Cobaya, create a new interpolator with it
        if ((self.theory['NL_flag_phot_matter'] in [4, 5]) and
            (self.theory['NL_flag_phot_baryon'] in [1, 2]) and
            ('hm' in (self.nonlinear_dic['option_extra_wavenumber'] +
                      self.nonlinear_dic['option_extra_redshift'] +
                      self.nonlinear_dic['option_extra_cosmo']))):
            self.nonlinear_dic['P_NL_extra'] = self.hm_extra_interpolator()
        else:
            self.nonlinear_dic['P_NL_extra'] = None

        self.nonlinear_dic['NL_boost'] = self.linear_boost
        self.nonlinear_dic['Bar_boost'] = self.linear_boost
        # Calculate boost factor if NL_flag_phot_matter is 4 or 5 or in
        # particular cases when it is necessary for NL_flag_phot_matter = 2, 3
        if self.theory['NL_flag_phot_matter'] in [2, 3, 4, 5]:
            self.calculate_boost()

        # Calculate baryon boost factor if NL_flag_phot_baryon is > 0
        if (self.theory['NL_flag_phot_baryon'] > 0):
            self.calculate_baryon_boost()

        # Compute Pgg(k,mu) if NL_flag_spectro is 1
        if self.theory['NL_flag_spectro'] == 1:
            self.calculate_eft()

        # Compute Pbibj terms of bias expansion if NL_flag_phot_bias is 1
        if self.theory['NL_flag_phot_bias'] == 1:
            self.calculate_phot_nl_bias()

        # Update of classes for power spectra interpolators
        self.Pgg_phot_model.update_dic(cosmo_dic, self.nonlinear_dic,
                                       self.misc)
        self.PgL_phot_model.update_dic(cosmo_dic, self.nonlinear_dic,
                                       self.misc)
        self.PLL_phot_model.update_dic(cosmo_dic, self.nonlinear_dic,
                                       self.misc)
        self.Pgg_spectro_model.update_dic(cosmo_dic, self.nonlinear_dic,
                                          self.misc)

        # Compute the one-loop terms for the TATT model
        if self.theory['IA_flag'] == 1:
            self.theory['a00e'], self.theory['c00e'], \
                self.theory['a0e0e'], self.theory['a0b0b'], \
                self.theory['ae2e2'], self.theory['ab2b2'], \
                self.theory['a0e2'], self.theory['b0e2'], \
                self.theory['d0ee2'], self.theory['d0bb2'] = \
                self.misc.ia_tatt_terms(self.theory['k_win'])

        # Create the interpolators for all the parameters of the
        # galaxy bias expansion (b1, b2, bG2, bG3)
        # if the non-linear contribution for photo galaxy bias are asked
        if self.theory['NL_flag_phot_bias'] == 1:
            self.create_phot_galbias_nl()
        else:
            self.theory['b2_inter'] = None
            self.theory['bG2_inter'] = None
            self.theory['bG3_inter'] = None

        return self.theory

    def calculate_eft(self):
        """Calculates EFT.

        Computes anisotropic galaxy power spectrum, and adds it to the
        nonlinear dictionary as an array of interpolator objects (one for
        each redshift bin, as specified by the class attribute ``zmeans``).
        """
        # Initializing EFT object
        self.eftobj = EFTofLSS(self.theory)
        # Check if rescaling by growth can be carried out (currently the only
        # discriminant is whether massive neutrinos are included or not)
        use_growth_rescaling = False if self.theory['mnu'] != 0.0 else True
        # Computing P(k,mu) interpolator for each specified redshift
        Pkmu = np.array(
            [self.eftobj.P_kmu_z(
                redshift=z, use_growth_rescaling=use_growth_rescaling,
                IRres=self.theory['IR_resum'], **rb.select_spectro_parameters(
                    float(z), self.nuis, self.zbins))
             for i, z in enumerate(self.zmeans)])
        # Storing in the nonlinear dictionary
        self.nonlinear_dic['P_kmu'] = Pkmu

    def calculate_boost(self):
        """Calculates the boost factor.

        Checks nonlinear photometric flag, computes the corresponding
        boost factor, and adds it to the nonlinear dictionary.
        """

        if (self.theory['NL_flag_phot_matter'] == 4):

            wavenumber, boost, redshift_max, flag_range, norm_dist = \
                self.ee2_boost()

        elif (self.theory['NL_flag_phot_matter'] == 5):

            wavenumber, boost, redshift_max, flag_range, norm_dist = \
                self.bacco_boost()

        elif (self.theory['NL_flag_phot_matter'] == 2 and
              self.theory['NL_flag_phot_baryon'] == 2):

            wavenumber, boost, redshift_max, flag_range, norm_dist = \
                self.hm_matter_boost(option='2016')

        elif (self.theory['NL_flag_phot_matter'] == 3 and
              self.theory['NL_flag_phot_baryon'] == 1):

            wavenumber, boost, redshift_max, flag_range, norm_dist = \
                self.hm_matter_boost(option='2020')

        else:
            return

        # Extrapolation function for all cases. In the cosmology case, the
        # options are {const, hm_simple, hm_smooth}, for wavenumber and
        # redshift, options can be selected from {const, power_law,
        # hm_simple} and wavenumber extrapolation has the additional option
        # of hm_smooth.
        wavenumber_out, boost_ext = self.extend_boost(
           wavenumber, boost, redshift_max, flag_range, norm_dist,
           option_wavenumber=self.nonlinear_dic['option_extra_wavenumber'],
           option_redshift=self.nonlinear_dic['option_extra_redshift'],
           option_cosmo=self.nonlinear_dic['option_extra_cosmo'])

        self.nonlinear_dic['NL_boost'] = interpolate.RectBivariateSpline(
            self.theory['z_win'], wavenumber_out, boost_ext, kx=1, ky=1)

    def calculate_baryon_boost(self):
        """Calculates Baryonic feedback Boost

        Checks nonlinear baryonic feedback flag, computes the corresponding
        boost-factor, and adds it to the nonlinear dictionary.
        """

        if self.theory['NL_flag_phot_baryon'] == 4:

            wavenumber, boost, redshift_max, flag_range = \
                self.bacco_baryon_boost(
                    option=self.nonlinear_dic['option_extra_bar'])

        elif self.theory['NL_flag_phot_baryon'] == 3:

            wavenumber, boost, redshift_max, flag_range = \
                self.BCemu_baryon_boost(
                    option=self.nonlinear_dic['option_extra_bar'])

        elif (self.theory['NL_flag_phot_baryon'] == 2 and
              self.theory['NL_flag_phot_matter'] != 3):

            wavenumber, boost, redshift_max, flag_range = \
                self.hm_baryon_boost(option='2020')

        elif (self.theory['NL_flag_phot_baryon'] == 1 and
              self.theory['NL_flag_phot_matter'] != 2):

            wavenumber, boost, redshift_max, flag_range = \
                self.hm_baryon_boost(option='2016')
        else:
            return

        # Extrapolation function for all cases. In the cosmology case, the
        # only option is const, for wavenumber and redshift, options can be
        # selected from {const, power_law} but are here chosen to be power law
        # by default as that is the best option available
        wavenumber_out, boost_ext = \
            self.extend_boost(wavenumber, boost, redshift_max, flag_range,
                              option_wavenumber="power_law",
                              option_redshift="power_law",
                              option_cosmo="const")

        self.nonlinear_dic['Bar_boost'] = \
            interpolate.RectBivariateSpline(
                    self.theory['z_win'],
                    wavenumber_out, boost_ext, kx=1, ky=1)

    def linear_boost(self, redshift, wavenumber):
        """Linear Boost

        Returns the boost factor for the linear case (i.e. 1).

        Returns
        -------
        Boost: float
           Value of linear boost at input redshift and wavenumber(s)

        """
        boost = 1.0
        return [boost]

    def ee2_boost(self):
        """EE2 Boost.

        Returns the boost factor for the EE2 (i.e. NL_flag_phot_matter==4).
        Parameter ranges are as follows:

        - Omb: [0.04, 0.06],
        - Omm: [0.24, 0.40],
        - H0: [61, 73],
        - As: [1.7e-9, 2.5e-9]
        - ns: [0.92, 1.00],
        - w: [-1.3, -0.7],
        - wa: [-0.7,  0.5],
        - mnu: [0.00, 0.15],
        - wavenumber (h/Mpc): [0.0087, 9.41]
        - redshift: [0, 10].

        Returns
        -------
        Wavenumber: numpy.ndarray
           Scales used by EE2 in units of 1/Mpc
        Boost: numpy.ndarray
           Array with boost at `z_win` redshifts and scales `wavenumber_out`

        """

        assert self.theory['Omk'] == 0, 'Non flat geometries not supported' \
                                        + ' in EE2'

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        # This is the maximum redshift for which EE2 will give a prediction.
        redshift_max = self.ee2.z_max

        redshifts = self.theory['z_win'][self.theory['z_win'] <= redshift_max]

        hubble = self.theory['H0'] / 100

        Omb = self.theory['ombh2'] / hubble ** 2

        params = {
            'Omega_b': Omb,
            'Omega_m': self.theory['Omm'],
            'h': hubble,
            'A_s': self.theory['As'],
            'n_s': self.theory['ns'],
            'm_ncdm': self.theory['mnu'],
            'w0_fld': self.theory['w'],
            'wa_fld': self.theory['wa']
        }

        ee2_bounds = self.ee2.bounds
        flag_emu = True
        dist_ranges = np.zeros(len(params))

        # Replace cosmo params with limiting cases
        for i, key in enumerate(params.keys()):
            if params[key] < ee2_bounds[key][0]:
                # Compute distance to edge of range
                dist_ranges[i] = (params[key] - ee2_bounds[key][0]) \
                                 / (ee2_bounds[key][1] - ee2_bounds[key][0])
                # Set parameter to the lower bound
                params[key] = ee2_bounds[key][0]

                flag_emu = False

            elif params[key] > ee2_bounds[key][1]:
                # Compute distance to edge of range
                dist_ranges[i] = (params[key] - ee2_bounds[key][1]) \
                                 / (ee2_bounds[key][1] - ee2_bounds[key][0])
                # Set parameter to the upper bound
                params[key] = ee2_bounds[key][1]

                flag_emu = False

        norm_dist = np.sqrt(np.dot(dist_ranges, dist_ranges))

        wavenumber, boost = self.ee2.get_boost(params, redshifts)

        boost_arr = np.array([boost[i] for i in range(len(redshifts))])

        wavenumber_out = wavenumber * hubble

        return wavenumber_out, boost_arr, redshift_max, flag_emu, norm_dist

    def bacco_boost(self):
        """BACCO Boost.

        Returns the boost factor for the BACCO case
        (i.e. NL_flag_phot_matter==5).
        Parameter ranges (in the variables of BACCO) are as follows:

        - sigma8_cold: [0.73, 0.9]
        - ns: [0.92, 1.01]
        - omega_cold: [0.23 ,0.4]
        - omega_baryon: [0.04, 0.06]
        - hubble: [0.6, 0.8]
        - neutrino_mass: [0.0, 0.4]
        - w0: [-1.15, -0.85]
        - wa: [-0.3, 0.3]
        - wavenumber (h/Mpc): [0.01, 5.00]
        - redshift: [0, 1.5]

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Scales used by BACCO in units of 1/Mpc
        boost_arr: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales
           ``wavenumber_out``. Value of the BACCO boost at input r
           edshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported' \
                                        + ' in BACCO'

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        redshift_max = \
            1.0 / self.bemu.emulator['nonlinear']['bounds'][-1][0] - 1.0

        hubble = self.theory['H0'] / 100

        Omb = self.theory['ombh2'] / hubble ** 2

        # Dictionary with parameters for BACCO.
        # expfactor is the scale factor, here set to 1.0, but later changed
        # for each redshift
        params = {
            'omega_cold': self.theory['Omc'] + Omb,
            'A_s': self.theory['As'],
            'omega_baryon': Omb,
            'ns': self.theory['ns'],
            'hubble': hubble,
            'neutrino_mass': self.theory['mnu'],
            'w0': self.theory['w'],
            'wa': self.theory['wa'],
            'expfactor': 1.0
        }

        # Check if params are in range for sigma8 emulator. We do this via
        # this if block. When it is in range, it does nothing, otherwise
        # it returns a predetermined set of outputs which will allow
        # the extrapolation functions to work.
        bemu_s8_bounds = self.bemu.emulator['sigma8']['bounds']
        bemu_NL_bounds = self.bemu.emulator['nonlinear']['bounds']

        mask_bounds = np.ones(len(bemu_NL_bounds)) > 0
        mask_bounds[1] = False
        mask_bounds[-1] = False
        short_NL_bounds = bemu_NL_bounds[mask_bounds]

        mixed_bounds = np.copy(bemu_s8_bounds)

        mixed_bounds[:, 0] = \
            np.where((short_NL_bounds - bemu_s8_bounds)[:, 0] > 0,
                     short_NL_bounds[:, 0], bemu_s8_bounds[:, 0])
        mixed_bounds[:, 1] = \
            np.where((short_NL_bounds - bemu_s8_bounds)[:, 1] < 0,
                     short_NL_bounds[:, 1], bemu_s8_bounds[:, 1])

        dist_ranges = np.zeros(len(params) - 1)
        # Set flag_emu to true as default
        flag_emu = True

        for ik, key in enumerate(self.bemu.emulator['sigma8']['keys']):
            if params[key] < mixed_bounds[ik, 0]:
                # Compute distance to edge of range
                dist_ranges[ik] = (params[key] - mixed_bounds[ik, 0]) \
                            / (mixed_bounds[ik, 1] - mixed_bounds[ik, 0])

                params[key] = mixed_bounds[ik, 0]

                flag_emu = False

            elif params[key] > mixed_bounds[ik, 1]:
                # Compute distance to edge of range
                dist_ranges[ik] = (params[key] - mixed_bounds[ik, 1]) \
                            / (mixed_bounds[ik, 1] - mixed_bounds[ik, 0])

                params[key] = mixed_bounds[ik, 1]

                flag_emu = False

        # Calculate sigma8 so that it is available for checking against bounds
        sigma8_cold = self.bemu.get_sigma8(**params)

        # Finally, check if sigma8_cold is in range in the same way as above.
        if (sigma8_cold < bemu_NL_bounds[1, 0]):
            dist_ranges[-1] = \
                (sigma8_cold - bemu_NL_bounds[1, 0]) / \
                (bemu_NL_bounds[1, 1] - bemu_NL_bounds[1, 0])

            flag_emu = False

            sigma8_cold = bemu_NL_bounds[1, 0]

        elif (sigma8_cold > bemu_NL_bounds[1, 1]):
            dist_ranges[-1] = \
                (sigma8_cold - bemu_NL_bounds[1, 1]) / \
                (bemu_NL_bounds[1, 1] - bemu_NL_bounds[1, 0])

            flag_emu = False

            sigma8_cold = bemu_NL_bounds[1, 1]

        norm_dist = np.sqrt(np.dot(dist_ranges, dist_ranges))

        params.pop('A_s')
        params['sigma8_cold'] = sigma8_cold

        redshifts = self.theory['z_win'][self.theory['z_win'] <= redshift_max]

        wavenumber = self.bemu.emulator['nonlinear']['k']
        boost_arr = np.ones((len(redshifts), len(wavenumber)))

        for i, redshift in enumerate(redshifts):

            params['expfactor'] = 1.0 / (1.0 + redshift)
            wavenumber, boost_arr[i] = \
                self.bemu.get_nonlinear_boost(cold=False, **params)

        wavenumber_out = wavenumber * hubble

        return wavenumber_out, boost_arr, redshift_max, flag_emu, norm_dist

    def hm_matter_boost(self, option="2020",
                        wavenumber_in=None, redshift_in=None):
        """Computes HMcode Boost

        Returns the boost factor for the hmcode matter
        (i.e. NL_flag_phot_matter==2 or 3)

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Scales used by HMcode in units of 1/Mpc
        boost_arr: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales wavenumber_out
        Value of the HMcode matter boost at input redshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported'

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        hubble = self.theory['H0'] / 100.

        if wavenumber_in is None:
            wavenumber_hm = self.theory['Pk_delta'].k
            wavenumber_out = wavenumber_hm
            mask_hm = np.ones(len(wavenumber_hm)) > 0
        else:
            # HMcode can only work if it is passed a full set of wavenumbers
            wavenumber_base = self.theory['Pk_delta'].k

            wavenumber_minus = \
                wavenumber_base[wavenumber_base < wavenumber_in[0]]

            # The part with wavenumber>wavenumber_in
            wavenumber_plus = \
                wavenumber_base[wavenumber_base > wavenumber_in[-1]]

            # Join them together for output
            wavenumber_hm = np.concatenate((wavenumber_minus,
                                            wavenumber_in,
                                            wavenumber_plus))
            wavenumber_out = wavenumber_in

            mask_hm = ((wavenumber_hm >= wavenumber_in[0]) &
                       (wavenumber_hm <= wavenumber_in[-1]))

        if redshift_in is None:
            redshifts = self.theory['z_win']
        else:
            redshifts = redshift_in
        # Setting redshift_max to the maximum redshift in array
        redshift_max = redshifts[-1]

        # Since this function will also be used in extrapolation, we put the
        # special case for HMcode2020 that does not require further
        # calculations here.
        if self.NL_Bar_matrix[self.theory['NL_flag_phot_matter'],
                              self.theory['NL_flag_phot_baryon']] == 3:
            boost_arr = \
                self.theory['Pk_halomodel_recipe'].P(redshifts,
                                                     wavenumber_out) \
                / self.theory['Pk_delta'].P(redshifts,
                                            wavenumber_out)

            return wavenumber_out, boost_arr, redshift_max, True, 0

        elif ((self.nonlinear_dic['P_NL_extra'] is not None) and
              (option == "2020")):
            boost_arr = \
                self.nonlinear_dic['P_NL_extra'](redshifts, wavenumber_out) \
                / self.theory['Pk_delta'].P(redshifts, wavenumber_out)

            return wavenumber_out, boost_arr, redshift_max, True, 0

        # Setup HMcode internal cosmology
        hm_cos = hmcode.Cosmology()

        # Set HMcode internal cosmological parameters
        hm_cos.om_m = self.theory['Omm']
        hm_cos.om_b = self.theory['Omb']
        hm_cos.om_v = 1. - self.theory['Omm']
        hm_cos.h = hubble
        hm_cos.ns = self.theory['ns']
        hm_cos.m_nu = self.theory['mnu']

        P_lin = \
            self.theory['Pk_delta'].P(redshifts, wavenumber_hm) * hubble ** 3

        # Set the linear power spectrum for HMcode
        hm_cos.set_linear_power_spectrum(wavenumber_hm / hubble,
                                         redshifts, P_lin)

        # Set the halo model in HMcode
        if option == "2020":
            hmod = hmcode.Halomodel(hmcode.HMcode2020, verbose=False)
        elif option == "2016":
            hmod = hmcode.Halomodel(hmcode.HMcode2016, verbose=False)

        # Power spectrum calculation
        P_NL = hmcode.calculate_nonlinear_power_spectrum(hm_cos, hmod,
                                                         verbose=False)

        boost_arr = P_NL / P_lin

        return wavenumber_out, boost_arr[:, mask_hm], redshift_max, True, 0

    def hm_extra_interpolator(self):
        """Computes interpolator for HMcode Boost for extrapolation

        Returns the interpolator for the boost factor for the HMcode2020 matter

        Returns
        -------
        boost_int: scipy.interpolate.RectBivariateSpline
           Interpolator of the HMcode matter boost at input redshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported'

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        hubble = self.theory['H0'] / 100.

        wavenumber_hm = self.theory['Pk_delta'].k

        redshifts = self.theory['z_win']

        # Setup HMcode internal cosmology
        hm_cos = hmcode.Cosmology()

        # Set HMcode internal cosmological parameters
        hm_cos.om_m = self.theory['Omm']
        hm_cos.om_b = self.theory['Omb']
        hm_cos.om_v = 1. - self.theory['Omm']
        hm_cos.h = hubble
        hm_cos.ns = self.theory['ns']
        hm_cos.m_nu = self.theory['mnu']

        P_lin = \
            self.theory['Pk_delta'].P(redshifts, wavenumber_hm) * hubble ** 3

        # Set the linear power spectrum for HMcode
        hm_cos.set_linear_power_spectrum(wavenumber_hm / hubble,
                                         redshifts, P_lin)

        # Set the halo model in HMcode (always 2020)
        hmod = hmcode.Halomodel(hmcode.HMcode2020, verbose=False)

        # Power spectrum calculation
        P_NL = hmcode.calculate_nonlinear_power_spectrum(hm_cos, hmod,
                                                         verbose=False)

        P_NL_int = interpolate.RectBivariateSpline(
            self.theory['z_win'], wavenumber_hm, P_NL / hubble ** 3,
            kx=3, ky=3)

        return P_NL_int

    def bacco_baryon_boost(self, option='lin'):
        """BACCO Boost

        Returns the boost factor for the BACCO baryons
        (i.e. NL_flag_phot_baryon==4)

        Parameters
        ----------
        option: str
            Choice of extrapolation option, between 'const' and 'lin'

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Scales used by BACCO in units of 1/Mpc
        boost_arr: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales wavenumber_out
        Value of the BACCO baryon boost at input redshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported'

        # Order of params in dictionary is very important, as it needs to be
        # the same as the order of the array with the bounds of the emulator
        # Using total sigma8 here instead of sigma8_cold as approx.
        params = {
            'omega_cold': self.theory['Omc'] + self.theory['Omb'],
            'sigma8_cold': self.theory['sigma8_0'],
            'omega_baryon': self.theory['Omb'],
            'ns': self.theory['ns'],
            'hubble': self.theory['H0'] / 100,
            'neutrino_mass': self.theory['mnu'],
            'w0': self.theory['w'],
            'wa': self.theory['wa'],
        }

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        bounds_arr = self.bemu.emulator['baryon']['bounds']

        redshift_max = 1.0 / bounds_arr[-1][0] - 1.0

        redshifts = self.theory['z_win'][self.theory['z_win'] <= redshift_max]

        # If extrapolation is necessary, replace cosmo params with limiting
        # cases. The assumption here is that the dependence of the baryon
        # suppression for most cosmological parameters is very small, so it is
        # acceptable to extrapolate this dependence as a constant in all cases.
        for i, key in enumerate(params.keys()):
            if i not in [0, 2]:
                if params[key] < bounds_arr[i, 0]:
                    # Set parameter to the lower bound
                    params[key] = bounds_arr[i, 0]

                elif params[key] > bounds_arr[i, 1]:
                    # Set parameter to the upper bound
                    params[key] = bounds_arr[i, 1]

        # If extrapolation is necessary in the baryon fraction, that is taken
        # into account differently than for the other cosmological parameters
        fb = self.theory['Omb'] / (params['omega_cold'])

        Omm_min = bounds_arr[0, 0]
        Omm_max = bounds_arr[0, 1]
        Omb_min = bounds_arr[2, 0]
        Omb_max = bounds_arr[2, 1]

        fb_min = Omb_min / Omm_max
        fb_max = Omb_max / Omm_min

        # Determine whether fb is in range, even if Omb and Omm are not
        flag_fb_neg = (fb >= fb_min)
        flag_fb_pos = (fb <= fb_max)
        flag_fb = flag_fb_neg & flag_fb_pos

        flag_Oms = (params['omega_baryon'] > Omb_max) or \
                   (params['omega_cold'] > Omm_max) or \
                   (params['omega_cold'] < Omm_min) or \
                   (params['omega_baryon'] < Omb_min)

        if flag_fb and flag_Oms:
            fb_med_u = Omb_max / Omm_max
            fb_med_d = Omb_min / Omm_min

            if (fb > fb_med_u) and (params['omega_baryon'] > Omb_max):
                params['omega_baryon'] = Omb_max
                params['omega_cold'] = Omb_max / fb

            elif (fb < fb_med_u) and (params['omega_cold'] > Omm_max):
                params['omega_baryon'] = fb * Omm_max
                params['omega_cold'] = Omm_max

            elif (fb > fb_med_d) and (params['omega_cold'] < Omm_min):
                params['omega_baryon'] = fb * Omm_min
                params['omega_cold'] = Omm_min

            elif (fb < fb_med_d) and (params['omega_baryon'] < Omb_min):
                params['omega_baryon'] = Omb_min
                params['omega_cold'] = Omb_min / fb

        # Assume params are in range
        flag_emu = (redshifts > -1) & flag_fb

        # Evaluate baryonic feedback params at the requested redshifts
        par_array = self.bacco_params_ev(redshifts)

        # Get two copies, in case it is necessary to extrapolate and save
        # additional points
        par_array_extra = copy.deepcopy(par_array)
        par_array_extra2 = copy.deepcopy(par_array)

        # Index of the first feedback parameter in the bounds array
        index_bar_par = 8

        # Compute the mean points in the parameter ranges
        means = np.append(np.mean(bounds_arr[index_bar_par: -1], axis=1),
                          (fb_min + fb_max) / 2)

        # Array for storing the 1/20 of the distances to the means and the
        # distance to the edges of parameter ranges to use in the linear
        # extrapolation method.
        diff_pars = np.zeros((2, 8, len(redshifts)))

        # Inverse distance fraction set to 20
        inv_dist_frac = 20

        for i, key in enumerate(par_array.keys(), start=index_bar_par):
            flag_emu_i_pos = (par_array[key] <= bounds_arr[i, 1])
            flag_emu_i_neg = (par_array[key] >= bounds_arr[i, 0])

            # Save edge of ranges for all parameters outside the range
            par_array_extra[key][~flag_emu_i_pos] = bounds_arr[i, 1]
            par_array_extra[key][~flag_emu_i_neg] = bounds_arr[i, 0]

            # Save additionally parameter points near the edge of the range to
            # compute derivatives with respect to parameters
            par_array_extra2[key][~flag_emu_i_pos] = \
                bounds_arr[i, 1] - (bounds_arr[i, 1] -
                                    means[i - index_bar_par]) / inv_dist_frac
            par_array_extra2[key][~flag_emu_i_neg] = \
                bounds_arr[i, 0] - (bounds_arr[i, 0] -
                                    means[i - index_bar_par]) / inv_dist_frac

            # Save distances from requested parameters and edge of the ranges
            diff_pars[0, i - index_bar_par, (~flag_emu_i_pos)] = \
                par_array[key][~flag_emu_i_pos] - bounds_arr[i, 1]
            diff_pars[0, i - index_bar_par, (~flag_emu_i_neg)] = \
                par_array[key][~flag_emu_i_neg] - bounds_arr[i, 0]

            # Save distances between edge and point close to the edge
            diff_pars[1, i - index_bar_par, (~flag_emu_i_pos)] = \
                (bounds_arr[i, 1] - means[i - index_bar_par]) / inv_dist_frac
            diff_pars[1, i - index_bar_par, (~flag_emu_i_neg)] = \
                (bounds_arr[i, 0] - means[i - index_bar_par]) / inv_dist_frac

            flag_emu = flag_emu & flag_emu_i_pos & flag_emu_i_neg

        # Do the same operations for the baryon fraction fb
        if (~flag_fb_pos):
            diff_pars[0, -1, :] = fb - fb_max
            diff_pars[1, -1, :] = (fb_max - fb_min) / (2 * inv_dist_frac)

            par_array_extra['omega_baryon'] = Omb_max
            par_array_extra['omega_cold'] = Omm_min

            par_array_extra2['omega_baryon'] = \
                (fb_max - (fb_max - fb_min) / (2 * inv_dist_frac)) * Omm_min
            par_array_extra2['omega_cold'] = Omm_min

        elif (~flag_fb_neg):
            diff_pars[0, -1, :] = fb - fb_min
            diff_pars[1, -1, :] = (fb_min - fb_max) / (2 * inv_dist_frac)

            par_array_extra['omega_baryon'] = Omb_min
            par_array_extra['omega_cold'] = Omm_max

            par_array_extra2['omega_baryon'] = \
                (fb_min - (fb_min - fb_max) / (2 * inv_dist_frac)) * Omm_max
            par_array_extra2['omega_cold'] = Omm_max

        else:
            par_array_extra['omega_baryon'] = params['omega_baryon']
            par_array_extra['omega_cold'] = params['omega_cold']

            par_array_extra2['omega_baryon'] = params['omega_baryon']
            par_array_extra2['omega_cold'] = params['omega_cold']

        wavenumber = self.bemu.emulator['nonlinear']['k']
        boost_arr = np.ones((len(redshifts), len(wavenumber)))

        # Compute boost with given params (or updated ones, if outside range)
        for i, redshift in enumerate(redshifts):

            params_bar = {'M_c': par_array['M_c_bacco'][i],
                          'eta': par_array['eta_bacco'][i],
                          'beta': par_array['beta_bacco'][i],
                          'M1_z0_cen': par_array['M1_z0_cen_bacco'][i],
                          'theta_inn': par_array['theta_inn_bacco'][i],
                          'M_inn': par_array['M_inn_bacco'][i],
                          'theta_out': par_array['theta_out_bacco'][i]
                          }

            params.update(params_bar)

            params['expfactor'] = 1. / (1. + redshift)

            # Case when parameters are in range at the current redshift
            if flag_emu[i]:

                wavenumber, boost_arr[i] = \
                    self.bemu.get_baryonic_boost(k=wavenumber, **params)

            # If parameters are not in range, perform extrapolation
            else:
                params_bar = \
                    {'M_c': par_array_extra['M_c_bacco'][i],
                     'eta': par_array_extra['eta_bacco'][i],
                     'beta': par_array_extra['beta_bacco'][i],
                     'M1_z0_cen': par_array_extra['M1_z0_cen_bacco'][i],
                     'theta_inn': par_array_extra['theta_inn_bacco'][i],
                     'M_inn': par_array_extra['M_inn_bacco'][i],
                     'theta_out': par_array_extra['theta_out_bacco'][i],
                     'omega_baryon': par_array_extra['omega_baryon'],
                     'omega_cold': par_array_extra['omega_cold']
                     }

                params.update(params_bar)

                wavenumber, boost1 = \
                    self.bemu.get_baryonic_boost(k=wavenumber, **params)

                # Linear extrapolation case
                if option == 'lin':

                    params_bar = \
                        {'M_c': par_array_extra2['M_c_bacco'][i],
                         'eta': par_array_extra2['eta_bacco'][i],
                         'beta': par_array_extra2['beta_bacco'][i],
                         'M1_z0_cen': par_array_extra2['M1_z0_cen_bacco'][i],
                         'theta_inn': par_array_extra2['theta_inn_bacco'][i],
                         'M_inn': par_array_extra2['M_inn_bacco'][i],
                         'theta_out': par_array_extra2['theta_out_bacco'][i],
                         'omega_baryon': par_array_extra2['omega_baryon'],
                         'omega_cold': par_array_extra2['omega_cold']
                         }

                    params.update(params_bar)

                    wavenumber, boost2 = \
                        self.bemu.get_baryonic_boost(k=wavenumber, **params)

                    step_12 = \
                        np.sqrt(np.dot(diff_pars[1, :, i], diff_pars[1, :, i]))
                    delta_par = np.sqrt(np.dot(diff_pars[0, :, i],
                                        diff_pars[0, :, i]))

                    derivative = \
                        (np.log(boost1) - np.log(boost2)) / step_12 * delta_par

                    boost_arr[i] = boost1 * np.exp(derivative)

                # Constant extrapolation case
                elif option == "const":

                    boost_arr[i] = boost1

                else:
                    raise Exception('Wrong extrapolation option for baryons.')

        wavenumber_out = wavenumber * self.theory['H0'] / 100

        return wavenumber_out, boost_arr, redshift_max, True

    def BCemu_baryon_boost(self, option='lin'):
        """Computes BCemu baryon Boost

        Returns the boost factor for the BCemu baryons
        (i.e. NL_flag_phot_baryon==3)

        Parameters
        ----------
        option: str
            Choice of extrapolation option, between 'const' and 'lin'

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Scales used by BCemu in units of 1/Mpc
        boost_arr: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales wavenumber_out
        Value of the BCemu baryon boost at input redshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported'

        # Omega_cold = self.theory['Omc'] + self.theory['Omb']
        # fb = self.theory['Omb'] / Omega_cold
        fb = self.theory['Omb'] / self.theory['Omm']

        bc_limits = {
                     'log10Mc_bcemu': [11, 15],
                     'mu_bcemu': [0.0, 2.0],
                     'thej_bcemu': [2, 8],
                     'gamma_bcemu': [1, 4],
                     'delta_bcemu': [3, 11],
                     'eta_bcemu': [0.05, 0.4],
                     'deta_bcemu': [0.05, 0.4],
                     'fb': [0.1, 0.25]
                    }

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        # Maximum redshift for BCemu, here hard-coded as it is not a parameter
        # that BCemu provides internally
        redshift_max = 2.0

        redshifts = self.theory['z_win'][self.theory['z_win'] <= redshift_max]

        par_array = self.bcemu_params_ev(redshifts)
        par_array['fb'] = fb * np.ones(len(redshifts))
        par_array_extra = copy.deepcopy(par_array)
        par_array_extra2 = copy.deepcopy(par_array)

        # Create the default case, with all True except when fb is outside
        flag_emu = ((redshifts > -1) &
                    (fb >= bc_limits['fb'][0] and fb <= bc_limits['fb'][1]))

        # Array for storing the 1/20 of the distances to the means and the
        # distance to the edges of parameter ranges to use in the linear
        # extrapolation method.
        diff_pars = np.zeros((2, 8, len(redshifts)))

        # Inverse distance fraction set to 20
        inv_dist_frac = 20

        for i, key in enumerate(par_array.keys()):
            flag_emu_i_pos = (par_array[key] <= bc_limits[key][1])
            flag_emu_i_neg = (par_array[key] >= bc_limits[key][0])

            par_array_extra[key][~flag_emu_i_pos] = bc_limits[key][1]
            par_array_extra[key][~flag_emu_i_neg] = bc_limits[key][0]

            par_array_extra2[key][~flag_emu_i_pos] = \
                bc_limits[key][1] - (bc_limits[key][1] -
                                     bc_limits[key][0]) / (2 * inv_dist_frac)
            par_array_extra2[key][~flag_emu_i_neg] = \
                bc_limits[key][0] - (bc_limits[key][0] -
                                     bc_limits[key][1]) / (2 * inv_dist_frac)

            diff_pars[0, i, (~flag_emu_i_pos)] = \
                par_array[key][~flag_emu_i_pos] - bc_limits[key][1]
            diff_pars[0, i, (~flag_emu_i_neg)] = \
                par_array[key][~flag_emu_i_neg] - bc_limits[key][0]
            diff_pars[1, i, (~flag_emu_i_pos)] = \
                (bc_limits[key][1] - bc_limits[key][0]) / (2 * inv_dist_frac)
            diff_pars[1, i, (~flag_emu_i_neg)] = \
                (bc_limits[key][0] - bc_limits[key][1]) / (2 * inv_dist_frac)

            flag_emu = flag_emu & flag_emu_i_pos & flag_emu_i_neg

        wavenumber = self.bcemu.ks0
        boost_arr = np.ones((len(redshifts), len(wavenumber)))

        for i, redshift in enumerate(redshifts):

            params = {
                      'log10Mc': par_array['log10Mc_bcemu'][i],
                      'mu': par_array['mu_bcemu'][i],
                      'thej': par_array['thej_bcemu'][i],
                      'gamma': par_array['gamma_bcemu'][i],
                      'delta': par_array['delta_bcemu'][i],
                      'eta': par_array['eta_bcemu'][i],
                      'deta': par_array['deta_bcemu'][i],
                      'fb': fb
            }

            if flag_emu[i]:

                boost_arr[i] = \
                    self.bcemu.get_boost(redshift, params, wavenumber)

            else:

                params = {
                          'log10Mc': par_array_extra['log10Mc_bcemu'][i],
                          'mu': par_array_extra['mu_bcemu'][i],
                          'thej': par_array_extra['thej_bcemu'][i],
                          'gamma': par_array_extra['gamma_bcemu'][i],
                          'delta': par_array_extra['delta_bcemu'][i],
                          'eta': par_array_extra['eta_bcemu'][i],
                          'deta': par_array_extra['deta_bcemu'][i],
                          'fb': par_array_extra['fb'][i]
                }

                boost1 = self.bcemu.get_boost(redshift, params, wavenumber)

                # Linear extrapolation case
                if option == 'lin':

                    params = {
                              'log10Mc': par_array_extra2['log10Mc_bcemu'][i],
                              'mu': par_array_extra2['mu_bcemu'][i],
                              'thej': par_array_extra2['thej_bcemu'][i],
                              'gamma': par_array_extra2['gamma_bcemu'][i],
                              'delta': par_array_extra2['delta_bcemu'][i],
                              'eta': par_array_extra2['eta_bcemu'][i],
                              'deta': par_array_extra2['deta_bcemu'][i],
                              'fb': par_array_extra2['fb'][i]
                    }

                    boost2 = self.bcemu.get_boost(redshift, params, wavenumber)

                    step_12 = \
                        np.sqrt(np.dot(diff_pars[1, :, i], diff_pars[1, :, i]))
                    delta_par = np.sqrt(np.dot(diff_pars[0, :, i],
                                        diff_pars[0, :, i]))

                    derivative = \
                        (np.log(boost1) - np.log(boost2)) / step_12 * delta_par

                    boost_arr[i] = boost1 * np.exp(derivative)

                # Constant extrapolation case
                elif option == "const":

                    boost_arr[i] = boost1

                else:
                    raise Exception('Wrong extrapolation option for baryons.')

        wavenumber_out = wavenumber * self.theory['H0'] / 100

        # The last argument is True so the extrapolator does not activate
        return wavenumber_out, boost_arr, redshift_max, True

    def hm_baryon_boost(self, option="2020"):
        """Computes HMcode Boost

        Returns the boost factor for the hmcode baryons
        (i.e. NL_flag_phot_baryon==1, 2 depending on the option)

        Parameters
        ----------
        option: str
            option of HMcode version between 2016 and 2020

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Scales used by HMcode in units of 1/Mpc
        boost_arr: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales wavenumber_out
        Value of the HMcode baryon boost at input redshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported'

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        hubble = self.theory['H0'] / 100.

        wavenumber_out = self.theory['Pk_delta'].k

        # Limit wavenumber range on large scales, since pyHMcode has not yet
        # corrected the large scales for the 2020 feedback model
        if ((option == "2020") and (self.theory['NL_flag_phot_matter'] == 1)):
            wavenumber_out = wavenumber_out[wavenumber_out >= 0.01 * hubble]

        # Setting redshift_max to the maximum redshift in 'z_win'
        redshifts = self.theory['z_win']
        redshift_max = redshifts[-1]

        # Setup HMcode internal cosmology
        hm_cos = hmcode.Cosmology()

        # Set HMcode internal cosmological parameters
        hm_cos.om_m = self.theory['Omm']
        hm_cos.om_b = self.theory['Omb']
        hm_cos.om_v = 1. - self.theory['Omm']
        hm_cos.h = hubble
        hm_cos.ns = self.theory['ns']
        hm_cos.m_nu = self.theory['mnu']

        P_lin = \
            self.theory['Pk_delta'].P(redshifts, wavenumber_out) * hubble ** 3

        # Set the linear power spectrum for HMcode
        hm_cos.set_linear_power_spectrum(wavenumber_out / hubble,
                                         redshifts, P_lin)

        # Set the halo model in HMcode
        if option == "2020":
            if self.nonlinear_dic['P_NL_extra'] is not None:

                P_NL = self.nonlinear_dic['P_NL_extra'](redshifts,
                                                        wavenumber_out) * \
                                                        hubble ** 3
            else:
                hmod = hmcode.Halomodel(hmcode.HMcode2020, verbose=False)
                P_NL = hmcode.calculate_nonlinear_power_spectrum(hm_cos, hmod,
                                                                 verbose=False)
        elif option == "2016":
            hmod = hmcode.Halomodel(hmcode.HMcode2016, verbose=False)
            P_NL = hmcode.calculate_nonlinear_power_spectrum(hm_cos, hmod,
                                                             verbose=False)

        if self.theory['NL_flag_phot_matter'] == 1:
            if option == "2020":

                hm_cos.theat = 10**self.nuis['HMCode_logT_AGN']
                hmod = hmcode.Halomodel(hmcode.HMcode2020_feedback,
                                        verbose=False)
            elif option == "2016":

                hmod = hmcode.Halomodel(hmcode.HMcode2016,
                                        verbose=False)
                hmod.eta0 = self.nuis['HMCode_eta_baryon']
                hmod.As = self.nuis['HMCode_A_baryon']
            # Power spectrum calculation
            P_bar = hmcode.calculate_nonlinear_power_spectrum(hm_cos, hmod,
                                                              verbose=False)
        else:
            P_bar = self.theory['Pk_halomodel_recipe'].P(redshifts,
                                                         wavenumber_out) * \
                                                         hubble ** 3

        boost_arr = P_bar / P_NL

        return wavenumber_out, boost_arr, redshift_max, True

    def extend_boost(self, wavenumber_in, boost_in,
                     redshift_max, flag_range, norm_dist=0,
                     option_wavenumber="hm_simple",
                     option_redshift="hm_simple",
                     option_cosmo="hm_simple"):
        """Calculates extrapolation of the boost outside its given range.

        Returns nonlinear boost array and
        corresponding scales in 1/Mpc.

        Options for wavenumber extrapolation:
            - const, `power_law`, `hm_simple`, `hm_smooth`
        Options for redshift extrapolation:
            - const, `power_law`, `hm_simple`
        Options for cosmo extrapolation:
            - const, `hm_simple`, `hm_smooth`

        Both hm_simple and hm_smooth extrapolate with HMcode, but the smooth
        case uses a tanh to interpolate between another extrapolation and
        HMcode, making it a continuous extrapolation.

        Parameters
        ----------
        wavenumber_in: numpy.ndarray
            Scales used by emulator in units of 1/Mpc
        boost_in: numpy.ndarray
           Array with boost at `z_win` redshifts and scales `wavenumber_in`
        redshift_max: float
            Max redshift of emulator
        flag_range: bool
            Flag for cosmo params range of emulator
        option_wavenumber: string
            Option for wavenumber extrapolation
        option_redshift: str
            Option for redshift extrapolation
        option_cosmo: str
            Option for cosmology extrapolation

        Returns
        -------
        Wavenumber: numpy.ndarray
           Concatenation of scales used by emulator with those used by
           linear power spectrum, in units of 1/Mpc
        Boost: numpy.ndarray
           Array with nonlinear boost at `z_win` redshifts
           and scales `wavenumber_out`

        """

        hubble = self.theory['H0'] / 100.

        try:
            wavenumber_max_Pkd = self.theory['Pk_delta'].k[-1]
        except AttributeError:
            try:
                wavenumber_max_Pkd = self.theory['Pk_delta'].kmax
            except AttributeError:
                wavenumber_max_Pkd = wavenumber_in[-1]

        wavenumber_base = self.theory['k_win'][(self.theory['k_win'] <=
                                                wavenumber_max_Pkd)]

        # Extrapolate everything with HMcode if cosmology is not in range
        # and the option to extrapolate is HMcode.
        # Note that 'Pk_halomodel_recipe' always includes the prediction from
        # HMcode since for all values of NL_flag_phot_matter corresponding to
        # the emulators (4, 5) the nonlinear model given by cobaya is HMcode.
        if (not flag_range) and option_cosmo == "hm_simple":

            wavenumber_out, boost_out, _, _, _ = \
                self.hm_matter_boost(option="2020")

        # If cosmology is in range, then proceed to do the other extrapolations
        # Or if the cosmology extrapolation option is const, then we still need
        # to perform extrapolation in the other variables, so do the same thing
        elif flag_range or (option_cosmo in ["const", "hm_smooth"]):

            # Split the wavenumber range into 3 parts, one is just
            # wavenumber_in, others are:
            # The part with wavenumber<wavenumber_in
            wavenumber_minus = \
                        wavenumber_base[wavenumber_base < wavenumber_in[0]]

            # The part with wavenumber>wavenumber_in
            wavenumber_plus = \
                wavenumber_base[wavenumber_base > wavenumber_in[-1]]

            # Join them together for output
            wavenumber_out = np.concatenate((wavenumber_minus,
                                             wavenumber_in,
                                             wavenumber_plus))
            boost_out = np.ones((len(self.theory['z_win']),
                                 len(wavenumber_out)))

            # Use correct result in the range of the emulator in redshift and
            # wavenumber
            boost_out[:boost_in.shape[0],
                      len(wavenumber_minus):(len(wavenumber_minus) +
                      len(wavenumber_in))] = boost_in

            if redshift_max < self.theory['z_win'][-1]:
                # Choose redshifts out-of-range of the emulator
                redshift_out = self.theory['z_win'][boost_in.shape[0]:]
                redshift_in = self.theory['z_win'][:boost_in.shape[0]]

                if option_redshift == "hm_simple":
                    # Use HMCode for those
                    boost_out[boost_in.shape[0]:,
                              len(wavenumber_minus):(len(wavenumber_minus) +
                              len(wavenumber_in))] = \
                        self.hm_matter_boost(option="2020",
                                             redshift_in=redshift_out,
                                             wavenumber_in=wavenumber_in)[1]

                elif option_redshift == "const":
                    # Use const
                    boost_out[boost_in.shape[0]:,
                              len(wavenumber_minus):(len(wavenumber_minus) +
                              len(wavenumber_in))] = \
                        np.ones((len(redshift_out), len(wavenumber_in))) * \
                        boost_in[-1, :][None, :]

                elif option_redshift == "power_law":
                    # Use power law

                    n_extra_b = \
                        (np.log(boost_in[-1, :]) - np.log(boost_in[-2, :])) \
                        / (np.log(redshift_in[-1]) - np.log(redshift_in[-2]))

                    boost_out[boost_in.shape[0]:,
                              len(wavenumber_minus):(len(wavenumber_minus) +
                              len(wavenumber_in))] = \
                        boost_in[-1, :][None, :] \
                        * ((redshift_out / redshift_in[-1])[:, None]) \
                        ** n_extra_b[None, :]

                else:
                    raise Exception('Wrong redshift extrapolation option.')

            # Choose redshifts in-range of the emulator
            redshift_all = self.theory['z_win']

            # Use linear recipe for wavenumber<wavenumber_in so keep boost=1
            # Different options for wavenumber>wavenumber_in
            if wavenumber_base[-1] > wavenumber_in[-1]:

                if option_wavenumber == "const":
                    # Use final boost for wavenumber>wavenumber_in
                    boost_out[:, (len(wavenumber_minus) +
                                  len(wavenumber_in)):] = \
                        np.ones((len(redshift_all), len(wavenumber_plus))) * \
                        boost_out[:, (len(wavenumber_minus) +
                                      len(wavenumber_in) - 1)][:, None]

                elif option_wavenumber == "hm_simple":
                    # Use HMCode for wavenumber>wavenumber_in
                    boost_out[:, (len(wavenumber_minus) +
                                  len(wavenumber_in)):] = \
                        self.hm_matter_boost(option="2020",
                                             redshift_in=redshift_all,
                                             wavenumber_in=wavenumber_plus)[1]

                elif option_wavenumber == "hm_smooth":
                    # Use modulated HMCode for wavenumber>wavenumber_in
                    boost_hmcode = \
                        self.hm_matter_boost(option="2020",
                                             redshift_in=redshift_all,
                                             wavenumber_in=wavenumber_plus)[1]

                    i_last = len(wavenumber_minus) + len(wavenumber_in)

                    # Compute boost spectral index at last point
                    n_extra_b = (np.log(boost_out[:, i_last - 1]) -
                                 np.log(boost_out[:, i_last - 2])) / \
                        (np.log(wavenumber_in[-1]) - np.log(wavenumber_in[-2]))

                    # Get a power-law extrapolation
                    boost_powerlaw = boost_out[:, i_last - 1][:, None] \
                        * ((wavenumber_plus / wavenumber_in[-1])[None, :]) \
                        ** n_extra_b[:, None]

                    # Mix power-law with HMcode using tanh
                    # tanh params
                    tanh_slope = self.nonlinear_dic['wavenumber_tanh_slope']
                    tanh_scale = self.nonlinear_dic['wavenumber_tanh_scale'] \
                        * np.log(wavenumber_in[-1])

                    boost_out[:, (len(wavenumber_minus) +
                                  len(wavenumber_in)):] = \
                        boost_powerlaw + 0.5 * (boost_hmcode -
                                                boost_powerlaw) * \
                        (np.tanh(tanh_slope * (np.log(wavenumber_plus) -
                                               tanh_scale)) + 1.0)[None, :]

                elif option_wavenumber == "power_law":
                    # Use power law in wavenumber for wavenumber>wavenumber_in

                    i_last = len(wavenumber_minus) + len(wavenumber_in)

                    n_extra_b = (np.log(boost_out[:, i_last - 1]) -
                                 np.log(boost_out[:, i_last - 2])) / \
                        (np.log(wavenumber_in[-1]) - np.log(wavenumber_in[-2]))

                    boost_out[:, (len(wavenumber_minus) +
                                  len(wavenumber_in)):] = \
                        boost_out[:, i_last - 1][:, None] \
                        * ((wavenumber_plus / wavenumber_in[-1])[None, :]) \
                        ** n_extra_b[:, None]

                else:
                    raise Exception('Wrong wavenumber extrapolation option.')

            if (not flag_range) and option_cosmo == "hm_smooth":

                boost_hmcode = \
                    self.hm_matter_boost(option="2020",
                                         redshift_in=self.theory['z_win'],
                                         wavenumber_in=wavenumber_out)[1]

                dist_trans = np.min(2 * norm_dist, initial=1)

                boost_out = \
                    boost_out * (1 - dist_trans) + boost_hmcode * dist_trans

        else:
            raise Exception('Wrong cosmo extrapolation option.')

        return wavenumber_out, boost_out

    def bcemu_params_interpolator(self, redshift_means):
        r"""Baryonic parameters interpolator for BCemu

        Creates a linear interpolator for the BCemu params for the
        photometric probes at a given redshift z

        Parameters
        ----------
        redshift_means: array_like
            Array of tomographic redshift bin means for photometric probes.
            Default is Euclid IST: Forecasting choices.
        """

        nuisance_par = self.theory['nuisance_parameters']

        par_name_list = \
            ['log10Mc_bcemu', 'mu_bcemu', 'thej_bcemu', 'gamma_bcemu',
             'delta_bcemu', 'eta_bcemu', 'deta_bcemu']

        par_int_dict = dict.fromkeys(par_name_list)

        for par in par_name_list:

            par_list = [nuisance_par[f'{par}_bin{idx}']
                        for idx, vl in enumerate(redshift_means, start=1)]
            par_int_dict[par] = \
                rb.linear_interpolator(redshift_means, par_list)

        self.nonlinear_dic['bcemu_par_inter'] = par_int_dict

    def bcemu_params_ev(self, redshift):
        r"""Interpolated BCemu params

        Gets BCemu params for the photometric probes by
        interpolation at a given redshift z

        Note: for redshifts above the final bin (z > 2.5), we use the params
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.001), we use the params of the first bin.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate params.
        bin_edges: numpy.ndarray
            Array of tomographic redshift bin edges for photometric probes.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        par_res_dict: dict
            Dictionary with values of params at input redshift(s)
        """

        par_name_list = \
            ['log10Mc_bcemu', 'mu_bcemu', 'thej_bcemu', 'gamma_bcemu',
             'delta_bcemu', 'eta_bcemu', 'deta_bcemu']

        par_res_dict = dict.fromkeys(par_name_list)

        if self.theory['Baryon_redshift_model']:

            nuisance_par = self.theory['nuisance_parameters']

            for par in par_name_list:

                X_0 = nuisance_par[f'{par}_0']
                nu_X = nuisance_par[f'nu_{par}']

                par_res_dict[par] = X_0 * (1 + redshift) ** (- nu_X)

        else:
            for key in par_name_list:
                par_res_dict[key] = \
                    self.nonlinear_dic['bcemu_par_inter'][key](redshift)

        return par_res_dict

    def bacco_params_interpolator(self, redshift_means):
        r"""Baryonic parameters interpolator for BACCO baryons

        Creates a linear interpolator for the BACCO params for the
        photometric probes at a given redshift z

        Parameters
        ----------
        redshift_means: array_like
            Array of tomographic redshift bin means for photometric probes.
            Default is Euclid IST: Forecasting choices.
        """

        nuisance_par = self.theory['nuisance_parameters']

        par_name_list = \
            ['M_c_bacco', 'eta_bacco', 'beta_bacco', 'M1_z0_cen_bacco',
             'theta_inn_bacco', 'M_inn_bacco', 'theta_out_bacco']

        par_int_dict = dict.fromkeys(par_name_list)

        for par in par_name_list:

            par_list = [nuisance_par[f'{par}_bin{idx}']
                        for idx, vl in enumerate(redshift_means, start=1)]
            par_int_dict[par] = \
                interpolate.interp1d(redshift_means, par_list,
                                     fill_value="extrapolate")

        self.nonlinear_dic['bacco_par_inter'] = par_int_dict

    def bacco_params_ev(self, redshift):
        r"""Interpolated BACCO baryon params

        Gets BACCO params for the photometric probes by
        interpolation at a given redshift z

        Note: for redshifts above the final bin (z > 2.5), we use the params
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.001), we use the params of the first bin.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate params.
        bin_edges: numpy.ndarray
            Array of tomographic redshift bin edges for photometric probes.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        par_res_dict: dict
            Dictionary with values of params at input redshift(s)
        """

        par_name_list = \
            ['M_c_bacco', 'eta_bacco', 'beta_bacco', 'M1_z0_cen_bacco',
             'theta_inn_bacco', 'M_inn_bacco', 'theta_out_bacco']

        par_res_dict = dict.fromkeys(par_name_list)

        if self.theory['Baryon_redshift_model']:

            nuisance_par = self.theory['nuisance_parameters']

            for par in par_name_list:

                X_0 = nuisance_par[f'{par}_0']
                nu_X = nuisance_par[f'nu_{par}']

                par_res_dict[par] = X_0 * (1 + redshift) ** (- nu_X)

        else:
            for key in par_name_list:
                par_res_dict[key] = \
                    self.nonlinear_dic['bacco_par_inter'][key](redshift)

        return par_res_dict

    def calculate_phot_nl_bias(self):
        """Calculates non-linear bias terms to be used for Pgg_phot

        Description
        """
        # Initializing EFT object
        eftobj = EFTofLSS(self.theory)
        # Computing loops at redshift z=0
        eftobj._Pgg_k_terms_L()
        # Creating 2D array of interpolators
        Pb1b2, Pb1bG2, Pb2b2, Pb2bG2, PbG2bG2, PZ1bG3, PZ1bG2 =\
            eftobj.P_realspace_terms_kz()

        # Storing in the nonlinear dictionary
        self.nonlinear_dic['Pb1b2_kz'] = Pb1b2
        self.nonlinear_dic['Pb1bG2_kz'] = Pb1bG2
        self.nonlinear_dic['Pb2b2_kz'] = Pb2b2
        self.nonlinear_dic['Pb2bG2_kz'] = Pb2bG2
        self.nonlinear_dic['PbG2bG2_kz'] = PbG2bG2
        self.nonlinear_dic['PZ1bG3_kz'] = PZ1bG3
        self.nonlinear_dic['PZ1bG2_kz'] = PZ1bG2

    def create_phot_galbias_nl(self, model=None):
        r"""Creates the photometric non-linear galaxy bias.

        Creates all the parameter of the galaxy
        bias expansion (b1, b2, bG2, bG3) for
        photometric GC as function/interpolator of the redshift.
        The functions are stored in the cosmo dictionaries
        'b1_inter', 'b2_inter', 'bG2_inter', 'bG3_inter'.

        The redshift evolution model for the bias is selected from the key
        'bias_model' in theory. For now, there are only 2 possible models
        the linear interpolation and the 3rd order polynomial.

        Parameters
        ----------
        model: integer
            selection of the bias model.
            If None, uses the one stored in theory['bias_model']
        x_values: numpy.ndarray of float
            x-values for the interpolator.
        y_values: numpy.ndarray of float
            y-values for the interpolator.

        Raises
        ------
        ValueError
            If the bias model parameter in the cosmo dictionary
            is not 1 or 3
        """

        if model is None:
            bias_model = self.theory['bias_model']
        else:
            bias_model = model

        if bias_model == 1:
            bias_interpolators = self.istf_phot_galbias_nl_interpolator(
                    self.theory['redshift_bins_means_phot'])
            self.theory['b1_inter'] = bias_interpolators[0]
            self.theory['b2_inter'] = bias_interpolators[1]
            self.theory['bG2_inter'] = bias_interpolators[2]
            self.theory['bG3_inter'] = bias_interpolators[3]

        elif bias_model == 3:
            self.theory['b1_inter'] = self.poly_phot_galbias_nl('b1')
            self.theory['b2_inter'] = self.poly_phot_galbias_nl('b2')
            self.theory['bG2_inter'] = self.poly_phot_galbias_nl('bG2')
            self.theory['bG3_inter'] = self.poly_phot_galbias_nl('bG3')
        else:
            raise ValueError('Parameter bias_model cannot be different from'
                             '1 or 3 for non-linear photometric bias. It is:'
                             f'{bias_model}')

    def istf_phot_galbias_nl_interpolator(self, redshift_means):
        r"""IST:F Photometric non-linear galaxy bias interpolators.

        Returns a linear interpolator for each
        parameter of the galaxy bias expansion for the
        photometric GC probes at a given redshift.

        Parameters
        ----------
        redshift_means: numpy.ndarray of float
            Array of tomographic redshift bin means for GCphot

        Returns
        -------
        b1 interpolator, b2 interpolator,
        bG2 interpolator, bG3 interpolator: rb.linear_interpolator,
        rb.linear_interpolator, rb.linear_interpolator,
        rb.linear_interpolator
            Linear interpolators of photometric non-linear galaxy
            bias parameters
        """

        nuisance_par = self.theory['nuisance_parameters']

        istf_b1_list = [nuisance_par[f'b1_photo_bin{idx}']
                        for idx, vl in
                        enumerate(redshift_means, start=1)]
        istf_b2_list = [nuisance_par[f'b2_photo_bin{idx}']
                        for idx, vl in
                        enumerate(redshift_means, start=1)]
        istf_bG2_list = [nuisance_par[f'bG2_photo_bin{idx}']
                         for idx, vl in
                         enumerate(redshift_means, start=1)]
        istf_bG3_list = [nuisance_par[f'bG3_photo_bin{idx}']
                         for idx, vl in
                         enumerate(redshift_means, start=1)]

        return (rb.linear_interpolator(redshift_means, istf_b1_list),
                rb.linear_interpolator(redshift_means, istf_b2_list),
                rb.linear_interpolator(redshift_means, istf_bG2_list),
                rb.linear_interpolator(redshift_means, istf_bG3_list))

    def poly_phot_galbias_nl(self, which_bias):
        r"""Polynomial photometric non-linear galaxy bias.

        Computes non-linear bias parameters using a 3rd order
        polynomial function of redshift.

        Parameters
        ----------
        redshift: float or numpy.ndarray
            Redshift(s) at which to calculate bias

        Returns
        -------
        Photometric polynomial galaxy biases
        (b1, b2, bG2, bG3): float or numpy.ndarray
            Value(s) of photometric galaxy biases at input redshift(s)
        """

        nuisance = self.theory['nuisance_parameters']

        return np.poly1d([nuisance[f'{which_bias}_{i}_poly_photo']
                          for i in range(4)])

    def Pmm_phot_def(self, redshift, wavenumber):
        r"""Interface for ``Pmm_phot_def``.

        Returns the matter-matter power spectrum,
        defined in the :obj:`pLL_phot module`.
        """

        flag_halo_emu = \
            self.halo_emu_matrix[self.theory['NL_flag_phot_matter'],
                                 self.theory['NL_flag_phot_baryon']]

        switcher = {1: self.PLL_phot_model.Pmm_phot_halo,
                    2: self.PLL_phot_model.Pmm_phot_emu
                    }
        Pmm_phot_func = \
            switcher.get(flag_halo_emu,
                         "Invalid modeling option")

        return Pmm_phot_func(redshift, wavenumber)

    def Pgg_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Interface for ``Pgg_spectro_def``.

        Returns the spectroscopic galaxy-galaxy power spectrum,
        defined in the :obj:`pgg_spectro` module.
        """
        switcher = {1: self.Pgg_spectro_model.Pgg_spectro_eft}

        Pgg_spectro_func = switcher.get(self.theory['NL_flag_spectro'],
                                        "Invalid modeling option")
        return Pgg_spectro_func(redshift, wavenumber, mu_rsd)

    def noise_Pgg_spectro(self, redshift, wavenumber, mu_rsd):
        r"""Interface for ``noise_Pgg_spectro``.

        Returns the noise contributions to the galaxy-galaxy power spectrum,
        defined in the :obj:`pgg_spectro` module.
        """
        noise_func = self.Pgg_spectro_model.noise_Pgg_spectro
        return noise_func(redshift, wavenumber, mu_rsd)

    def Pgdelta_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Interface for ``Pgdelta_spectro_def``.

        Returns the spectroscopic galaxy-density power spectrum,
        defined in the :obj:`pgg_spectro` module.
        """
        return self.Pgg_spectro_model.Pgdelta_spectro_def(redshift,
                                                          wavenumber, mu_rsd)

    def Pgg_phot_def(self, redshift, wavenumber):
        r"""Interface for ``Pgg_phot_def``.

        Returns the galaxy-galaxy power spectrum,
        defined in the :obj:`pgg_phot` module.
        """
        flag_halo_emu = \
            self.halo_emu_matrix[self.theory['NL_flag_phot_matter'],
                                 self.theory['NL_flag_phot_baryon']]

        switcher = {(1, 0): self.Pgg_phot_model.Pgg_phot_halo,
                    (2, 0): self.Pgg_phot_model.Pgg_phot_emu,
                    (1, 1): self.Pgg_phot_model.Pgg_phot_halo_NLbias,
                    (2, 1): self.Pgg_phot_model.Pgg_phot_emu_NLbias,
                    }
        Pgg_phot_func = \
            switcher.get((flag_halo_emu, self.theory['NL_flag_phot_bias']),
                         "Invalid modeling option")

        return Pgg_phot_func(redshift, wavenumber)

    def Pii_def(self, redshift, wavenumber):
        r"""Interface for ``Pii_def``.

        Returns the intrinsic-intrinsic power spectrum,
        defined in the defined in the :obj:`pLL_phot` module.
        """

        flag_halo_emu = \
            self.halo_emu_matrix[self.theory['NL_flag_phot_matter'],
                                 self.theory['NL_flag_phot_baryon']]

        switcher_IA = {0: 'NLA',
                       1: 'TATT'}

        if switcher_IA.get(self.theory['IA_flag'],
                           "Invalid modeling option") == 'NLA':

            switcher = {1: self.PLL_phot_model.Pii_halo_nla,
                        2: self.PLL_phot_model.Pii_emu_nla
                        }
            Pii_func = \
                switcher.get(flag_halo_emu,
                             "Invalid modeling option")
        elif switcher_IA.get(self.theory['IA_flag'],
                             "Invalid modeling option") == 'TATT':

            switcher = {1: self.PLL_phot_model.Pii_ee_halo_tatt,
                        2: self.PLL_phot_model.Pii_ee_emu_tatt
                        }
            Pii_func = \
                switcher.get(flag_halo_emu,
                             "Invalid modeling option")
        else:
            print("Invalid modeling IA option")

        return Pii_func(redshift, wavenumber)

    def Pdeltai_def(self, redshift, wavenumber):
        r"""Interface for ``Pdeltai_def``.

        Returns the density-intrinsic power spectrum,
        defined in the defined in the :obj:`pLL_phot` module.
        """

        flag_halo_emu = \
            self.halo_emu_matrix[self.theory['NL_flag_phot_matter'],
                                 self.theory['NL_flag_phot_baryon']]

        switcher_IA = {0: 'NLA',
                       1: 'TATT'}

        if switcher_IA.get(self.theory['IA_flag'],
                           "Invalid modeling option") == 'NLA':
            switcher = {1: self.PLL_phot_model.Pdeltai_halo_nla,
                        2: self.PLL_phot_model.Pdeltai_emu_nla
                        }
            Pdeltai_func = \
                switcher.get(flag_halo_emu,
                             "Invalid modeling option")
        elif switcher_IA.get(self.theory['IA_flag'],
                             "Invalid modeling option") == 'TATT':
            switcher = {1: self.PLL_phot_model.Pdeltai_halo_tatt,
                        2: self.PLL_phot_model.Pdeltai_emu_tatt
                        }
            Pdeltai_func = \
                switcher.get(flag_halo_emu,
                             "Invalid modeling option")
        else:
            print("Invalid modeling IA option")

        return Pdeltai_func(redshift, wavenumber)

    def Pgi_phot_def(self, redshift, wavenumber):
        r"""Interface for ``Pgi_phot``.

        Returns the galaxy-intrinsic power spectrum,
        defined in the :obj:`pgL_phot` module
        """

        flag_halo_emu = \
            self.halo_emu_matrix[self.theory['NL_flag_phot_matter'],
                                 self.theory['NL_flag_phot_baryon']]

        switcher_IA = {0: 'NLA',
                       1: 'TATT'}

        if switcher_IA.get(self.theory['IA_flag'],
                           "Invalid modeling option") == 'NLA':
            switcher = {1: self.PgL_phot_model.Pgi_phot_halo_nla,
                        2: self.PgL_phot_model.Pgi_phot_emu_nla
                        }
            Pgi_func = \
                switcher.get(flag_halo_emu,
                             "Invalid modeling option")
        elif switcher_IA.get(self.theory['IA_flag'],
                             "Invalid modeling option") == 'TATT':
            switcher = {1: self.PgL_phot_model.Pgi_phot_halo_tatt,
                        2: self.PgL_phot_model.Pgi_phot_emu_tatt
                        }
            Pgi_func = \
                switcher.get(flag_halo_emu,
                             "Invalid modeling option")
        else:
            print("Invalid modeling IA option")

        return Pgi_func(redshift, wavenumber)

    def Pgi_spectro_def(self, redshift, wavenumber):
        r"""Interface for ``Pgi_spectro_def``.

        Returns the spectroscopic galaxy-intrinsic power spectrum,
        defined in the :obj:`pgL_phot` module
        """
        return self.PgL_phot_model.Pgi_spectro_def(redshift, wavenumber)

    def Pgdelta_phot_def(self, redshift, wavenumber):
        r"""Interface for ``Pgdelta_phot_def``.

        Returns the photometric galaxy-density power spectrum,
        defined in the :obj:`pgL_phot` module
        """
        flag_halo_emu = \
            self.halo_emu_matrix[self.theory['NL_flag_phot_matter'],
                                 self.theory['NL_flag_phot_baryon']]

        switcher = {(1, 0): self.PgL_phot_model.Pgdelta_phot_halo,
                    (2, 0): self.PgL_phot_model.Pgdelta_phot_emu,
                    (1, 1): self.PgL_phot_model.Pgdelta_phot_halo_NLbias,
                    (2, 1): self.PgL_phot_model.Pgdelta_phot_emu_NLbias,
                    }
        PgL_phot_func = \
            switcher.get((flag_halo_emu, self.theory['NL_flag_phot_bias']),
                         "Invalid modeling option")

        return PgL_phot_func(redshift, wavenumber)
