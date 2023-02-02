"""Nonlinear

Class to compute nonlinear recipes.
"""

import numpy as np
from scipy import interpolate
from cloe.non_linear.miscellanous import Misc
from cloe.non_linear.pgg_spectro import Pgg_spectro_model
from cloe.non_linear.pgg_phot import Pgg_phot_model
from cloe.non_linear.pgL_phot import PgL_phot_model
from cloe.non_linear.pLL_phot import PLL_phot_model
import euclidemu2
import baccoemu


class NonlinearError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class Nonlinear:
    """
    Class to compute nonlinear recipes
    """

    def __init__(self, cosmo_dic):
        """Initialise

        Initialises class and nonlinear code

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from Cosmology class
        """
        self.theory = cosmo_dic

        self.misc = Misc(cosmo_dic)

        # Empty variables for emulator class instances
        self.ee2 = None
        self.bemu = None

        self.nonlinear_dic = {'NL_boost': None,
                              'option_extra_wavenumber': 'hm_smooth',
                              'option_extra_redshift': 'hm_simple'}

        self.Pgg_spectro_model = Pgg_spectro_model(cosmo_dic,
                                                   self.nonlinear_dic,
                                                   self.misc)

        self.Pgg_phot_model = Pgg_phot_model(cosmo_dic, self.nonlinear_dic,
                                             self.misc)
        self.PgL_phot_model = PgL_phot_model(cosmo_dic, self.nonlinear_dic,
                                             self.misc)
        self.PLL_phot_model = PLL_phot_model(cosmo_dic, self.nonlinear_dic,
                                             self.misc)

    def update_dic(self, cosmo_dic):
        """Updates Dic

        Calls all routines updating the cosmo dictionary

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from Cosmology class

        Returns
        -------
        cosmo_dic: dict
            Updated dictionary

        """
        self.theory = cosmo_dic

        # Creation of EE2 instance on the first request of EE2
        if (self.theory['NL_flag'] == 3 and self.ee2 is None):

            self.ee2 = euclidemu2.PyEuclidEmulator()

        # Creation of BACCO emulator instance on the first request of BACCO
        if (self.theory['NL_flag'] == 4 and self.bemu is None):

            self.bemu = baccoemu.Matter_powerspectrum(verbose=False)

        self.calculate_boost()

        self.misc.update_dic(cosmo_dic)

        self.Pgg_spectro_model.update_dic(cosmo_dic,
                                          self.nonlinear_dic, self.misc)
        self.Pgg_phot_model.update_dic(cosmo_dic,
                                       self.nonlinear_dic, self.misc)
        self.PgL_phot_model.update_dic(cosmo_dic,
                                       self.nonlinear_dic, self.misc)
        self.PLL_phot_model.update_dic(cosmo_dic,
                                       self.nonlinear_dic, self.misc)

        return self.theory

    def calculate_boost(self):
        """Calculates Boost

        Checks nonlinear flag, computes the corresponding
        boost-factor and adds it to the nonlinear dictionary
        """

        if (self.theory['NL_flag'] > 2):

            if (self.theory['NL_flag'] == 3):

                wavenumber, boost, redshift_max, flag_range = self.ee2_boost()

            elif (self.theory['NL_flag'] == 4):

                wavenumber, boost, redshift_max, flag_range = \
                                                            self.bacco_boost()

            else:
                raise Exception('Invalid value of NL_flag,'
                                ' valid options are [0,1,2,3,4]')

            # Extrapolation function for all cases. In the cosmology case, the
            # only available option is hm_simple, while for wavenumber and
            # redshift, options can be selected from {const, power_law,
            # hm_simple} and wavenumber extrapolation has the additional option
            # of hm_smooth.
            wavenumber_out, boost_ext = self.extend_boost(
               wavenumber,
               boost,
               redshift_max,
               flag_range,
               option_wavenumber=self.nonlinear_dic['option_extra_wavenumber'],
               option_redshift=self.nonlinear_dic['option_extra_redshift'],
               option_cosmo="hm_simple")

            self.nonlinear_dic['NL_boost'] = \
                interpolate.RectBivariateSpline(
                        self.theory['z_win'],
                        wavenumber_out, boost_ext, kx=1, ky=1)

    def linear_boost(self):
        """Linear Boost

        Returns the boost factor for the linear case (i.e. 1)

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate the boost
        scale: float
            Wave mode at which to calculate the boost

        Returns
        -------
        boost: float
           Value of linear boost at input redshift and scale

        """
        boost = 1.0
        return boost

    def ee2_boost(self):
        """EE2 Boost

        Returns the boost factor for the EE2 (i.e. NL_flag==3). Parameter
        ranges are as follows:

        - Omb: [0.04, 0.06],
        - Omm: [0.24, 0.40],
        - H0: [61, 73],
        - As: [1.7e-9, 2.5e-9]
        - ns: [0.92, 1.00],
        - w: [-1.3, -0.7],
        - wa: [-0.7,  0.5],
        - mnu: [0.00, 0.15],
        - wavenumber (h/Mpc): [0.0087, 9.41]
        - redshift: [0, 10]

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Scales used by EE2 in units of 1/Mpc
        boost_arr: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales wavenumber_out

        """

        assert self.theory['Omk'] == 0, 'Non flat geometries not supported' \
                                        + ' in EE2'

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        # This is the maximum redshift for which EE2 will give a prediction.
        # It is hard-coded here since there is currently no way to retrieve
        # this from the EE2 class. This is likely to change in the future and
        # this will then be corrected.
        redshift_max = 10.0

        redshifts = self.theory['z_win'][self.theory['z_win'] <= redshift_max]

        # We check whether we are in range of the emulator in this try/except
        # block. When it is in range, there is no error, otherwise it returns
        # a predetermined set of outputs which will allow the extrapolation
        # functions to work later
        try:
            wavenumber, boost = self.ee2.get_boost(self.theory, redshifts)
        except ValueError:
            return None, None, None, False

        boost_arr = np.array([boost[i] for i in range(len(redshifts))])

        h = self.theory['H0'] / 100.0
        wavenumber_out = wavenumber * h

        return wavenumber_out, boost_arr, redshift_max, True

    def bacco_boost(self):
        """BACCO Boost

        Returns the boost factor for the BACCO case (i.e. NL_flag==4).
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
           Array with boost at 'z_win' redshifts and scales wavenumber_out
        Value of the BACCO boost at input redshift and scale
        """
        assert self.theory['Omk'] == 0, 'Non flat geometries not supported' \
                                        + ' in BACCO'

        redshift_max = \
            1.0 / self.bemu.emulator['nonlinear']['bounds'][-1][0] - 1.0

        # Dictionary with parameters for BACCO.
        # expfactor is the scale factor, here set to 1.0, but later changed
        # for each redshift
        params = {
            'omega_cold': self.theory['Omc'] + self.theory['Omb'],
            'A_s': self.theory['As'],
            'omega_baryon': self.theory['Omb'],
            'ns': self.theory['ns'],
            'hubble': self.theory['H0'] / 100,
            'neutrino_mass': self.theory['mnu'],
            'w0': self.theory['w'],
            'wa': self.theory['wa'],
            'expfactor': 1.0
        }

        # Check if params are in range for sigma8 emulator. We do this via
        # this if/else block. When it is not in range it returns a
        # predetermined set of outputs which will allow the extrapolation
        # functions to work.
        ord_par = [params[key] for key in self.bemu.emulator['sigma8']['keys']]
        if (np.all(ord_par >=
                   self.bemu.emulator['sigma8']['bounds'][:, 0]) and
            np.all(ord_par <=
                   self.bemu.emulator['sigma8']['bounds'][:, 1])):
            flag_emu = True
        else:
            flag_emu = False
            return None, None, redshift_max, flag_emu

        # Calculate sigma8 so that it is available for checking against bounds
        sigma8_cold = self.bemu.get_sigma8(**params)

        # Finally, check if sigma8_cold is in range in the same way as above.
        if ((sigma8_cold >=
             self.bemu.emulator['nonlinear']['bounds'][1, 0]) and
            (sigma8_cold <=
             self.bemu.emulator['nonlinear']['bounds'][1, 1])):
            flag_emu = True
            params.pop('A_s')
            params['sigma8_cold'] = sigma8_cold
        else:
            flag_emu = False
            return None, None, redshift_max, flag_emu

        if self.theory['z_win'] is None:
            raise Exception('Boltzmann code redshift binning has not been '
                            'supplied to cosmo_dic.')

        redshifts = self.theory['z_win'][self.theory['z_win'] <= redshift_max]

        wavenumber = self.bemu.emulator['nonlinear']['k']
        boost_arr = np.ones((len(redshifts), len(wavenumber)))

        for i, redshift in enumerate(redshifts):

            params['expfactor'] = 1.0 / (1.0 + redshift)
            wavenumber, boost_arr[i] = \
                self.bemu.get_nonlinear_boost(cold=False, **params)

        wavenumber_out = wavenumber * self.theory['H0'] / 100.0

        return wavenumber_out, boost_arr, redshift_max, flag_emu

    def extend_boost(self, wavenumber_in, boost_in,
                     redshift_max, flag_range,
                     option_wavenumber="hm_simple",
                     option_redshift="hm_simple",
                     option_cosmo="hm_simple"):
        """Calculates extrapolation of the boost outside its given range

        Returns nonlinear boost array and
        corresponding scales in 1/Mpc.

        Options for wavenumber extrapolation:
            - const, power_law, hm_simple, hm_smooth
        Options for redshift extrapolation:
            - const, power_law, hm_simple
        Options for cosmo extrapolation:
            - const, hm_simple

        Both hm_simple and hm_smooth extrapolate with HMcode, but the smooth
        case uses a tanh to interpolate between a power law extrapolation and
        HMcode, making it a continuous extrapolation.

        Parameters
        ----------
        wavenumber_in: numpy.ndarray
            Scales used by emulator in units of 1/Mpc
        boost_in: numpy.ndarray
           Array with boost at 'z_win' redshifts and scales wavenumber_in
        redshift_max: float
            Max redshift of emulator
        flag_range: bool
            Flag for cosmo params range of emulator
        option_wavenumber: string
            Option for wavenumber extrapolation
        option_redshift: string
            Option for redshift extrapolation
        option_cosmo: string
            Option for cosmology extrapolation

        Returns
        -------
        wavenumber_out: numpy.ndarray
           Concatenation of scales used by emulator with those used by
           linear power spectrum, in units of 1/Mpc
        boost_out: numpy.ndarray
           Array with nonlinear boost at 'z_win' redshifts
           and scales wavenumber_out

        """

        h = self.theory['H0'] / 100.

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
        # HMcode since for all values of NL_flag corresponding to the emulators
        # (3, 4) the nonlinear model given by cobaya is HMcode.
        if (not flag_range) and option_cosmo == "hm_simple":

            wavenumber_out = wavenumber_base
            boost_out = \
                self.theory['Pk_halomodel_recipe'].P(self.theory['z_win'],
                                                     wavenumber_base) \
                / self.theory['Pk_delta'].P(self.theory['z_win'],
                                            wavenumber_base)

        # If cosmology is in range, then proceed to do the other extrapolations
        # Or if the cosmology extrapolation option is const, then we still need
        # to perform extrapolation in the other variables, so do the same thing
        elif flag_range or (option_cosmo == "const"):

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
                        self.theory['Pk_halomodel_recipe'].P(redshift_out,
                                                             wavenumber_in) \
                        / self.theory['Pk_delta'].P(redshift_out,
                                                    wavenumber_in)

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
            if option_wavenumber == "const":
                # Use final boost for wavenumber>wavenumber_in
                boost_out[:, len(wavenumber_minus) + len(wavenumber_in):] = \
                    np.ones((len(redshift_all), len(wavenumber_plus))) * \
                    boost_out[:, (len(wavenumber_minus) +
                                  len(wavenumber_in) - 1)][:, None]

            elif option_wavenumber == "hm_simple":
                # Use HMCode for wavenumber>wavenumber_in
                boost_out[:, len(wavenumber_minus) + len(wavenumber_in):] = \
                    self.theory['Pk_halomodel_recipe'].P(redshift_all,
                                                         wavenumber_plus) \
                    / self.theory['Pk_delta'].P(redshift_all,
                                                wavenumber_plus)

            elif option_wavenumber == "hm_smooth":
                # Use modulated HMCode for wavenumber>wavenumber_in
                boost_hmcode = \
                    self.theory['Pk_halomodel_recipe'].P(redshift_all,
                                                         wavenumber_plus) \
                    / self.theory['Pk_delta'].P(redshift_all,
                                                wavenumber_plus)

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
                # tanh params (currently hard-coded, but may be introduced
                # as new variables to be changed by the user in the future)
                tanh_slope = 10.0
                tanh_scale = 1.15 * np.log(wavenumber_in[-1])

                boost_out[:, len(wavenumber_minus) + len(wavenumber_in):] = \
                    boost_powerlaw + 0.5 * (boost_hmcode - boost_powerlaw) * \
                    (np.tanh(tanh_slope * (np.log(wavenumber_plus) -
                                           tanh_scale)) + 1.0)[None, :]

            elif option_wavenumber == "power_law":
                # Use power law in wavenumber for wavenumber>wavenumber_in

                i_last = len(wavenumber_minus) + len(wavenumber_in)

                n_extra_b = (np.log(boost_out[:, i_last - 1]) -
                             np.log(boost_out[:, i_last - 2])) \
                    / (np.log(wavenumber_in[-1]) - np.log(wavenumber_in[-2]))

                boost_out[:, len(wavenumber_minus) + len(wavenumber_in):] = \
                    boost_out[:, i_last - 1][:, None] \
                    * ((wavenumber_plus / wavenumber_in[-1])[None, :]) \
                    ** n_extra_b[:, None]

            else:
                raise Exception('Wrong wavenumber extrapolation option.')

        else:
            raise Exception('Wrong cosmo extrapolation option.')

        return wavenumber_out, boost_out

    def Pmm_phot_def(self, redshift, wavenumber):
        r"""Interface for Pmm_phot_def

        Returns the matter-matter power spectrum,
        defined in the pLL_phot module
        """
        switcher = {1: self.PLL_phot_model.Pmm_phot_halo,
                    2: self.PLL_phot_model.Pmm_phot_halo,
                    3: self.PLL_phot_model.Pmm_phot_emu,
                    4: self.PLL_phot_model.Pmm_phot_emu
                    }
        Pmm_phot_func = \
            switcher.get(self.theory['NL_flag'],
                         "Invalid modeling option")

        return Pmm_phot_func(redshift, wavenumber)

    def Pgg_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Interface for Pgg_spectro_def

        Returns the spectroscopic galaxy-galaxy power spectrum,
        defined in the pgg_spectro module
        """
        return self.Pgg_spectro_model.Pgg_spectro_def(redshift,
                                                      wavenumber, mu_rsd)

    def Pgdelta_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Interface for Pgdelta_spectro_def

        Returns the spectroscopic galaxy-density power spectrum,
        defined in the pgg_spectro module
        """
        return self.Pgg_spectro_model.Pgdelta_spectro_def(redshift,
                                                          wavenumber, mu_rsd)

    def Pgg_phot_def(self, redshift, wavenumber):
        r"""Interface for Pgg_phot_def

        Returns the galaxy-galaxy power spectrum,
        defined in the pgg_phot module
        """
        switcher = {1: self.Pgg_phot_model.Pgg_phot_halo,
                    2: self.Pgg_phot_model.Pgg_phot_halo,
                    3: self.Pgg_phot_model.Pgg_phot_emu,
                    4: self.Pgg_phot_model.Pgg_phot_emu
                    }
        Pgg_phot_func = \
            switcher.get(self.theory['NL_flag'],
                         "Invalid modeling option")

        return Pgg_phot_func(redshift, wavenumber)

    def Pii_def(self, redshift, wavenumber):
        r"""Interface for Pii_def

        Returns the intrinsic-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        switcher = {1: self.PLL_phot_model.Pii_halo,
                    2: self.PLL_phot_model.Pii_halo,
                    3: self.PLL_phot_model.Pii_emu,
                    4: self.PLL_phot_model.Pii_emu
                    }
        Pii_func = \
            switcher.get(self.theory['NL_flag'],
                         "Invalid modeling option")

        return Pii_func(redshift, wavenumber)

    def Pdeltai_def(self, redshift, wavenumber):
        r"""Interface for Pdeltai_def

        Returns the density-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        switcher = {1: self.PLL_phot_model.Pdeltai_halo,
                    2: self.PLL_phot_model.Pdeltai_halo,
                    3: self.PLL_phot_model.Pdeltai_emu,
                    4: self.PLL_phot_model.Pdeltai_emu
                    }
        Pdeltai_func = \
            switcher.get(self.theory['NL_flag'],
                         "Invalid modeling option")

        return Pdeltai_func(redshift, wavenumber)

    def Pgi_phot_def(self, redshift, wavenumber):
        r"""Interface for Pgi phot

        Returns the galaxy-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        switcher = {1: self.PLL_phot_model.Pgi_phot_halo,
                    2: self.PLL_phot_model.Pgi_phot_halo,
                    3: self.PLL_phot_model.Pgi_phot_emu,
                    4: self.PLL_phot_model.Pgi_phot_emu
                    }
        Pgi_func = \
            switcher.get(self.theory['NL_flag'],
                         "Invalid modeling option")

        return Pgi_func(redshift, wavenumber)

    def Pgi_spectro_def(self, redshift, wavenumber):
        r"""Interface for Pgi_spectro_def

        Returns the spectroscopic galaxy-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        return self.PLL_phot_model.Pgi_spectro_def(redshift, wavenumber)

    def Pgdelta_phot_def(self, redshift, wavenumber):
        r"""Interface for Pgdelta_phot_def

        Returns the photometric galaxy-density power spectrum,
        defined in the pgL_phot module
        """
        switcher = {1: self.PgL_phot_model.Pgdelta_phot_halo,
                    2: self.PgL_phot_model.Pgdelta_phot_halo,
                    3: self.PgL_phot_model.Pgdelta_phot_emu,
                    4: self.PgL_phot_model.Pgdelta_phot_emu
                    }
        PgL_phot_func = \
            switcher.get(self.theory['NL_flag'],
                         "Invalid modeling option")
        return PgL_phot_func(redshift, wavenumber)
