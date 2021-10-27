"""COBAYA INTERFACE
"""


# General import
import numpy as np
from astropy import constants as const

# Cobaya import of general Likelihood class
from cobaya.likelihood import Likelihood
# Import cobaya model wrapper for fiducial model (TEMPORARY)
from cobaya.model import get_model

# Import likelihoods classes and functions
from likelihood.cosmo.cosmology import Cosmology
from likelihood.like_calc.euclike import Euclike
from likelihood.auxiliary.observables_dealer import *

# Error classes


class CobayaInterfaceError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class EuclidLikelihood(Likelihood):
    r"""
    Class to define Euclid Likelihood

    Inherits from the Likelihood class
    of Cobaya
    """

    def initialize(self):
        r"""Initialize

        Set up values for initial variables
        and create instance of Cosmology class

        """

        # For now, example sampling in wavenumber (k)
        self.k_min_Boltzmannn = 0.001
        # Note: k_min is not passed to Cobaya to build
        # the matter power spectrum interpolator.
        # The k_min is internally chosen by Cobaya.
        # This needs to be changed.

        self.k_max_Boltzmannn = 50.0
        self.k_min_GC_phot_interp = 0.001
        self.k_max_GC_phot_interp = 1000.0
        self.k_samp_GC = 1000
        self.k_win = np.logspace(np.log10(self.k_min_GC_phot_interp),
                                 np.log10(self.k_max_GC_phot_interp),
                                 self.k_samp_GC)

        # For now, example sampling in redshift (z)
        self.z_min = 0.0
        self.z_max = 4.0
        self.z_samp = 100
        self.z_win = np.linspace(self.z_min, self.z_max, self.z_samp)
        # Logarithmic sampling below
        # self.z_min1 = 0.0
        # self.z_min2 = 1e-4
        # self.z_min3 = 1e-3
        # self.z_minlog = -2
        # self.z_max = 4.0
        # self.z_samp = 140
        # self.z_win = np.logspace(self.z_minlog, np.log10(self.z_max),
        #                          self.z_samp)
        # self.z_win[0] = self.z_min1
        # self.z_win[1] = self.z_min2
        # self.z_win[2] = self.z_min3
        # Check the selection and specification requirements
        self.observables = \
            observables_selection_specifications_checker(
                self.observables_selection,
                self.observables_specifications)
        # Visualization of the observables matrix
        if self.plot_observables_selection:
            self.observables_pf = observables_visualization(
             observables['selection'])
        # Initialize Euclike module
        # To Sergio: pass to Euclike self.observables, which is the merged dict
        # if I do it now, the code will complain
        self.likefinal = Euclike(self.data, self.observables_selection)

        # Initialize Cosmology class for sampling
        self.cosmo = Cosmology()

        # Initialize the fiducial model
        # This will work if CAMB is installed globally
        self.fiducial_cosmology = Cosmology()
        # Update fiducial cosmo dic with fiducial info from reader
        self.fiducial_cosmology.cosmo_dic.update(
            self.likefinal.data_spectro_fiducial_cosmo)
        self.info_fiducial = {'params': {
            'ombh2': self.fiducial_cosmology.cosmo_dic['ombh2'],
            'omch2': self.fiducial_cosmology.cosmo_dic['omch2'],
            'omnuh2': self.fiducial_cosmology.cosmo_dic['omnuh2'],
            'omk': self.fiducial_cosmology.cosmo_dic['Omk'],
            'H0': self.fiducial_cosmology.cosmo_dic['H0'],
            'H0_Mpc': self.cosmo.cosmo_dic['H0'] / const.c.to('km/s').value,
            'Omnu': (self.fiducial_cosmology.cosmo_dic['omnuh2'] /
                     (self.cosmo.cosmo_dic['H0'] / 100.)**2.),
            'tau': self.fiducial_cosmology.cosmo_dic['tau'],
            'mnu': self.fiducial_cosmology.cosmo_dic['mnu'],
            'nnu': self.fiducial_cosmology.cosmo_dic['nnu'],
            'ns': self.fiducial_cosmology.cosmo_dic['ns'],
            'As': self.fiducial_cosmology.cosmo_dic['As'],
            'w': self.fiducial_cosmology.cosmo_dic['w'],
            'wa': self.fiducial_cosmology.cosmo_dic['wa']
        },
            'theory': {'camb':
                       {'stop_at_error': True,
                        'extra_args': {'num_massive_neutrinos': 1,
                                       'dark_energy_model': 'ppf'}}},
            # Likelihood: we load the likelihood as an external function
            'likelihood': {'one': None}}
        # Update fiducial cobaya dictionary with the IST-F
        # Fiducial values of biases
        self.info_fiducial['params'].update(
            self.fiducial_cosmology.cosmo_dic['nuisance_parameters'])
        # Use get_model wrapper for fiducial
        model_fiducial = get_model(self.info_fiducial)
        model_fiducial.add_requirements({"omegam": None,
                                         "omegab": None,
                                         "omegac": None,
                                         "omnuh2": None,
                                         "omeganu": None,
                                         "Pk_interpolator":
                                         {"z": self.z_win,
                                          "k_max": self.k_max_Boltzmannn,
                                          "nonlinear": [False, True],
                                          "vars_pairs": ([["delta_tot",
                                                           "delta_tot"],
                                                          ["Weyl",
                                                           "Weyl"]])},
                                         "comoving_radial_distance":
                                         {"z": self.z_win},
                                         "angular_diameter_distance":
                                         {"z": self.z_win},
                                         "Hubble": {"z": self.z_win,
                                                    "units": "km/s/Mpc"},
                                         "fsigma8": {"z": self.z_win,
                                                     "units": None},
                                         "sigma8_z": {"z": self.z_win}})

        # Evaluation of posterior, required by Cobaya
        model_fiducial.logposterior({})

        # Update fiducial cosmology dictionary
        self.fiducial_cosmology.cosmo_dic['Omc'] = \
            model_fiducial.provider.get_param('omegac')
        self.fiducial_cosmology.cosmo_dic['Omm'] = \
            model_fiducial.provider.get_param('omegam')
        self.fiducial_cosmology.cosmo_dic['Omk'] = \
            model_fiducial.provider.get_param('omk')
        self.fiducial_cosmology.cosmo_dic['z_win'] = self.z_win
        self.fiducial_cosmology.cosmo_dic['k_win'] = self.k_win
        self.fiducial_cosmology.cosmo_dic['comov_dist'] = \
            model_fiducial.provider.get_comoving_radial_distance(
            self.z_win),
        self.fiducial_cosmology.cosmo_dic['angular_dist'] = \
            model_fiducial.provider.get_angular_diameter_distance(
            self.z_win),
        self.fiducial_cosmology.cosmo_dic['H'] = \
            model_fiducial.provider.get_Hubble(
            self.z_win),
        self.fiducial_cosmology.cosmo_dic['H_Mpc'] = \
            model_fiducial.provider.get_Hubble(
            self.z_win, units='1/Mpc'),
        self.fiducial_cosmology.cosmo_dic['Pk_delta'] = \
            model_fiducial.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=False)
        self.fiducial_cosmology.cosmo_dic['Pk_weyl'] = \
            model_fiducial.provider.get_Pk_interpolator(
            ("Weyl", "Weyl"), nonlinear=False),
        self.fiducial_cosmology.cosmo_dic['fsigma8'] = \
            model_fiducial.provider.get_fsigma8(
            self.z_win)
        self.fiducial_cosmology.cosmo_dic['sigma8'] = \
            model_fiducial.provider.get_sigma8_z(
            self.z_win)
        # Update dictionary with interpolators
        self.fiducial_cosmology.update_cosmo_dic(
            self.fiducial_cosmology.cosmo_dic['z_win'],
            0.05)

    def get_requirements(self):
        r"""Get Requirements

        New 'theory needs'. Asks for the theory
        requirements to the theory code via
        Cobaya.

        Returns
        -------
        dictionary specifying quantities i
        calculated by a theory code are needed

        """

        return {"omegam": None,
                "omegab": None,
                "omegac": None,
                "omnuh2": None,
                "omeganu": None,
                "Pk_interpolator":
                {"z": self.z_win,
                 "k_max": self.k_max_Boltzmannn,
                 "nonlinear": False,
                 "vars_pairs": ([["delta_tot",
                                  "delta_tot"],
                                 ["Weyl",
                                  "Weyl"]])},
                "comoving_radial_distance": {"z": self.z_win},
                "angular_diameter_distance": {"z": self.z_win},
                "Hubble": {"z": self.z_win, "units": "km/s/Mpc"},
                "sigma8_z": {"z": self.z_win},
                "fsigma8": {"z": self.z_win, "units": None}}

    def passing_requirements(self, model, **params_dic):
        r"""Passing Requirements

        Gets cosmological quantities from the theory code
        from COBAYA and passes them to an instance of the
        Cosmology class.

        Cosmological quantities are saved in the cosmo_dic
        attribute of the Cosmology class.

        """

        try:
            self.cosmo.cosmo_dic['H0'] = self.provider.get_param("H0")
            self.cosmo.cosmo_dic['H0_Mpc'] = \
                self.cosmo.cosmo_dic['H0'] / const.c.to('km/s').value
            self.cosmo.cosmo_dic['As'] = self.provider.get_param('As')
            self.cosmo.cosmo_dic['ns'] = self.provider.get_param('ns')
            self.cosmo.cosmo_dic['omch2'] = self.provider.get_param('omch2')
            self.cosmo.cosmo_dic['ombh2'] = self.provider.get_param('ombh2')
            self.cosmo.cosmo_dic['Omc'] = self.provider.get_param('omegac')
            self.cosmo.cosmo_dic['Omb'] = self.provider.get_param('omegab')
            self.cosmo.cosmo_dic['Omm'] = self.provider.get_param('omegam')
            self.cosmo.cosmo_dic['Omk'] = self.provider.get_param('omk')
            self.cosmo.cosmo_dic['mnu'] = self.provider.get_param('mnu')
            self.cosmo.cosmo_dic['omnuh2'] = self.provider.get_param('omnuh2')
            self.cosmo.cosmo_dic['Omnu'] = self.provider.get_param('omeganu')
            self.cosmo.cosmo_dic['w'] = self.provider.get_param('w')
            self.cosmo.cosmo_dic['wa'] = self.provider.get_param('wa')
            self.cosmo.cosmo_dic['nnu'] = self.provider.get_param('nnu')
            self.cosmo.cosmo_dic['tau'] = self.provider.get_param('tau')
            self.cosmo.cosmo_dic['comov_dist'] = \
                self.provider.get_comoving_radial_distance(self.z_win)
            self.cosmo.cosmo_dic['angular_dist'] = \
                self.provider.get_angular_diameter_distance(self.z_win)
            self.cosmo.cosmo_dic['H'] = self.provider.get_Hubble(self.z_win)
            self.cosmo.cosmo_dic['H_Mpc'] = \
                self.provider.get_Hubble(self.z_win, units='1/Mpc')
            self.cosmo.cosmo_dic['Pk_delta'] = \
                self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False)
            self.cosmo.cosmo_dic['Pk_weyl'] = \
                self.provider.get_Pk_interpolator(
                ("Weyl", "Weyl"), nonlinear=False)
            self.cosmo.cosmo_dic['z_win'] = self.z_win
            self.cosmo.cosmo_dic['k_win'] = self.k_win
            self.cosmo.cosmo_dic['sigma8'] = self.provider.get_sigma8_z(
                self.cosmo.cosmo_dic['z_win'])
            self.cosmo.cosmo_dic['fsigma8'] = self.provider.get_fsigma8(
                self.cosmo.cosmo_dic['z_win'])
            # Filter nuisance parameters for new dict
            new_keys = params_dic.keys() - self.cosmo.cosmo_dic.keys()
            only_nuisance_params = {your_key: params_dic[your_key]
                                    for your_key in new_keys}
            self.cosmo.cosmo_dic['nuisance_parameters'].update(
                **only_nuisance_params)

        except (TypeError, AttributeError):
            self.cosmo.cosmo_dic['H0'] = model.provider.get_param("H0")
            self.cosmo.cosmo_dic['H0_Mpc'] = \
                self.cosmo.cosmo_dic['H0'] / const.c.to('km/s').value
            self.cosmo.cosmo_dic['As'] = model.provider.get_param('As')
            self.cosmo.cosmo_dic['ns'] = model.provider.get_param('ns')
            self.cosmo.cosmo_dic['omch2'] = model.provider.get_param('omch2')
            self.cosmo.cosmo_dic['ombh2'] = model.provider.get_param('ombh2')
            self.cosmo.cosmo_dic['Omc'] = model.provider.get_param('omegac')
            self.cosmo.cosmo_dic['Omb'] = model.provider.get_param('omegab')
            self.cosmo.cosmo_dic['Omm'] = model.provider.get_param('omegam')
            self.cosmo.cosmo_dic['Omk'] = model.provider.get_param('omk')
            self.cosmo.cosmo_dic['mnu'] = model.provider.get_param('mnu')
            self.cosmo.cosmo_dic['mnu'] = model.provider.get_param('mnu')
            self.cosmo.cosmo_dic['omnuh2'] = model.provider.get_param('omnuh2')
            self.cosmo.cosmo_dic['Omnu'] = model.provider.get_param('omeganu')
            self.cosmo.cosmo_dic['w'] = model.provider.get_param('w')
            self.cosmo.cosmo_dic['wa'] = model.provider.get_param('wa')
            self.cosmo.cosmo_dic['nnu'] = model.provider.get_param('nnu')
            self.cosmo.cosmo_dic['tau'] = model.provider.get_param('tau')
            self.cosmo.cosmo_dic['comov_dist'] = \
                model.provider.get_comoving_radial_distance(self.z_win)
            self.cosmo.cosmo_dic['angular_dist'] = \
                model.provider.get_angular_diameter_distance(self.z_win)
            self.cosmo.cosmo_dic['H'] = model.provider.get_Hubble(self.z_win)
            self.cosmo.cosmo_dic['H_Mpc'] = \
                model.provider.get_Hubble(self.z_win, units='1/Mpc')
            self.cosmo.cosmo_dic['Pk_delta'] = \
                model.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False)
            self.cosmo.cosmo_dic['Pk_weyl'] = \
                model.provider.get_Pk_interpolator(
                ("Weyl", "Weyl"), nonlinear=False)
            self.cosmo.cosmo_dic['z_win'] = self.z_win
            self.cosmo.cosmo_dic['k_win'] = self.k_win
            self.cosmo.cosmo_dic['sigma8'] = model.provider.get_sigma8_z(
                self.cosmo.cosmo_dic['z_win'])
            self.cosmo.cosmo_dic['fsigma8'] = model.provider.get_fsigma8(
                self.cosmo.cosmo_dic['z_win'])
            new_keys = params_dic.keys() - self.cosmo.cosmo_dic.keys()
            only_nuisance_params = {your_key: params_dic[your_key]
                                    for your_key in new_keys}
            self.cosmo.cosmo_dic['nuisance_parameters'].update(
                **only_nuisance_params)

    def logp(self, **params_values):
        r"""Logp

        Executes passing_requirements,
        updates cosmology dictionary,
        calls log_likelihood

        Parameters
        ----------
        **params_values: tuple
              List of (sampled) parameters obtained from
              the theory code or asked by the likelihood
        Returns
        -------
        loglike: float
            value of the function log_likelihood
        """
        model = None
        self.passing_requirements(model, **params_values)
        # Update cosmo_dic to interpolators
        self.cosmo.update_cosmo_dic(self.cosmo.cosmo_dic['z_win'], 0.05)
        loglike = self.likefinal.loglike(self.cosmo.cosmo_dic,
                                         self.fiducial_cosmology.cosmo_dic)

        return loglike
