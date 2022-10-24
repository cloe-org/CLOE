"""COBAYA INTERFACE
"""


# General import
import numpy as np
from astropy import constants as const
import warnings

# Cobaya import of general Likelihood class
from cobaya.likelihood import Likelihood
# Import cobaya model wrapper for fiducial model (TEMPORARY)
from cobaya.model import get_model

# Import CLOE classes and functions
from cloe.cosmo.cosmology import Cosmology
from cloe.like_calc.euclike import Euclike
from cloe.auxiliary.observables_dealer import *

# Error classes


class CobayaInterfaceError(Exception):
    r"""
    Class to define Exception Error
    """

    pass


class EuclidLikelihood(Likelihood):
    r"""
    Class to define the Euclid Likelihood

    Inherits from the Likelihood class
    of Cobaya
    """

    def initialize(self):
        r"""Initialize

        Set up values for initial variables
        and create instance of Cosmology class

        """

        self.k_max_Boltzmann = 50.0
        if self.k_min_extrap < 1E-5:
            warnings.warn(
                'WARNING: the requested extrapolated k_min is too low. \
                k_min_extrap changed to 1E-5')
            self.k_min_extrap = 1E-5
        self.k_win = np.logspace(np.log10(self.k_min_extrap),
                                 np.log10(self.k_max_extrap),
                                 self.k_samp)
        self.z_win = np.linspace(self.z_min, self.z_max, self.z_samp)
        # Check the selection and specification requirements
        self.observables = \
            observables_selection_specifications_checker(
                self.observables_selection,
                self.observables_specifications)
        # Visualization of the observables matrix
        if self.plot_observables_selection:
            self.observables_pf = observables_visualization(
             self.observables['selection'])
        self.observables['selection']['add_phot_RSD'] = self.add_phot_RSD
        # Select which power spectra to require from the Boltzmann solver
        if self.NL_flag > 0:
            self.use_NL = [False, True]
        else:
            self.use_NL = False
        # Initialize Euclike module
        self.likefinal = Euclike(self.data, self.observables)

        # Initialize Cosmology class for sampling
        self.cosmo = Cosmology()

        # Initialize the fiducial model
        self.set_fiducial_cosmology()

        # Here we add the fiducial angular diameter distance and Hubble factor
        # to the cosmo dictionary. In this way we can avoid passing the whole
        # fiducial dictionary
        self.cosmo.cosmo_dic['fid_d_z_func'] = \
            self.fiducial_cosmology.cosmo_dic['d_z_func']
        self.cosmo.cosmo_dic['fid_H_z_func'] = \
            self.fiducial_cosmology.cosmo_dic['H_z_func']

        self.cosmo.cosmo_dic['redshift_bins'] = self.data['spectro']['edges']

    def set_fiducial_cosmology(self):
        r"""Sets the fiducial cosmology class

        This method reads the input fiducial cosmology from the instance of
        the EuclidLikelihood class, and sets up a dedicated Cosmology class.
        """
        # This will work if CAMB is installed globally
        self.fiducial_cosmology = Cosmology()
        # Update fiducial cosmo dic with fiducial info from reader
        self.fiducial_cosmology.cosmo_dic.update(
            self.likefinal.data_spectro_fiducial_cosmo)
        self.info_fiducial = {
            'params': {
                'ombh2': self.fiducial_cosmology.cosmo_dic['ombh2'],
                'omch2': self.fiducial_cosmology.cosmo_dic['omch2'],
                'omnuh2': self.fiducial_cosmology.cosmo_dic['omnuh2'],
                'omk': self.fiducial_cosmology.cosmo_dic['Omk'],
                'H0': self.fiducial_cosmology.cosmo_dic['H0'],
                'H0_Mpc': (self.cosmo.cosmo_dic['H0'] /
                           const.c.to('km/s').value),
                'Omnu': (self.fiducial_cosmology.cosmo_dic['omnuh2'] /
                         (self.cosmo.cosmo_dic['H0'] / 100.0)**2.0),
                'tau': self.fiducial_cosmology.cosmo_dic['tau'],
                'mnu': self.fiducial_cosmology.cosmo_dic['mnu'],
                'nnu': self.fiducial_cosmology.cosmo_dic['nnu'],
                'ns': self.fiducial_cosmology.cosmo_dic['ns'],
                'As': self.fiducial_cosmology.cosmo_dic['As'],
                'w': self.fiducial_cosmology.cosmo_dic['w'],
                'wa': self.fiducial_cosmology.cosmo_dic['wa']
            },
            'theory': {
                'camb': {
                    'stop_at_error': True,
                    'extra_args': {
                        'num_massive_neutrinos': 1,
                        'dark_energy_model': 'ppf'
                    }
                }
            },
            # Likelihood: we load the likelihood as an external function
            'likelihood': {
                'one': None
            }
        }

        # Update fiducial cobaya dictionary with the IST-F
        # Fiducial values of biases
        self.info_fiducial['params'].update(
            self.fiducial_cosmology.cosmo_dic['nuisance_parameters'])
        # Use get_model wrapper for fiducial
        model_fiducial = get_model(self.info_fiducial)
        model_fiducial.add_requirements({
            'omegam': None,
            'omegab': None,
            'omegac': None,
            'omnuh2': None,
            'omeganu': None,
            'Pk_interpolator': {
                'z': self.z_win,
                'k_max': self.k_max_Boltzmann,
                'nonlinear': False,
                'vars_pairs': ([['delta_tot', 'delta_tot'],
                                ['Weyl', 'Weyl']])
            },
            'comoving_radial_distance': {
                'z': self.z_win
            },
            'angular_diameter_distance': {
                'z': self.z_win
            },
            'Hubble': {
                'z': self.z_win,
                'units': 'km/s/Mpc'
            },
            'fsigma8': {
                'z': self.z_win,
                'units': None
            },
            'sigma8_z': {
                'z': self.z_win
            }
        })

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
            ('delta_tot', 'delta_tot'), nonlinear=False,
            extrap_kmin=self.k_min_extrap,
            extrap_kmax=self.k_max_extrap)
        self.fiducial_cosmology.cosmo_dic['Pk_weyl'] = \
            model_fiducial.provider.get_Pk_interpolator(
            ('Weyl', 'Weyl'), nonlinear=False,
            extrap_kmax=self.k_max_extrap),
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

        return {'omegam': None,
                'omegab': None,
                'omegac': None,
                'omnuh2': None,
                'omeganu': None,
                'Pk_interpolator':
                {'z': self.z_win,
                 'k_max': self.k_max_Boltzmann,
                 'nonlinear': self.use_NL,
                 'vars_pairs': ([['delta_tot',
                                  'delta_tot'],
                                 ['Weyl',
                                  'Weyl']])},
                'comoving_radial_distance': {'z': self.z_win},
                'angular_diameter_distance': {'z': self.z_win},
                'Hubble': {'z': self.z_win, 'units': 'km/s/Mpc'},
                'sigma8_z': {'z': self.z_win},
                'fsigma8': {'z': self.z_win, 'units': None},
                # temporary, see #767
                'CAMBdata': None}

    def passing_requirements(self, model, info, **params_dic):
        r"""Passing Requirements

        Gets cosmological quantities from the theory code
        from COBAYA and passes them to an instance of the
        Cosmology class.

        Cosmological quantities are saved in the cosmo_dic
        attribute of the Cosmology class.

        """

        try:
            # TEMPORARY (UNTIL JESUS GIVES US THE REQUIREMENTS), see #767
            self.cosmo.cosmo_dic['CAMBdata'] = self.provider.get_CAMBdata()
            # ------------------------------------------------
            self.cosmo.cosmo_dic['NL_flag'] = self.NL_flag
            self.cosmo.cosmo_dic['use_gamma_MG'] = self.use_gamma_MG
            self.cosmo.cosmo_dic['H0'] = self.provider.get_param('H0')
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
            self.cosmo.cosmo_dic['gamma_MG'] = \
                self.provider.get_param('gamma_MG')
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
                ('delta_tot', 'delta_tot'), nonlinear=False,
                extrap_kmin=self.k_min_extrap,
                extrap_kmax=self.k_max_extrap)
            self.cosmo.cosmo_dic['Pk_weyl'] = \
                self.provider.get_Pk_interpolator(
                ('Weyl', 'Weyl'), nonlinear=False,
                extrap_kmax=self.k_max_extrap)
            if self.NL_flag > 0:
                self.cosmo.cosmo_dic['Pk_halofit'] = \
                    self.provider.get_Pk_interpolator(
                    ('delta_tot', 'delta_tot'), nonlinear=True,
                    extrap_kmax=self.k_max_extrap)
                self.cosmo.cosmo_dic['Pk_weyl_NL'] = \
                    self.provider.get_Pk_interpolator(
                    ('Weyl', 'Weyl'), nonlinear=True,
                    extrap_kmax=self.k_max_extrap)
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
            self.cosmo.cosmo_dic['CAMBdata'] = model.provider.get_CAMBdata()
            self.cosmo.cosmo_dic['NL_flag'] = \
                info['likelihood']['Euclid']['NL_flag']
            self.cosmo.cosmo_dic['use_gamma_MG'] = self.use_gamma_MG
            self.cosmo.cosmo_dic['H0'] = model.provider.get_param('H0')
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
            self.cosmo.cosmo_dic['gamma_MG'] = \
                model.provider.get_param('gamma_MG')
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
                ('delta_tot', 'delta_tot'), nonlinear=False,
                extrap_kmin=self.k_min_extrap,
                extrap_kmax=self.k_max_extrap)
            self.cosmo.cosmo_dic['Pk_weyl'] = \
                model.provider.get_Pk_interpolator(
                ('Weyl', 'Weyl'), nonlinear=False,
                extrap_kmax=self.k_max_extrap)
            if info['likelihood']['Euclid']['NL_flag'] > 0:
                self.cosmo.cosmo_dic['Pk_halofit'] = \
                    model.provider.get_Pk_interpolator(
                    ('delta_tot', 'delta_tot'), nonlinear=True,
                    extrap_kmax=self.k_max_extrap)
                self.cosmo.cosmo_dic['Pk_weyl_NL'] = \
                    model.provider.get_Pk_interpolator(
                    ('Weyl', 'Weyl'), nonlinear=True,
                    extrap_kmax=self.k_max_extrap)
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
            if 'observables_specifications' in info['likelihood']['Euclid']:
                self.observables_specifications = \
                    info['likelihood']['Euclid']['observables_specifications']
            self.observables = \
                observables_selection_specifications_checker(
                    info['likelihood']['Euclid']['observables_selection'],
                    self.observables_specifications)

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
        info = None
        self.passing_requirements(model, info, **params_values)
        # Update cosmo_dic to interpolators
        self.cosmo.update_cosmo_dic(self.cosmo.cosmo_dic['z_win'], 0.05)
        loglike = self.likefinal.loglike(self.cosmo.cosmo_dic)
        return loglike
