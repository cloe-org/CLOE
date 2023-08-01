"""COBAYA INTERFACE

Contains the interface with Cobaya. It defines the Euclid Likelihood and
inherits from the Cobaya likelihood :py:mod:`cobaya.likelihood.Likelihood`.
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
from cloe.auxiliary.params_converter import camb_to_classy

# Error classes


class CobayaInterfaceError(Exception):
    r"""
    Class to define Exception Error.
    """

    pass


class EuclidLikelihood(Likelihood):
    r"""
    Class to define the :obj:`EuclidLikelihood`.

    Inherits from the :obj:`Likelihood` class
    of Cobaya.
    """

    def initialize(self):
        r"""Initialise.

        Sets up values for initial variables
        and creates instance of :obj:`cosmology` class.

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
        self.observables['selection']['matrix_transform_phot'] = \
            self.matrix_transform_phot
        if 'GCspectro' in self.observables['specifications'].keys():
            self.observables['specifications']['GCspectro']['statistics'] = \
                self.statistics_spectro
        # set the parameters used to determine what statistics to use
        # for the photometric probes
        for key in self.observables['specifications'].keys():
            if key in ['WL', 'GCphot', 'WL-GCphot']:
                self.observables['specifications'][key]['statistics'] = \
                    self.statistics_photo
        # Select which power spectra to require from the Boltzmann solver
        if self.NL_flag_phot_matter > 0:
            self.use_NL = [False, True]
        else:
            self.use_NL = False
        # Initialize Euclike module
        self.likefinal = Euclike(self.data, self.observables)
        # Here we set the naming convention for the cosmological parameters
        # accepted by the selected Boltzmann solver
        if not self.solver:
            warnings.warn('Boltzmann solver not specified at instantiation '
                          'of EuclidLikelihood class. Default set to CAMB')
            self.solver = 'camb'
        if self.solver == 'camb':
            self.pnames = \
                dict(zip(camb_to_classy.keys(), camb_to_classy.keys()))
        elif self.solver == 'classy':
            self.pnames = \
                dict(zip(camb_to_classy.keys(), camb_to_classy.values()))
        # Initialize Cosmology class for sampling
        self.cosmo = Cosmology()
        # Adding GCspectro redshift bins to cosmo dictionary and setting up
        # the internal class for Pgg_spectro with this information.
        self.cosmo.nonlinear.theory['redshift_bins_means_spectro'] = \
            self.data['spectro']['edges']
        self.cosmo.nonlinear.set_Pgg_spectro_model()
        self.cosmo.cosmo_dic['redshift_bins_means_spectro'] = \
            self.data['spectro']['edges']
        # Adding Redshift bins means for the photo catalogue
        self.cosmo.cosmo_dic['redshift_bins_means_phot'] = \
            self.data['photo']['redshifts']
        # Initialize the fiducial model
        self.set_fiducial_cosmology()
        # "Here we add the fiducial Hubble function (fid_H_z_func),
        # comoving distance (fid_r_z_func),
        # and angular diameter distance (fid_d_z_func)
        # to the cosmo dictionary
        # These quantities
        # are requested in different parts of the spectro and photo class.
        self.cosmo.cosmo_dic['fid_d_z_func'] = \
            self.fiducial_cosmology.cosmo_dic['d_z_func']
        self.cosmo.cosmo_dic['fid_r_z_func'] = \
            self.fiducial_cosmology.cosmo_dic['r_z_func']
        self.cosmo.cosmo_dic['fid_H_z_func'] = \
            self.fiducial_cosmology.cosmo_dic['H_z_func']
        # Create a separate dictionary with fiducial cosmo quantities that are
        # available at initialization, before cosmo_dic is created.
        self.likefinal.fiducial_cosmo_quantities_dic.update(
            self.fiducial_cosmology.cosmo_dic)
        # Compute the data vectors
        # and initialize possible matrix transforms
        self.likefinal.get_masked_data()
        # Add the luminosity_ratio_z_func to the cosmo_dic after data has been
        # read and stored in the data_ins attribute of Euclike
        self.cosmo.cosmo_dic['luminosity_ratio_z_func'] = \
            self.likefinal.data_ins.luminosity_ratio_interpolator
        # Pass the observables selection to the cosmo dictionary
        self.cosmo.cosmo_dic['obs_selection'] = self.observables['selection']

    def set_fiducial_cosmology(self):
        r"""Sets the fiducial cosmology class.

        Method that reads the input fiducial cosmology from the instance of
        the :obj:`EuclidLikelihood` class, and sets up a
        dedicated :obj:`cosmology` class.
        """
        # This will work if CAMB is installed globally
        self.fiducial_cosmology = Cosmology()
        # Update fiducial cosmo dic with fiducial info from reader
        self.fiducial_cosmology.cosmo_dic.update(
            self.likefinal.data_spectro_fiducial_cosmo)
        self.info_fiducial = {
            'params': {
                self.pnames['ombh2']:
                    self.fiducial_cosmology.cosmo_dic['ombh2'],
                self.pnames['omch2']:
                    self.fiducial_cosmology.cosmo_dic['omch2'],
                self.pnames['omnuh2']:
                    self.fiducial_cosmology.cosmo_dic['omnuh2'],
                self.pnames['omk']: self.fiducial_cosmology.cosmo_dic['Omk'],
                'H0': self.fiducial_cosmology.cosmo_dic['H0'],
                self.pnames['tau']: self.fiducial_cosmology.cosmo_dic['tau'],
                self.pnames['mnu']: self.fiducial_cosmology.cosmo_dic['mnu'],
                self.pnames['ns']: self.fiducial_cosmology.cosmo_dic['ns'],
                self.pnames['As']: self.fiducial_cosmology.cosmo_dic['As'],
                self.pnames['w']: self.fiducial_cosmology.cosmo_dic['w'],
                self.pnames['wa']: self.fiducial_cosmology.cosmo_dic['wa']
            },
            'theory': {
                self.solver: {
                    'stop_at_error': True,
                    'extra_args': {}
                }
            },
            # Likelihood: we load the likelihood as an external function
            'likelihood': {
                'one': None
            }
        }
        if self.solver == 'camb':
            (self.info_fiducial['theory']['camb']['extra_args']
             ['num_massive_neutrinos']) = 1
            self.info_fiducial['params']['nnu'] = \
                self.fiducial_cosmology.cosmo_dic['nnu']
        elif self.solver == 'classy':
            nrad = 4.41e-3
            self.info_fiducial['params']['N_ncdm'] = 1
            self.info_fiducial['params']['N_ur'] = \
                ((self.fiducial_cosmology.cosmo_dic['nnu'] - nrad) *
                 (3.0 - self.info_fiducial['params']['N_ncdm']) / 3.0 +
                 nrad)
            self.info_fiducial['params']['Omega_Lambda'] = 0.0

        # Update fiducial cobaya dictionary with the IST-F
        # Fiducial values of biases
        # With classy, this makes the code break
        # self.info_fiducial['params'].update(
        #    self.fiducial_cosmology.cosmo_dic['nuisance_parameters'])
        # Use get_model wrapper for fiducial
        model_fiducial = get_model(self.info_fiducial)
        model_fiducial.add_requirements({
            'omegam': None,
            'Pk_interpolator': {
                'z': self.z_win,
                'k_max': self.k_max_Boltzmann,
                'nonlinear': False,
                'vars_pairs': ([['delta_tot', 'delta_tot'],
                                ['delta_nonu', 'delta_nonu'],
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
        self.fiducial_cosmology.cosmo_dic['Omm'] = \
            model_fiducial.provider.get_param('omegam')
        self.fiducial_cosmology.cosmo_dic['Omk'] = \
            model_fiducial.provider.get_param(self.pnames['omk'])
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
        self.fiducial_cosmology.cosmo_dic['Pk_cb'] = \
            model_fiducial.provider.get_Pk_interpolator(
            ('delta_nonu', 'delta_nonu'), nonlinear=False,
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
        # In order to make the update_cosmo_dic method to work, we need to
        # specify also in this case the information on the GCspectro bins
        # and Photo bins
        self.fiducial_cosmology.cosmo_dic['redshift_bins_means_spectro'] = \
            self.data['spectro']['edges']
        self.fiducial_cosmology.nonlinear.theory[
            'redshift_bins_means_spectro'] = \
            self.data['spectro']['edges']
        self.fiducial_cosmology.cosmo_dic['redshift_bins_means_phot'] = \
            self.data['photo']['redshifts']
        self.fiducial_cosmology.nonlinear.set_Pgg_spectro_model()
        # Update dictionary with interpolators
        self.fiducial_cosmology.cosmo_dic['luminosity_ratio_z_func'] = \
            self.likefinal.data_ins.luminosity_ratio_interpolator
        self.fiducial_cosmology.update_cosmo_dic(
            self.fiducial_cosmology.cosmo_dic['z_win'], 0.05)

    def get_requirements(self):
        r"""Gets requirements.

        Updates new `theory needs`. Asks for the theory
        requirements to the theory code via Cobaya.

        Returns
        -------
        Dictionary: dict
             Dictionary with quantitities
             calculated by the theory code

        """
        requirements = \
            {'omegam': None,
             'Pk_interpolator':
                {'z': self.z_win,
                 'k_max': self.k_max_Boltzmann,
                 'nonlinear': self.use_NL,
                 'vars_pairs': ([['delta_tot', 'delta_tot'],
                                 ['delta_nonu', 'delta_nonu'],
                                 ['Weyl', 'Weyl']])},
                'comoving_radial_distance': {'z': self.z_win},
                'angular_diameter_distance': {'z': self.z_win},
                'Hubble': {'z': self.z_win, 'units': 'km/s/Mpc'},
                'sigma8_z': {'z': self.z_win},
                'fsigma8': {'z': self.z_win, 'units': None}}
        if self.solver == 'camb':
            derived = {'omegac': None, 'omnuh2': None, 'omeganu': None,
                       'nnu': None}
            requirements = requirements | derived

        return requirements

    def passing_requirements(self, model, info, **params_dic):
        r"""Passing requirements.

        Gets cosmological quantities from the theory code
        from Cobaya and passes them to an instance of the
        Cosmology class.

        Cosmological quantities are saved in the
        cosmology dictionary attribute of the Cosmology class.

        """

        try:
            self.cosmo.cosmo_dic['NL_flag_phot_matter'] = \
                self.NL_flag_phot_matter
            self.cosmo.cosmo_dic['NL_flag_spectro'] = self.NL_flag_spectro
            self.cosmo.cosmo_dic['bias_model'] = self.bias_model
            self.cosmo.cosmo_dic['magbias_model'] = self.magbias_model
            self.cosmo.cosmo_dic['use_gamma_MG'] = self.use_gamma_MG
            self.cosmo.cosmo_dic['matrix_transform_phot'] = \
                self.matrix_transform_phot
            self.cosmo.cosmo_dic['H0'] = self.provider.get_param('H0')
            self.cosmo.cosmo_dic['H0_Mpc'] = \
                self.cosmo.cosmo_dic['H0'] / const.c.to('km/s').value
            self.cosmo.cosmo_dic['tau'] = \
                self.provider.get_param(self.pnames['tau'])
            self.cosmo.cosmo_dic['As'] = \
                self.provider.get_param(self.pnames['As'])
            self.cosmo.cosmo_dic['ns'] = \
                self.provider.get_param(self.pnames['ns'])
            self.cosmo.cosmo_dic['omch2'] = \
                self.provider.get_param(self.pnames['omch2'])
            self.cosmo.cosmo_dic['ombh2'] = \
                self.provider.get_param(self.pnames['ombh2'])
            self.cosmo.cosmo_dic['Omk'] = \
                self.provider.get_param(self.pnames['omk'])
            try:
                self.cosmo.cosmo_dic['mnu'] = \
                    self.provider.get_param(self.pnames['mnu'])
            except KeyError:
                self.cosmo.cosmo_dic['omnuh2'] = \
                    self.provider.get_param(self.pnames['omnuh2'])
            self.cosmo.cosmo_dic['w'] = \
                self.provider.get_param(self.pnames['w'])
            self.cosmo.cosmo_dic['wa'] = \
                self.provider.get_param(self.pnames['wa'])
            self.cosmo.cosmo_dic['Omm'] = self.provider.get_param('omegam')
            if self.use_gamma_MG:
                self.cosmo.cosmo_dic['gamma_MG'] = \
                    self.provider.get_param('gamma_MG')
            if self.solver == 'camb':
                self.cosmo.cosmo_dic['Omc'] = self.provider.get_param('omegac')
                self.cosmo.cosmo_dic['omnuh2'] = \
                    self.provider.get_param('omnuh2')
                self.cosmo.cosmo_dic['Omnu'] = \
                    self.provider.get_param('omeganu')
                self.cosmo.cosmo_dic['nnu'] = self.provider.get_param('nnu')
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
            self.cosmo.cosmo_dic['Pk_cb'] = \
                self.provider.get_Pk_interpolator(
                ('delta_nonu', 'delta_nonu'), nonlinear=False,
                extrap_kmin=self.k_min_extrap,
                extrap_kmax=self.k_max_extrap)
            self.cosmo.cosmo_dic['Pk_weyl'] = \
                self.provider.get_Pk_interpolator(
                ('Weyl', 'Weyl'), nonlinear=False,
                extrap_kmax=self.k_max_extrap)
            if self.NL_flag_phot_matter > 0:
                self.cosmo.cosmo_dic['Pk_halomodel_recipe'] = \
                    self.provider.get_Pk_interpolator(
                    ('delta_tot', 'delta_tot'), nonlinear=True,
                    extrap_kmin=self.k_min_extrap,
                    extrap_kmax=self.k_max_extrap)
                self.cosmo.cosmo_dic['Pk_weyl_NL'] = \
                    self.provider.get_Pk_interpolator(
                    ('Weyl', 'Weyl'), nonlinear=True,
                    extrap_kmin=self.k_min_extrap,
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
            self.cosmo.cosmo_dic['NL_flag_phot_matter'] = \
                info['likelihood']['Euclid']['NL_flag_phot_matter']
            self.cosmo.cosmo_dic['NL_flag_spectro'] = \
                info['likelihood']['Euclid']['NL_flag_spectro']
            self.cosmo.cosmo_dic['bias_model'] = self.bias_model
            self.cosmo.cosmo_dic['add_phot_RSD'] = \
                info['likelihood']['Euclid']['add_phot_RSD']
            self.matrix_transform_phot = \
                info['likelihood']['Euclid']['matrix_transform_phot']
            self.cosmo.cosmo_dic['matrix_transform_phot'] = \
                info['likelihood']['Euclid']['matrix_transform_phot']
            self.cosmo.cosmo_dic['magbias_model'] = self.magbias_model
            self.cosmo.cosmo_dic['use_gamma_MG'] = self.use_gamma_MG
            self.cosmo.cosmo_dic['H0'] = model.provider.get_param('H0')
            self.cosmo.cosmo_dic['H0_Mpc'] = \
                self.cosmo.cosmo_dic['H0'] / const.c.to('km/s').value
            self.cosmo.cosmo_dic['tau'] = \
                model.provider.get_param(self.pnames['tau'])
            self.cosmo.cosmo_dic['As'] = \
                model.provider.get_param(self.pnames['As'])
            self.cosmo.cosmo_dic['ns'] = \
                model.provider.get_param(self.pnames['ns'])
            self.cosmo.cosmo_dic['omch2'] = \
                model.provider.get_param(self.pnames['omch2'])
            self.cosmo.cosmo_dic['ombh2'] = \
                model.provider.get_param(self.pnames['ombh2'])
            self.cosmo.cosmo_dic['Omk'] = \
                model.provider.get_param(self.pnames['omk'])
            try:
                self.cosmo.cosmo_dic['mnu'] = \
                    model.provider.get_param(self.pnames['mnu'])
            except KeyError:
                self.cosmo.cosmo_dic['omnuh2'] = \
                    model.provider.get_param(self.pnames['omnuh2'])
            self.cosmo.cosmo_dic['mnu'] = \
                model.provider.get_param(self.pnames['mnu'])
            self.cosmo.cosmo_dic['w'] = \
                model.provider.get_param(self.pnames['w'])
            self.cosmo.cosmo_dic['wa'] = \
                model.provider.get_param(self.pnames['wa'])
            self.cosmo.cosmo_dic['Omm'] = model.provider.get_param('omegam')
            if self.solver == 'camb':
                self.cosmo.cosmo_dic['Omc'] = \
                    model.provider.get_param('omegac')
                self.cosmo.cosmo_dic['omnuh2'] = \
                    model.provider.get_param('omnuh2')
                self.cosmo.cosmo_dic['Omnu'] = \
                    model.provider.get_param('omeganu')
                self.cosmo.cosmo_dic['nnu'] = model.provider.get_param('nnu')
            self.cosmo.cosmo_dic['Omm'] = model.provider.get_param('omegam')
            if self.use_gamma_MG:
                self.cosmo.cosmo_dic['gamma_MG'] = \
                    model.provider.get_param('gamma_MG')
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
            self.cosmo.cosmo_dic['Pk_cb'] = \
                model.provider.get_Pk_interpolator(
                ('delta_nonu', 'delta_nonu'), nonlinear=False,
                extrap_kmin=self.k_min_extrap,
                extrap_kmax=self.k_max_extrap)
            self.cosmo.cosmo_dic['Pk_weyl'] = \
                model.provider.get_Pk_interpolator(
                ('Weyl', 'Weyl'), nonlinear=False,
                extrap_kmin=self.k_min_extrap,
                extrap_kmax=self.k_max_extrap)
            if info['likelihood']['Euclid']['NL_flag_phot_matter'] > 0:
                self.cosmo.cosmo_dic['Pk_halomodel_recipe'] = \
                    model.provider.get_Pk_interpolator(
                    ('delta_tot', 'delta_tot'), nonlinear=True,
                    extrap_kmin=self.k_min_extrap,
                    extrap_kmax=self.k_max_extrap)
                self.cosmo.cosmo_dic['Pk_weyl_NL'] = \
                    model.provider.get_Pk_interpolator(
                    ('Weyl', 'Weyl'), nonlinear=True,
                    extrap_kmin=self.k_min_extrap,
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
            if 'observables_selection' in info['likelihood']['Euclid']:
                self.observables_selection = \
                    info['likelihood']['Euclid']['observables_selection']
            self.observables = \
                observables_selection_specifications_checker(
                    info['likelihood']['Euclid']['observables_selection'],
                    self.observables_specifications)

    def logp(self, **params_values):
        r"""Logp.

        Executes ``passing_requirements``,
        updates cosmology dictionary,
        calls :obj:`log_likelihood`.

        Parameters
        ----------
        **params_values: tuple
              List of (sampled) parameters obtained from
              the theory code or asked by the likelihood
        Returns
        -------
        Likelihood: float
            Value of the function :obj:`log_likelihood`
        """
        model = None
        info = None
        self.passing_requirements(model, info, **params_values)
        # Update cosmo_dic to interpolators
        self.cosmo.update_cosmo_dic(self.cosmo.cosmo_dic['z_win'], 0.05)
        # Compute number of sampled parameters
        npar = 0
        for key in self.provider.model.sampled_dependence.keys():
            if any(isinstance(
                    self.provider.model.sampled_dependence[key][i],
                    EuclidLikelihood) for i in
                    range(len(self.provider.model.sampled_dependence[key]))):
                npar += 1
        loglike = self.likefinal.loglike(self.cosmo.cosmo_dic, npar)
        return loglike
