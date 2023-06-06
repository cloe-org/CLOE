"""COBAYA MODEL HANDLER

General class for Cobaya wrapper.

"""

# Use Cobaya Model wrapper

from cobaya.model import get_model
from cloe.cobaya_interface import EuclidLikelihood
import numpy as np
from cloe.tests.test_input.data import mock_data
from scipy import interpolate
from cloe.auxiliary.likelihood_yaml_handler import set_halofit_version


class CobayaModel:
    """Cobaya model."""

    def __init__(self, cosmo):
        self.cosmology = cosmo
        self.info = {}
        self.model = None
        z_min = 0.0
        z_max = 4.0
        z_samp = 100
        self.z_win = np.linspace(z_min, z_max, z_samp)
        # Note that this k_min does not currently interface with Cobaya
        # Once the Cobaya interface is adjusted to use a set k_min, the same
        # value should be used here.
        self.k_max_Boltzmann = 10.0
        self.k_min_GC_phot_interp = 0.001
        self.k_max_GC_phot_interp = 50.0
        self.k_samp_GC = 100
        self.k_win = np.logspace(np.log10(self.k_min_GC_phot_interp),
                                 np.log10(self.k_max_GC_phot_interp),
                                 self.k_samp_GC)
        self.cosmology.cosmo_dic['luminosity_ratio_z_func'] = \
            interpolate.InterpolatedUnivariateSpline(
                self.z_win,
                np.linspace(0.001, 1.7, 100))
        self.cosmology.nonlinear.theory['redshift_bins'] = \
            self.cosmology.cosmo_dic['redshift_bins']
        self.cosmology.nonlinear.set_Pgg_spectro_model()

    def define_info(self, cosmo_inst):
        """Defines information."""
        self.info = {'params': {
            'ombh2': cosmo_inst.cosmo_dic['ombh2'],
            'omch2': cosmo_inst.cosmo_dic['omch2'],
            'omnuh2': cosmo_inst.cosmo_dic['omnuh2'],
            'omegam': None,
            'omegab': None,
            'omeganu': None,
            'omegac': None,
            'omk': cosmo_inst.cosmo_dic['Omk'],
            'H0': cosmo_inst.cosmo_dic['H0'],
            'tau': cosmo_inst.cosmo_dic['tau'],
            'mnu': cosmo_inst.cosmo_dic['mnu'],
            'nnu': cosmo_inst.cosmo_dic['nnu'],
            'ns': cosmo_inst.cosmo_dic['ns'],
            'As': cosmo_inst.cosmo_dic['As'],
            'w': cosmo_inst.cosmo_dic['w'],
            'wa': cosmo_inst.cosmo_dic['wa']},
            'theory': {'camb':
                       {'stop_at_error': True,
                        'extra_args': {'num_massive_neutrinos': 1,
                                       'dark_energy_model': 'ppf'}}},
            # Likelihood: we load the likelihood as an external function
            'likelihood': {'euclid': {
                'external': EuclidLikelihood,
                'NL_flag_phot_matter':
                    cosmo_inst.cosmo_dic['NL_flag_phot_matter'],
                'NL_flag_spectro': cosmo_inst.cosmo_dic['NL_flag_spectro']}}}
        self.info['params'].update(
            cosmo_inst.cosmo_dic['nuisance_parameters'])
        self.info['data'] = mock_data

        set_halofit_version(self.info,
                            cosmo_inst.cosmo_dic['NL_flag_phot_matter'])

    def get_cobaya_model(self):
        """Gets Cobaya model."""
        self.define_info(self.cosmology)
        self.model = get_model(self.info)
        self.model.logposterior({})

    def update_cosmo(self):
        """Updates cosmology."""
        self.get_cobaya_model()
        self.cosmology.cosmo_dic['H0'] = self.model.provider.get_param('H0')
        self.cosmology.cosmo_dic['omch2'] = self.model.provider.get_param(
            'omch2')
        self.cosmology.cosmo_dic['omnuh2'] = self.model.provider.get_param(
            'omnuh2')
        self.cosmology.cosmo_dic['ombh2'] = self.model.provider.get_param(
            'ombh2')
        self.cosmology.cosmo_dic['Omc'] = \
            self.model.provider.get_param('omegac')
        self.cosmology.cosmo_dic['Omm'] = \
            self.model.provider.get_param('omegam')
        self.cosmology.cosmo_dic['Omk'] = \
            self.model.provider.get_param('omk')
        self.cosmology.cosmo_dic['Omnu'] = \
            self.model.provider.get_param('omeganu')
        self.cosmology.cosmo_dic['mnu'] = self.model.provider.get_param('mnu')
        self.cosmology.cosmo_dic['z_win'] = self.z_win
        self.cosmology.cosmo_dic['k_win'] = self.k_win
        self.cosmology.cosmo_dic['comov_dist'] = \
            self.model.provider.get_comoving_radial_distance(
            self.cosmology.cosmo_dic['z_win'])
        self.cosmology.cosmo_dic['H'] = \
            self.model.provider.get_Hubble(
            self.cosmology.cosmo_dic['z_win'])
        self.cosmology.cosmo_dic['H_Mpc'] = \
            self.model.provider.get_Hubble(
            self.cosmology.cosmo_dic['z_win'], units='1/Mpc')
        self.cosmology.cosmo_dic['angular_dist'] = \
            self.model.provider.get_angular_diameter_distance(
            self.cosmology.cosmo_dic['z_win'])
        self.cosmology.cosmo_dic['Pk_delta'] = \
            self.model.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=False)
        self.cosmology.cosmo_dic['Pk_cb'] = \
            self.model.provider.get_Pk_interpolator(
            ("delta_nonu", "delta_nonu"), nonlinear=False)
        self.cosmology.cosmo_dic['Pk_weyl'] = \
            self.model.provider.get_Pk_interpolator(
            ("Weyl", "Weyl"), nonlinear=False)
        if self.cosmology.cosmo_dic['NL_flag_phot_matter'] > 0:
            self.cosmology.cosmo_dic['Pk_halomodel_recipe'] = \
                self.model.provider.get_Pk_interpolator(
                ('delta_tot', 'delta_tot'), nonlinear=True)
        self.cosmology.cosmo_dic['fsigma8'] = \
            self.model.provider.get_fsigma8(self.z_win)
        self.cosmology.cosmo_dic['sigma8'] = \
            self.model.provider.get_sigma8_z(self.z_win)
        self.cosmology.update_cosmo_dic(self.cosmology.cosmo_dic['z_win'],
                                        0.05)
