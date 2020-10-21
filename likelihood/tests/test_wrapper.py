"""General class for COBAYA wrapper
=======

"""

# (GCH): Use Cobaya Model wrapper

from cobaya.model import get_model
from likelihood.cobaya_interface import EuclidLikelihood
import numpy as np


class CobayaModel:

    def __init__(self, cosmo):
        self.cosmology = cosmo
        self.info = {}
        self.model = None
        z_min = 0.0
        z_max = 4.0
        z_samp = 100
        self.z_win = np.linspace(z_min, z_max, z_samp)
        self.k_min = 0.002
        self.k_max = 10.0
        self.k_samp = 100
        self.k_win = np.linspace(self.k_min, self.k_max, self.k_samp)

    def define_info(self, cosmo_inst):
        self.info = {'params': {
            'ombh2': cosmo_inst.cosmo_dic['ombh2'],
            'omch2': cosmo_inst.cosmo_dic['omch2'],
            'omnuh2': cosmo_inst.cosmo_dic['omnuh2'],
            'H0': cosmo_inst.cosmo_dic['H0'],
            'tau': cosmo_inst.cosmo_dic['tau'],
            'mnu': cosmo_inst.cosmo_dic['mnu'],
            'nnu': cosmo_inst.cosmo_dic['nnu'],
            'ns': cosmo_inst.cosmo_dic['ns'],
            'As': cosmo_inst.cosmo_dic['As'],
            'like_selection': 12},
            'theory': {'camb':
                       {'stop_at_error': True,
                        'extra_args': {'num_massive_neutrinos': 1}}},
            # Likelihood: we load the likelihood as an external function
            'likelihood': {'euclid': EuclidLikelihood}}

    def get_cobaya_model(self):
        self.define_info(self.cosmology)
        self.model = get_model(self.info)
        self.model.logposterior({})

    def update_cosmo(self):
        self.get_cobaya_model()
        self.cosmology.cosmo_dic['H0'] = self.model.provider.get_param('H0')
        self.cosmology.cosmo_dic['omch2'] = self.model.provider.get_param(
            'omch2')
        self.cosmology.cosmo_dic['omnuh2'] = self.model.provider.get_param(
            'omnuh2')
        self.cosmology.cosmo_dic['ombh2'] = self.model.provider.get_param(
            'ombh2')
        self.cosmology.cosmo_dic['mnu'] = self.model.provider.get_param('mnu')
        self.cosmology.cosmo_dic['z_win'] = self.z_win
        self.cosmology.cosmo_dic['k_win'] = self.k_win
        self.cosmology.cosmo_dic['comov_dist'] = \
            self.model.provider.get_comoving_radial_distance(
            self.cosmology.cosmo_dic['z_win'])
        self.cosmology.cosmo_dic['H'] = \
            self.model.provider.get_Hubble(
            self.cosmology.cosmo_dic['z_win'])
        self.cosmology.cosmo_dic['angular_dist'] = \
            self.model.provider.get_angular_diameter_distance(
            self.cosmology.cosmo_dic['z_win'])
        self.cosmology.cosmo_dic['Pk_interpolator'] = \
            self.model.provider.get_Pk_interpolator(nonlinear=False)
        self.cosmology.cosmo_dic['Pk_delta'] = \
            self.model.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=False)
        self.cosmology.cosmo_dic['fsigma8'] = \
            self.model.provider.get_fsigma8(self.z_win)
        R, z, sigma_R = \
            self.model.provider.get_sigma_R()
        self.cosmology.cosmo_dic['sigma_8'] = \
            sigma_R[:, 0]
        self.cosmology.update_cosmo_dic(self.cosmology.cosmo_dic['z_win'],
                                        0.05)
