from cobaya.run import run
from likelihood.cobaya_interface import EuclidLikelihood
import numpy as np
from copy import deepcopy

info = {
    'params': {
        'ombh2': {'prior': {'min': 0.005, 'max': 0.1},
                  'ref': {'dist': 'norm', 'loc': 0.0224, 'scale': 0.0001},
                  'proposal': 0.0001,
                  'latex': '\Omega_\mathrm{b} h^2'},
        'omch2': {'prior': {'min': 0.001, 'max': 0.99},
                  'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                  'proposal': 0.0005,
                  'latex': '\Omega_\mathrm{c} h^2'},
        'H0': {'prior': {'min': 40., 'max': 100.},
               'ref': {'dist': 'norm', 'loc': 67., 'scale': 1.},
               'proposal': 0.5,
               'latex': 'H_0'},
        'logA': {'prior': {'min': 1.6, 'max': 7.},
                 'ref': {'dist': 'norm', 'loc': 3.05, 'scale': 0.001},
                 'proposal': 0.001,
                 'drop': True,
                 'latex': '\log(10^{10} A_\mathrm{s})'},
        'ns': {'prior': {'min': 0.6, 'max': 1.2},
               'ref': {'dist': 'norm', 'loc': 0.96, 'scale': 0.004},
               'proposal': 0.002,
               'latex': 'n_\mathrm{s}'},
        'As': {'value': 'lambda logA: 1e-10*np.exp(logA)',
               'latex': 'A_\mathrm{s}'},
        'w': -1.,
        'wa': 0.,
        'tau': 0.0925,
        'mnu': 0.06,
        'nnu': 3.046,
        'omk': 0.0,
        'omegam': {'latex': '\Omega_\mathrm{m}'},
        'omegab': {'latex': '\Omega_\mathrm{b}'},
        'sigma8': {'latex': '\sigma_8'},
        'like_selection': 2,
        'full_photo': False,
        'NL_flag': False,
        'b1_photo': 1.0997727037892875,
        'b2_photo': 1.220245876862528,
        'b3_photo': 1.2723993083933989,
        'b4_photo': 1.316624471897739,
        'b5_photo': 1.35812370570578,
        'b6_photo': 1.3998214171814918,
        'b7_photo': 1.4446452851824907,
        'b8_photo': 1.4964959071110084,
        'b9_photo': 1.5652475842498528,
        'b10_photo': 1.7429859437184225,
        'b1_spec': 1.46,
        'b2_spec': 1.61,
        'b3_spec': 1.75,
        'b4_spec': 1.90,
        'aia': 1.72,
        'nia': -0.41,
        'bia': 0.0},
    'theory': {'camb':
               {'stop_at_error': True,
                'extra_args':{'num_massive_neutrinos': 1,
                              'dark_energy_model': 'ppf',
                              'share_delta_neff': True}}},
    'sampler': {'mcmc': {'max_tries': 100000}},
    'likelihood': {'Euclid': EuclidLikelihood},
    'force': True,
    'output': 'chains/spectroscopic'
    }


updated_info, sampler = run(info)

