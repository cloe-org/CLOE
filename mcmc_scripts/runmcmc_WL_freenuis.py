import sys
import os

script_path = os.path.realpath(os.getcwd())
if script_path.endswith('mcmc_scripts'):
    sys.path.append(os.path.realpath(os.path.join(script_path, os.pardir)))
else:
    sys.path.append(script_path)

from cobaya.run import run
from likelihood.cobaya_interface import EuclidLikelihood

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
        'b1_spectro': 1.4614804,
        'b2_spectro': 1.6060949,
        'b3_spectro': 1.7464790,
        'b4_spectro': 1.8988660,
        'aia': {'prior': {'min': -5, 'max': 5.},
                'ref': {'dist': 'norm', 'loc': 1.72, 'scale': 0.1},
                'proposal': 0.1,
                'latex': 'A^{\rm IA}'},
        'nia': {'prior': {'min': -5., 'max': 5.},
                'ref': {'dist': 'norm', 'loc': -0.41, 'scale': 0.01},
                'proposal': 0.1,
                'latex': '\eta^{\rm IA}'},
        'bia': 0.0,
        'dz_1_GCphot': 0., 'dz_1_WL': 0.,
        'dz_2_GCphot': 0., 'dz_2_WL': 0.,
        'dz_3_GCphot': 0., 'dz_3_WL': 0.,
        'dz_4_GCphot': 0., 'dz_4_WL': 0.,
        'dz_5_GCphot': 0., 'dz_5_WL': 0.,
        'dz_6_GCphot': 0., 'dz_6_WL': 0.,
        'dz_7_GCphot': 0., 'dz_7_WL': 0.,
        'dz_8_GCphot': 0., 'dz_8_WL': 0.,
        'dz_9_GCphot': 0., 'dz_9_WL': 0.,
        'dz_10_GCphot': 0., 'dz_10_WL': 0.},
    'theory': {
        'camb': {
            'stop_at_error': True,
            'extra_args': {
                'num_massive_neutrinos': 1,
                'dark_energy_model': 'ppf'}}},
    'sampler': {
        'mcmc': {
            'max_tries': 100000}},
    'output': 'chains/chain_WL_freenuis',
    'debug': False,
    'timing': True,
    'force': True}

info['likelihood'] = {
    'Euclid': {
        'external': EuclidLikelihood,
        'observables_selection': {
            'WL': {
                'WL': True,
                'GCphot': False,
                'GCspectro': False},
            'GCphot': {
                'GCphot': False, 'GCspectro': False},
            'GCspectro': {
                'GCspectro': False}},
        'plot_observables_selection': False,
        'NL_flag': 0,
        'data': {
            'sample': 'ExternalBenchmark',
                'spectro': {
                    'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
                    'redshifts': ["1.", "1.2", "1.4", "1.65"]},
                'photo': {
                    'ndens_GC': 'niTab-EP10-RB00.dat',
                    'ndens_WL': 'niTab-EP10-RB00.dat',
                    'root_GC': 'Cls_{:s}_PosPos.dat',
                    'root_WL': 'Cls_{:s}_ShearShear.dat',
                    'root_XC': 'Cls_{:s}_PosShear.dat',
                    'IA_model': 'zNLA',
                    'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npy',
                    'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npy',
                    'cov_3x2': 'CovMat-3x2pt-{:s}-20Bins.npy',
                    'cov_model': 'Gauss'}}}}

updated_info, sampler = run(info)
