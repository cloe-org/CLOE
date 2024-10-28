import sys
import numpy as np
from pathlib import Path
import os
import copy as copy

parent_path = str(Path(Path(__file__).resolve().parents[1]))
sys.path.insert(0,parent_path)
print(parent_path)

from cobaya.run import run
from cloe.cobaya_interface import EuclidLikelihood
from cloe.auxiliary.likelihood_yaml_handler import set_halofit_version


flag_data = 3 # HMcode2020 matter
data_barflag = 2 # HMcode2020 baryons

covmodel = 'GaussSSC'  # or: Gauss
flag_cov = 2 # Warning: calling it here with the old flag number 2 = HMcode2020

flag_model = 3 # HMcode2020 matter
model_barflag = 4 # Bacco baryons

flag_data_str = 'flag_phot_matter_{:d}'.format(flag_data)
flag_model_str = 'flag_{:d}'.format(flag_model)
flag_cov_str = 'flag_{:d}'.format(flag_cov)

if data_barflag!=0:
    bar_str = '_Bar_flag_{:d}'.format(data_barflag)
    flag_data_str = flag_data_str+bar_str
if model_barflag!=0:
    bar_str = '_Bar_flag_{:d}'.format(model_barflag)
    flag_model_str = flag_model_str+bar_str

chains_name = 'WL_ellmax4000_3cosmo_2nuis'+'-model'+flag_model_str+'-data'+flag_data_str+'-cov_'+covmodel+'_'+flag_cov_str

print("Name of current script:")
print(sys.argv[0])
print("Output name file for chains")
print(chains_name)
print("If these two don't match, please check your settings")
#sys.exit()

import runmcmc_base as defdic

definfo = copy.deepcopy(defdic.info)

info={
                 'output': 'chains/'+chains_name,
                 'debug': True,
                 'timing': True,
                 'force': True,
                 #'resume' : True
                 }

definfo.update(info)

info['params'] = {
        'ombh2': 0.0224,
        'omch2': {'prior': {'min': 0.001, 'max': 0.99},
                  'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                  'proposal': 0.0005,
                  'latex': '\Omega_\mathrm{c} h^2'},
        'logA': {'prior': {'min': 1.6, 'max': 7.},
                 'ref': {'dist': 'norm', 'loc': 3.05, 'scale': 0.001},
                 'proposal': 0.001,
                 'drop': True,
                 'latex': '\log(10^{10} A_\mathrm{s})'},
        'As': {'value': 'lambda logA: 1e-10*np.exp(logA)',
               'latex': 'A_\mathrm{s}'},
        'H0': {'prior': {'min': 40., 'max': 100.},
               'ref': {'dist': 'norm', 'loc': 67., 'scale': 1.},
               'proposal': 0.5,
               'latex': 'H_0'},
        'ns': 0.96,
        'tau': 0.0925,
        'mnu': 0.06,
        'nnu': 3.046,
        'w': -1.0,
        'wa': 0.0,
        'omk': 0.0,
        'omegam': {'latex': '\Omega_\mathrm{m}'},
        'omegab': {'latex': '\Omega_\mathrm{b}'},
        'sigma8': {'latex': '\sigma_8'},
        'omeganu': None,
        'omnuh2': None,
        'omegac': None,
        'N_eff': None,
        # Photometric bias parameters
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
        # Spectroscopic bias parameters
        # HMcode baryon parameter
        # BCemu baryon parameter
        #'log10Mc_bcemu_bin1': 13.402342835406102,
        # BACCO baryon parameter
        'M_c_bacco_bin1': {'prior': {'min': 9., 'max': 15.},
               'ref': {'dist': 'norm', 'loc': 14.0, 'scale': 0.01},
                'proposal': 0.01,
                'latex': '\log_{10} M_{c_{1}}'},
        # Intrinsic alignment parameters
        'a1_ia': {'prior': {'min': -5, 'max': 5.},
                'ref': {'dist': 'norm', 'loc': 1.72, 'scale': 0.1},
                'proposal': 0.1,
                'latex': 'A1^\mathrm{IA}'},
        'eta1_ia': {'prior': {'min': -5., 'max': 5.},
                'ref': {'dist': 'norm', 'loc': -0.41, 'scale': 0.01},
                'proposal': 0.1,
                'latex': '\eta1^\mathrm{IA}'},
        'beta1_ia': 0.0,
                  }
definfo['params'].update(info['params'])

info['sampler'] =      {'mcmc': {'max_tries': 10000,
                         'learn_proposal_Rminus1_max': 20.,
                         'learn_proposal_Rminus1_max_early': 500.,
                         'learn_proposal_Rminus1_min': 0.}
                        }
definfo['sampler'].update(info['sampler'])

info['likelihood'] = {}

info['likelihood']['Euclid'] = {
                                   'external': EuclidLikelihood,
                                   'observables_selection': {
                                      'WL': {
                                        'WL': True,
                                        'GCphot': False,
                                        'GCspectro': False},
                                      'GCphot': {
                                        'GCphot': False,
                                        'GCspectro': False},
                                      'GCspectro': {
                                        'GCspectro': False},
                                     'add_phot_RSD': False,
                                     'matrix_transform_phot': False,
                                      },
                                   'plot_observables_selection': False,
                                   'NL_flag_phot_matter': flag_model,
                                  # Baryon flag
                                  # With this, the user can specify which baryon model they want
                                  # For the time-being the available options are:
                                        #0 -> No baryonic feedback
                                        #1 -> Bacco baryons
                                        #2 -> BCMemu baryons
                                        #3 -> HMcode baryons
                                  'NL_flag_phot_baryon': model_barflag,
                                  # This flag sets the redshift evolution for
                                  # baryonic parameters for emulators
                                  # The options are:
                                        # True -> use X(z) = X_0 * (1+z)^(-nu_X),
                                        # no. of params: 7*2 = 14
                                        # False -> use X_i at each redshift bin i
                                        # and interpolate, no. of params: 7*10 = 70
                                  'Baryon_redshift_model': False
                                  }

definfo['likelihood']['Euclid'].update(info['likelihood']['Euclid'])

info['likelihood']['Euclid']['data'] = {
                        'sample': 'ExternalBenchmark',
                        'photo': {
                            'luminosity_ratio': 'luminosity_ratio.dat',
                            'ndens_GC': 'niTab-EP10-RB00.dat',
                            'ndens_WL': 'niTab-EP10-RB00.dat',
                            'root_GC': 'Cls_{:s}_PosPos_NL_'+flag_data_str+'.dat',
                            'root_WL': 'Cls_{:s}_ShearShear_NL_'+flag_data_str+'.dat',
                            'root_XC': 'Cls_{:s}_PosShear_NL_'+flag_data_str+'.dat',
                            'IA_model': 'zNLA',
                            'cov_GC':     'CovMat-PosPos-{:s}-20bins-NL_'+flag_cov_str+'.npz',
                            'cov_WL': 'CovMat-ShearShear-{:s}-20bins-NL_'+flag_cov_str+'.npz',
                            'cov_3x2pt':     'CovMat-3x2pt-{:s}-20bins-NL_'+flag_cov_str+'.npz',
                            'cov_model': covmodel}}


definfo['likelihood']['Euclid']['data'].update(info['likelihood']['Euclid']['data'])

info = copy.deepcopy(definfo)

set_halofit_version(info, info['likelihood']['Euclid']['NL_flag_phot_matter'])

for key in info:
    print("**** Printing key ", key, " ****")
    print(info[key])

updated_info, sampler = run(info)
