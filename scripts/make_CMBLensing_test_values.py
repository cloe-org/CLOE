#!/usr/bin/env python

"""
This script computes the test values used in test_cmbx.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os, sys

likelihood_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0, likelihood_path)

import cloe
from cloe.cmbx_p.cmbx import CMBX
from cloe.cobaya_interface import EuclidLikelihood
from cloe.photometric_survey.photo import Photo
from astropy import constants as const
from pathlib import Path
from scipy import interpolate
import numpy as np
from cobaya.model import get_model

info = {
    #'params': Cobaya's protected key of the input dictionary. 
    # Includes the parameters that the user would like to sample over:
'params': {
        # (UC): each parameter below (which is a 'key' of another sub-dictionary) can contain a dictionary
        # with the key 'prior', 'latex'...
        # If the prior dictionary is not passed to a parameter, this parameter is fixed.
        # In this example, we are sampling the parameter ns
        # For more information see: https://cobaya.readthedocs.io/en/latest/example.html
        'ombh2': 0.022445, #Omega density of baryons times the reduced Hubble parameter squared
        'omch2': 0.1205579307, #Omega density of cold dark matter times the reduced Hubble parameter squared
        'H0': 67, #Hubble parameter evaluated today (z=0) in km/s/Mpc
        'tau': 0.0925, #optical depth
        'mnu': 0.06, #  sum of the mass of neutrinos in eV
        'nnu': 3.046, #N_eff of relativistic species 
        'As': 2.12605e-9, #Amplitude of the primordial scalar power spectrum
        'ns': 0.96, # primordial power spectrum tilt (sampled with an uniform prior)
        'w': -1, #Dark energy fluid model
        'wa': 0, #Dark energy fluid model
        'omk': 0.0, #curvature density
        'omegam': None, #DERIVED parameter: Omega matter density
        'omegab': None, #DERIVED parameter: Omega baryon density
        'omeganu': None, #DERIVED parameter: Omega neutrino density
        'omnuh2': 0.0, #DERIVED parameter: Omega neutrino density times de reduced Hubble parameter squared
        'omegac': None, #DERIVED parameter: Omega cold dark matter density
        'N_eff': None,
        # (UC): galaxy bias parameters:
        # The bias parameters below are currently fixed to the
        # values used by the Inter Science Taskforce: Forecast (IST:F)
        # and presented in the corresponding IST:F paper (arXiv: 1910.09273).
        # However, they can be changed by the user and even sample over them by putting a prior
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
        # Shear calibration multiplicative bias parameters                                                                                                                                                                                                                                                                                                            
        'multiplicative_bias_1': 0.,
        'multiplicative_bias_2': 0.,
        'multiplicative_bias_3': 0.,
        'multiplicative_bias_4': 0.,
        'multiplicative_bias_5': 0.,
        'multiplicative_bias_6': 0.,
        'multiplicative_bias_7': 0.,
        'multiplicative_bias_8': 0.,
        'multiplicative_bias_9': 0.,
        'multiplicative_bias_10': 0.,
        # Spectroscopic bias parameters
        'b1_spectro': 1.46,
        'b2_spectro': 1.61,
        'b3_spectro': 1.75,
        'b4_spectro': 1.90,
        # Intrinsic alignment parameters
        'aia': 1.72,
        'nia': -0.41,
        'bia': 0.0,
        # Redshift distributions nuisance parameters: shifts
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
    #'theory': Cobaya's protected key of the input dictionary.
    # Cobaya needs to ask some minimum theoretical requirements to a Boltzman Solver
    # (UC): you can choose between CAMB or CLASS
    # In this DEMO, we use CAMB and specify some CAMB arguments
    # such as the number of massive neutrinos
    # and the dark energy model
    #
    # ATTENTION: If you have CAMB/CLASS already installed and 
    # you are not using the likelihood conda environment 
    # or option (2) in cell (3) (Cobaya modules), you can add an extra key called 'path' within the camb dictionary
    # to point to your already installed CAMB code
    # NOTE: for values of the non-linear flag larger than 0, a new key is added in info['theory']['camb']['extra_args'],
    # i.e. 'halofit_version', which contains the requested version of halofit, as described above
    'theory': {'camb': 
               {'stop_at_error': True, 
                'extra_args':{'num_massive_neutrinos': 1,
                              'dark_energy_model': 'ppf', 
                            'lens_potential_accuracy':1,
                             }}},
    #'sampler': Cobaya's protected key of the input dictionary.
    # (UC): you can choose the sampler you want to use.
    # Check Cobaya's documentation to see the list of available samplers
    # In this DEMO, we use the 'evaluate' sampler to make a single computation of the posterior distributions
    # Note: if you want to run a simple MCMC sampling choose 'mcmc'
    'sampler': {'evaluate': None},  
    # 'packages_path': Cobaya's protected key of the input dictionary.
    # This is the variable you need to update
    # if you are running Cobaya with cobaya_modules (option (2) above).
    # If you are using the conda likelihood environment or option (1),
    # please, keep the line below commented
    #
    #'packages_path': modules_path,
    #
    #'output': Cobaya's protected key of the input dictionary.
    # Where are the results going to be stored, in case that the sampler produce output files? 
    # For example: chains...
    # (UC): modify the path below within 'output' to choose a name and a directory for those files
    'output': 'chains/my_euclid_experiment',
    #'debug': Cobaya's protected key of the input dictionary.
    # (UC): how much information you want Cobaya to print? If debug: True, it prints every single detail
    # that is going on internally in Cobaya
    'debug': False,
    #'timing': Cobaya's protected key of the input dictionary.
    # (UC): if timing: True, Cobaya returns how much time it took it to make a computation of the posterior
    # and how much time take each of the modules to perform their tasks
    'timing': True,
    #'force': Cobaya's protected key of the input dictionary.
    # (UC): if 'force': True, Cobaya forces deleting the previous output files, if found, with the same name
    'force': True,
    }


info['likelihood'] = {'Euclid': 
                     {'external': EuclidLikelihood, # Likelihood Class to be read as external
                     # Note: everything down below will overwrite the information read
                     # in the config folder
                     #
                     # Select which observables want to use during the analysis
                    'observables_selection': {
                         'WL': {'WL': True, 'GCphot': True, 'GCspectro': False},
                         'GCphot': {'GCphot': True, 'GCspectro': False},
                         'GCspectro': {'GCspectro': True}, 
                         'CMBlens': {'CMBlens': True, 'WL': True, 'GCphot': True, 'GCspectro':False},
                         'ISW': {'GCphot': False},
                         'CG': {'CG': False},
                         'add_phot_RSD' : False,
                         'matrix_transform_phot' : False
                     },
                     # Plot the selected observables matrx
                     'plot_observables_selection': True,  
                      # Non-linear flag
                      # With this, the user can specify which non-linear model they want
                      # For the time-being the available options are: 
                            #0 -> linear-only
                            #1 -> Takahashi
                            #2 -> Mead2020 (w/o baryon corrections)
                     'NL_flag_phot_matter': 0,
                     #
                     #'data': This give specifications for the paths of the input data files
                     'data': { 
                        #'sample' specifies the first folder below the main data folder
                        'sample': 'ExternalBenchmark',
                        #'spectro' and 'photo' specify paths to data files.
                        'spectro': {
                            # GC Spectro root name should contain z{:s} string
                            # to enable iteration over bins
                            'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
                            'redshifts': ["1.", "1.2", "1.4", "1.65"],
                            'edges': [0.9, 1.1, 1.3, 1.5, 1.8],
                            'root_mixing_matrix': 'mm_FS230degCircle_m3_nosm_obsz_z0.9-1.1.fits',
                            'Fourier': True},
                        'photo': {
                            'redshifts': [0.2095, 0.489, 0.619, 0.7335, 0.8445, 0.9595, 1.087, 1.2395, 1.45, 2.038],
                            'ndens_GC': 'niTab-EP10-RB00.dat',
                            'ndens_WL': 'niTab-EP10-RB00.dat',
                            # Photometric root names should contain z{:s} string
                            # to specify IA model
                            'root_GC': 'Cls_{:s}_PosPos_new.dat',
                            'root_WL': 'Cls_{:s}_ShearShear_new.dat',
                            'root_XC': 'Cls_{:s}_PosShear_new.dat',
                            'IA_model': 'zNLA',
                            # Photometric covariances root names should contain z{:s} string
                            # to specify how the covariance was calculated
                            'luminosity_ratio': 'luminosity_ratio.dat',
                            'photo_data': 'standard',
                            'Fourier': True,
                            'root_mixing_matrix': 'fs2_mms_10zbins_32ellbins.fits',
                            'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npy',
                            'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npy',
                            'cov_3x2': 'CovMat-3x2pt-{:s}-20Bins.npy',
                            'cov_model': 'Gauss'}, 
                          'cmbx': {
                            'root_CMBlens': 'Cls_kCMB.dat',
                            'root_CMBlensxWL': 'Cls_kCMBxWL.dat',
                            'root_CMBlensxGC': 'Cls_kCMBxGC.dat',                              
                            'ISW_model': 'zNLA',
                            'root_CMBisw': 'Cls_{:s}_ISWxGC.dat',
                            'cov_7x2pt': 'Cov_7x2pt_WL_GC_CMBX.npy',}
                     }
                    }}


model = get_model(info)
model.logposterior([])

z_max = 1080
z_samp_log = 20
zs_base = np.unique(np.append(np.linspace(0.0, 4.0, 100), np.logspace(np.log10(4.0),np.log10(z_max),z_samp_log)))
ks_base = np.logspace(-3.0, 1.0, 100)

cmov = model.provider.get_comoving_radial_distance(zs_base)
cur_dir = Path(__file__).resolve().parents[0]
np.savetxt(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/ComDist-LCDM-Lin-zNLA.dat', np.column_stack((zs_base,cmov)))

Hz = model.provider.get_Hubble(zs_base)
np.savetxt(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/Hz.dat', np.column_stack((zs_base,Hz)))

f_sig8 = model.provider.get_fsigma8(zs_base)
np.save(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/f_sig_8_arr.npy', f_sig8)

sig8 = model.provider.get_sigma8_z(zs_base)
np.save(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/sig_8_arr.npy', sig8)


like = model.likelihood['Euclid']
pdd = like.cosmo.cosmo_dic['Pmm_phot'](zs_base, ks_base)
np.save(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/pdd.npy',pdd)

pgd = like.cosmo.cosmo_dic['Pgdelta_phot'](zs_base, ks_base)
np.save(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/pgd.npy',pgd)

pdi = like.cosmo.cosmo_dic['Pdeltai'](zs_base, ks_base)
np.save(str(cur_dir) + '/../likelihood/tests/test_input_cmbx/pdi.npy',pdi)


photo = Photo(like.cosmo.cosmo_dic, like.likefinal.data_ins.nz_dict_WL, like.likefinal.data_ins.nz_dict_GC_Phot)

cmblens = CMBX(photo)
cmblens.cmbx_update(photo)


win_kCMBcheck = cmblens.kCMB_window(2.)        
print(f'win_kCMBcheck = {win_kCMBcheck}')

print(f'cl_kCMBcheck = {cmblens.Cl_kCMB(10.)}')

print(f'cl_kCMB_X_GCcheck = {cmblens.Cl_kCMB_X_GC_phot(10., 1)}')

print(f'cl_kCMB_X_WLcheck = {cmblens.Cl_kCMB_X_WL(10., 1)}')
