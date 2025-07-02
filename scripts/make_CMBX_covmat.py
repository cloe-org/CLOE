#!/usr/bin/env python

# This script generates the 7x2pt covariance matrix including
# - Euclid's GCphot and WL and  their cross-correlations
# - CMB lensing (with SO noise curve) and its cross-correlation with GCphot and WL
# - ISWxGCphot (with Planck's noise up to l= and then SO), assuming that its covariance with other probes is null


# To download the SO noise curves, run the command `git submodule update --init --recursive` from the main CLOE directory.

import numpy as np
import matplotlib.pyplot as plt
import time 
import os, sys

likelihood_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0, likelihood_path)

import cloe
from cloe.cobaya_interface import EuclidLikelihood
import numpy as np
from cobaya.model import get_model
from cloe.photometric_survey.photo import Photo
from cloe.cmbx_p.cmbx import CMBX
import matplotlib.pyplot as plt
from os.path import join as opj
import itertools


info = {
    #'params': Cobaya's protected key of the input dictionary. 
    # Includes the parameters that the user would like to sample over:
'params': {
        # (UC): each parameter below (which is a 'key' of another sub-dictionary) can contain a dictionary
        # with the key 'prior', 'latex'...
        # If the prior dictionary is not passed to a parameter, this parameter is fixed.
        # In this example, we are sampling the parameter ns
        # For more information see: https://cobaya.readthedocs.io/en/latest/example.html
        'ombh2': 0.022445, #Omega density of baryons times the reduced Hubble parameter squared
        'omch2': 0.1205579307, #Omega density of cold dark matter times the reduced Hubble parameter squared
        'H0': 67.0, #Hubble parameter evaluated today (z=0) in km/s/Mpc
        'tau': 0.0925, #optical depth
        'mnu': 0.06, #  sum of the mass of neutrinos in eV
        'nnu': 3.046, #N_eff of relativistic species 
        'As': 2.12605e-9, #Amplitude of the primordial scalar power spectrum
        'ns': 0.96, # primordial power spectrum tilt (sampled with an uniform prior)
        'w': -1.0, #Dark energy fluid model
        'wa': 0.0, #Dark energy fluid model
        'omk': 0.0, #curvature density
        'omegam': None, #DERIVED parameter: Omega matter density
        'omegab': None, #DERIVED parameter: Omega baryon density
        'omeganu': None, #DERIVED parameter: Omega neutrino density
        'omnuh2': None, #DERIVED parameter: Omega neutrino density times de reduced Hubble parameter squared
        'omegac': None, #DERIVED parameter: Omega cold dark matter density
        'N_eff': None},
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
                              'dark_energy_model': 'ppf'}}},
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
    # For example: chains...
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
#'likelihood': Cobaya's protected key of the input dictionary.
# (UC): The user can select which data wants to use for the analysis.
# Check Cobaya's documentation to see the list of the current available data experiments
# In this DEMO, we load the Euclid-Likelihood as an external function, and name it 'Euclid'
info['likelihood'] = {'Euclid': 
                     {'external': EuclidLikelihood, # Likelihood Class to be read as external
                     # Note: everything down below will overwrite the information read
                     # in the config folder
                     #
                     # Select which observables want to use during the analysis
                     # by setting them to True or False
                     'observables_selection': {
                         'WL': {'WL': True, 'GCphot': True, 'GCspectro': False},
                         'GCphot': {'GCphot': True, 'GCspectro': False},
                         'GCspectro': {'GCspectro': False},
                         'CG': {'CG': False},
                         # Add RSD to photometric probes
                         'add_phot_RSD': False,
                         # Switch to allow for matrix transformations of theory and data vectors
                         'matrix_transform_phot' : False # 'BNT' or 'BNT-test'
                     },
                     # Plot the selected observables matrx
                     'plot_observables_selection': True,  
                    # Switch to allow for matrix transformations of theory and data vectors
                    'matrix_transform_phot' : False, # 'BNT' or 'BNT-test,
                      # Nonlinear flags
                      # With this, the user can specify which nonlinear model they want
                      # For the time-being the available options are: 
                      # NL_flag_phot_matter
                        # 0 -> linear-only
                        # 1 -> Takahashi
                        # 2 -> Mead2016 (includes baryon corrections)
                        # 3 -> Mead2020 (w/o baryon corrections)
                        # 4 -> Mead2020_feedback (includes baryon corrections)
                        # 5 -> EE2
                        # 6 -> Bacco (matter)
                      # NL_flag_spectro
                        # 0 -> linear-only
                        # 1 -> EFT
                     'NL_flag_phot_matter': 0,
                     'NL_flag_spectro': 0,
                      # Baryon flag
                      # With this, the user can specify which baryon model they want
                      # For the time-being the available options are: 
                            #0 -> No baryonic feedback
                            #1 -> Mead2016 (baryons)
                            #2 -> Mead2020_feedback
                            #3 -> BCemu baryons
                            #4 -> Bacco baryons
                     'NL_flag_phot_baryon': 0,
                     # This flag sets the redshift evolution for baryonic parameters for emulators
                     # The options are:
                            # True -> use X(z) = X_0 * (1+z)^(-nu_X), no. of params: 7*2 = 14
                            # False -> use X_i at each redshift bin i and interpolate, no. of params: 7*10 = 70
                     'Baryon_redshift_model': False,
                     'solver': 'camb',
                     'params': {
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
                                # Magnification bias parameters
                                'magnification_bias_1': 0.0,
                                'magnification_bias_2': 0.0,
                                'magnification_bias_3': 0.0,
                                'magnification_bias_4': 0.0,
                                'magnification_bias_5': 0.0,
                                'magnification_bias_6': 0.0,
                                'magnification_bias_7': 0.0,
                                'magnification_bias_8': 0.0,
                                'magnification_bias_9': 0.0,
                                'magnification_bias_10': 0.0,
                                # Shear calibration multiplicative bias parameters                                 
                                'multiplicative_bias_1': 0.0,
                                'multiplicative_bias_2': 0.0,
                                'multiplicative_bias_3': 0.0,
                                'multiplicative_bias_4': 0.0,
                                'multiplicative_bias_5': 0.0,
                                'multiplicative_bias_6': 0.0,
                                'multiplicative_bias_7': 0.0,
                                'multiplicative_bias_8': 0.0,
                                'multiplicative_bias_9': 0.0,
                                'multiplicative_bias_10': 0.0,
                                # Spectroscopic bias parameters
                                'b1_spectro_bin1': 1.46,
                                'b1_spectro_bin2': 1.61,
                                'b1_spectro_bin3': 1.75,
                                'b1_spectro_bin4': 1.90,
                                # Intrinsic alignment parameters
                                'aia': 1.72,
                                'nia': -0.41,
                                'bia': 0.0,
                                # Redshift distributions nuisance parameters: shifts
                                'dz_1_GCphot': 0.0, 'dz_1_WL': 0.0,
                                'dz_2_GCphot': 0.0, 'dz_2_WL': 0.0,
                                'dz_3_GCphot': 0.0, 'dz_3_WL': 0.0,
                                'dz_4_GCphot': 0.0, 'dz_4_WL': 0.0,
                                'dz_5_GCphot': 0.0, 'dz_5_WL': 0.0,
                                'dz_6_GCphot': 0.0, 'dz_6_WL': 0.0,
                                'dz_7_GCphot': 0.0, 'dz_7_WL': 0.0,
                                'dz_8_GCphot': 0.0, 'dz_8_WL': 0.0,
                                'dz_9_GCphot': 0.0, 'dz_9_WL': 0.0,
                                'dz_10_GCphot': 0.0, 'dz_10_WL': 0.0,
                                'gamma_MG': 0.55,
                                'sigma_z': 0.002}, 
                     # k values for extrapolation of the matter power spectrum and size k-array
                     'k_max_extrap': 500.0,
                     'k_min_extrap': 1E-5,   
                     'k_samp': 1000,
                     # z limit values and size z-array
                     'z_min': 0.0,
                     'z_max': 4.0,
                     'z_samp': 100,
                     # Add RSD to photometric probes
                     'add_phot_RSD': False,
                     # Use MG gamma
                     'use_gamma_MG': False,
                     # Use redshift-dependent purity for GCspectro or not
                     'f_out_z_dep': False,
                     # Print theory predictions
                     'print_theory' : False,
                     # Add spectroscopic redshift errors
                     'GCsp_z_err' : True,
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
                            'Fourier': True,
                            'root_mixing_matrix': 'mm_FS230degCircle_m3_nosm_obsz_z0.9-1.1.fits',
                            'scale_cuts_fourier': 'GCspectro-Fourier.yaml'},
                        'photo': {
                            'redshifts': [0.2095, 0.489, 0.619, 0.7335, 0.8445, 0.9595, 1.087, 1.2395, 1.45, 2.038],
                            'ndens_GC': 'niTab-EP10-RB00.dat',
                            'ndens_WL': 'niTab-EP10-RB00.dat',
                            'luminosity_ratio': 'luminosity_ratio.dat',
                            # Photometric root names should contain z{:s} string
                            # to specify IA model
                            'root_GC': 'Cls_{:s}_PosPos.dat',
                            'root_WL': 'Cls_{:s}_ShearShear.dat',
                            'root_XC': 'Cls_{:s}_PosShear.dat',
                            'IA_model': 'zNLA',
                            'Fourier': True,
                            'root_mixing_matrix': 'fs2_mms_10zbins_32ellbins.fits',
                            'photo_data': 'standard',
                            # Photometric covariances root names should contain z{:s} string
                            # to specify how the covariance was calculated
                            'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npz',
                            'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npz',
                            'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npz',
                            'cov_model': 'Gauss'  # or 'BNT-Gauss' if BNT selected above
                            },
                         'cmbx': {
                                  'root_CMBlens': 'Cls_kCMB.dat',
                                  'root_CMBlensxWL': 'Cls_kCMBxWL.dat',
                                  'root_CMBlensxGC': 'Cls_kCMBxGC.dat',
                                  # 'root_CMBisw': 'Cls_{:s}_ISWxGC.dat',
                                  'root_CMBisw': 'Cls_zNLA_ISWxGC.dat',
                                  'ISW_model': 'zNLA',
                                  'cov_7x2pt': 'Cov_7x2pt_WL_GC_CMBX.npy'}}
                    }}

model = get_model(info)


# Evaluate the likelihood on the fiducial cosmology
logposterior = model.logposterior({})
like = model.likelihood['Euclid']
phot_ins = Photo(like.cosmo.cosmo_dic, like.likefinal.data_ins.nz_dict_WL, like.likefinal.data_ins.nz_dict_GC_Phot)
phot_ins.update(like.cosmo.cosmo_dic)

cmbx_ins = CMBX(phot_ins)
cmbx_ins.cmbx_update(phot_ins)
# To download the SO noise curves, run the command `git submodule update --init --recursive` from the main CLOE directory.
savepath = opj(os.path.dirname(os.path.dirname(cloe.__file__)), 'data', 'ExternalBenchmark')


# We assume that all probes are defined with the same ell binning except for ISW

ells = like.likefinal.data_ins.data_dict['WL']['ells']
numtomo_wl = like.likefinal.data_ins.numtomo_wl
numtomo_gcphot = like.likefinal.data_ins.numtomo_gcphot
cl_dict = {}
nl_dict = {}
cl_keys = {}

ells_ISW = np.loadtxt("../data/ExternalBenchmark/cmbx/Cls_zNLA_ISWxGC.dat")[:,0]

# Get CMB lensing N0 bias
# We load the noise curves generated by SO collaboration 
# We take the Minimum Variance baseline iterative analysis 
# (recommanded science case in their README)
SO_lensing = opj(savepath, 'so_noise_models', 'LAT_lensing_noise/lensing_v3_1_1/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat')
N0s_SO = np.loadtxt(SO_lensing).T
N0 = N0s_SO[7]
# SO noise curve start at ell = 2, so we append ell = 0 and ell = 1 to get more easily N0(ell)
N0 = np.insert(N0, 0, 0)
N0 = np.insert(N0, 0, 0)

# TT noise and power spectrum
# We use the noise from Planck for l=2-39, and from SO beyond l=40
sigma_T = [33.]
beam = 7.
theta2 = (beam*np.array([np.pi/60./180.]))**2/8.0/np.log(2.0)
sigma_T *= np.array([np.pi/60./180.])
nl_TT = sigma_T**2*np.exp(ells_ISW*(ells_ISW+1)*theta2)
cl_TT = np.array([like.cosmo.cosmo_dic['Cl']['tt'][int(ell)] for ell in ells_ISW])

# We load here the temperature noise curve from SO
SO_temperature = opj(savepath, 'so_noise_models', 'LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_CMB.txt')
NT_SO = np.loadtxt(SO_temperature).T
ind_SO = np.where(ells_ISW>=40)[0]
nl_TT[ind_SO] = NT_SO[1,ells_ISW[ind_SO].astype(int)-40]


#TODO: I assume there is total overlap between the Simons Obs and Euclid sky
# This should be updated later
# Simons Obs sky fraction is of 40%, Euclid fsky is of 0.36%
fsky = 0.363609919

# Define here the galaxies number density in each bin (in arcmin-2 converted to steradians)
ngal = 3*3600*(180/np.pi)**2
# Define here the total intrinsic ellipticity dispersion
sig_eps = 0.3

# Rough computation of dell, might be more sophisticated in the future
dell = np.diff(ells)
dell = np.insert(dell, 0, dell[0])   

dell_ISW = np.diff(ells_ISW)
dell_ISW = np.insert(dell_ISW, 0, dell_ISW[0])   
cl_keys["WL-WL"] = []
for i, j in like.likefinal.indices_diagonal_wl:
    k = 'WL{}-WL{}'.format(i, j)
    cl_keys["WL-WL"].append(k)
    cl_dict[k] = phot_ins.Cl_WL(ells, i, j)
    nl_dict[k] = np.zeros(len(ells))
    if (i == j):
        nl_dict[k] = sig_eps**2/ngal*np.ones(len(ells)) / 2.

cl_keys["WL-GC"] = []
for i, j in like.likefinal.indices_all:
    k = 'WL{}-GC{}'.format(j, i) # Beware the order of j and i here
    cl_keys["WL-GC"].append(k)
    cl_dict[k] = phot_ins.Cl_cross(ells, j, i)
    nl_dict[k] = np.zeros(len(ells))

cl_keys["GC-GC"] = []
for i, j in like.likefinal.indices_diagonal_gcphot:
    k = 'GC{}-GC{}'.format(i, j)
    cl_keys["GC-GC"].append(k)
    cl_dict[k] = phot_ins.Cl_GC_phot(ells, i, j)
    nl_dict[k] = np.zeros(len(ells))
    if (i == j):
        nl_dict[k] = 1/ngal*np.ones(len(ells))
            
cl_keys["kCMB-kCMB"] = ["kCMB-kCMB"]
cmbx_ins.zmax = 1000 # needs to be done in a cleaner way
cl_dict['kCMB-kCMB'] = cmbx_ins.Cl_kCMB(ells)
nl_dict['kCMB-kCMB'] = np.zeros(len(ells))

for il, l in enumerate(ells):
    nl_dict['kCMB-kCMB'][il] = N0[int(l)]

cl_keys["kCMB-WL"] = []
for i in range(numtomo_wl):
    k = 'kCMB-WL{}'.format(i+1)
    cl_keys["kCMB-WL"].append(k)
    cl_dict[k] = cmbx_ins.Cl_kCMB_X_WL(ells, i+1)
    nl_dict[k] = np.zeros(len(ells))

cl_keys["kCMB-GC"] = []
for i in range(numtomo_gcphot):
    k = 'kCMB-GC{}'.format(i+1)
    cl_keys["kCMB-GC"].append(k)
    cl_dict[k] = cmbx_ins.Cl_kCMB_X_GC_phot(ells, i+1)
    nl_dict[k] = np.zeros(len(ells))
        
cl_ISWxGC_i = []
for i in range(numtomo_gcphot):
    cl_ISWxGC_i.append(cmbx_ins.Cl_ISWxGC(ells_ISW, i+1))
        
# We create new dicts cl_GC and nl_GC but with the ells_ISW
cl_GC_ij = {}
nl_GC_ij = {}
cl_keys_ISW = []
for i, j in like.likefinal.indices_diagonal_gcphot:
    k = 'GC{}-GC{}'.format(i, j)
    cl_keys_ISW.append(k)
    cl_GC_ij[k] = phot_ins.Cl_GC_phot(ells_ISW, i, j)
    nl_GC_ij[k] = np.zeros(len(ells_ISW))
    if (i == j):
        nl_GC_ij[k] = 1/ngal*np.ones(len(ells_ISW))

# Get symetric Cls for permutations
keys_6x2pt = list(cl_dict.keys())
for key in keys_6x2pt:
    probeA, probeB = key.split('-')
    cl_dict['-'.join([probeB, probeA])] = cl_dict[key]
    nl_dict['-'.join([probeB, probeA])] = nl_dict[key]

keys_ISW = list(cl_GC_ij.keys())
for key in keys_ISW:
    probeA, probeB = key.split('-')
    cl_GC_ij['-'.join([probeB, probeA])] = cl_GC_ij[key]
    nl_GC_ij['-'.join([probeB, probeA])] = nl_GC_ij[key]

print("Filling 6x2pt covariance matrix...")
len_6x2pt = len(keys_6x2pt) * len(ells)
len_7x2pt = len_6x2pt + len(ells_ISW)*numtomo_gcphot
cov_mat_tot = np.zeros((len_7x2pt,len_7x2pt))
# Here we compute the 6x2pt part
ind1 = 0
for key1 in cl_keys.keys():
    for il, l1 in enumerate(ells):
        for z1 in range(len(cl_keys[key1])):
            ind2 = 0
            for key2 in cl_keys.keys():
                for il2, l2 in enumerate(ells):
                    for z2 in range(len(cl_keys[key2])):
                        if l1 == l2:
                            probeA, probeB = cl_keys[key1][z1].split('-')
                            probeC, probeD = cl_keys[key2][z2].split('-')
                            cov_mat_tot[ind1, ind2] = 1. / (fsky * (2. * l + 1.) * dell[il]) \
                                * (
                                    (cl_dict['-'.join([probeA, probeC])][il] + nl_dict['-'.join([probeA, probeC])][il]) *
                                    (cl_dict['-'.join([probeB, probeD])][il] + nl_dict['-'.join([probeB, probeD])][il]) +
                                    (cl_dict['-'.join([probeA, probeD])][il] + nl_dict['-'.join([probeA, probeD])][il]) *
                                    (cl_dict['-'.join([probeB, probeC])][il] + nl_dict['-'.join([probeB, probeC])][il])
                            )
                        ind2 += 1
            ind1 += 1
             
# We add the ISW part  
# Note that we couldn't add the ISW at the same time as the other probes due to the ell binning scheme which is different
# We don't take into account the covariance between ISWxGC and other probes, but we do take into account the covariance 
# between ISWxGCi and ISWxGCj where i and j can be different (ie in different redshift bins)
print("Filling 7x2pt covariance matrix...")
for il, l in enumerate(ells_ISW):
    for i in range(numtomo_gcphot):
        for j in range(i,numtomo_gcphot):
            cov_mat_tot[len_6x2pt+il*numtomo_gcphot+i,len_6x2pt+il*numtomo_gcphot+j] = 1. / (fsky * (2. * l + 1.) * dell_ISW[il]) \
                * (
                    (cl_GC_ij['GC{}-GC{}'.format(i+1, j+1)][il] + nl_GC_ij['GC{}-GC{}'.format(i+1, j+1)][il]) *
                    (cl_TT[il] + nl_TT[il]) +
                    cl_ISWxGC_i[i][il] * cl_ISWxGC_i[j][il]
            )
        
        
#cov_mat_tot = cov_mat_tot + cov_mat_tot.T - np.diag(cov_mat_tot.diagonal())

np.save(opj(savepath, 'Cov_7x2pt_WL_GC_CMBX'), cov_mat_tot)



# Make a plot for visual check

def cov_to_corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

plt.figure(figsize=(10, 10))
plt.imshow(cov_to_corr(cov_mat_tot), cmap='RdBu', vmin=-0.1, vmax=0.1)
plt.show()
