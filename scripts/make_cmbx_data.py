#!/usr/bin/env python

# Get the CMB lensing spectra used as mock data 
# This uses function from the DEMO notebook 
# Plot the curves for visual checking 

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

nlflag=0
barflag=0

# Set this to zero for linear Cl_kk from CAMB
# But the Cl_kk will be inacurate for L>1000
nlflag_camb=0

info = {
    'params': {
        'ombh2': 0.022445,
        'omch2': 0.1205579307,
        'H0': 67.0,
        'tau': 0.0925,
        'mnu': 0.06,
        'nnu': 3.046,
        'As': 2.12605e-9,
        'ns': 0.96,
        'w': -1.0,
        'wa': 0.0,
        'omk': 0.0,
        'omegam': None,
        'omegab': None,
        'omeganu': None,
        'omnuh2': None,
        'omegac': None,
        'N_eff': None,
        # HMcode baryon parameter
        'HMCode_logT_AGN':7.8
        # OTHER BARYON PARAMETERS ARE SET TO DEFAULTS IN params.yaml.
        # IF THOSE ARE NOT DEFINED RE-RUN NL DEMO.
    },
    'theory': {'camb': 
               {'stop_at_error': True, 
                'extra_args':{'num_massive_neutrinos': 1,
                              'dark_energy_model': 'ppf', 
                              'lens_potential_accuracy': nlflag_camb 

                              }}},
    'sampler': {'evaluate': None},
    'output': 'chains/my_euclid_experiment_xcmb',
    'debug': False,
    'timing': True,
    'force': True,
    }

info['likelihood'] = {'Euclid': 
                     {'external': EuclidLikelihood, # Likelihood Class to be read as external
                     'observables_selection': {
                         'WL': {'WL': True, 'GCphot': True, 'GCspectro': False},
                         'GCphot': {'GCphot': True, 'GCspectro': False},
                         'GCspectro': {'GCspectro': False},
                         'CMBlens': {'CMBlens': True, 'WL': True, 'GCphot': True, 'GCspectro':False},
                         'CMBisw': {'GCphot': True},
                         'CG': {'CG': False},
                         'add_phot_RSD': False,
                         'matrix_transform_phot' : False,
                     },
                     # Plot the selected observables matrx
                     'plot_observables_selection': True,  
                    # Switch to allow for matrix transformations of theory and data vectors
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
                     'NL_flag_phot_matter': nlflag,
                     'NL_flag_spectro': 0,
                      # Baryon flag
                      # With this, the user can specify which baryon model they want
                      # For the time-being the available options are: 
                            #0 -> No baryonic feedback
                            #1 -> Mead2016 (baryons)
                            #2 -> Mead2020_feedback
                            #3 -> BCemu baryons
                            #4 -> Bacco baryons
                     'NL_flag_phot_baryon': barflag,
                     # This flag sets the redshift evolution for baryonic parameters for emulators
                     # The options are:
                            # True -> use X(z) = X_0 * (1+z)^(-nu_X), no. of params: 7*2 = 14
                            # False -> use X_i at each redshift bin i and interpolate, no. of params: 7*10 = 70
                     'Baryon_redshift_model': False,
                     'solver': 'camb',
                     'params':{
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
                        'b1_spectro_bin1': 1.4614804,
                        'b1_spectro_bin2': 1.6060949,
                        'b1_spectro_bin3': 1.7464790,
                        'b1_spectro_bin4': 1.8988660,
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
                        'dz_10_GCphot': 0., 'dz_10_WL': 0.,
                        'gamma_MG': 0.55,
                        'sigma_z': 0.002}, 
                     'data': {
                        'sample': 'ExternalBenchmark',
                        'spectro': {
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
                            'root_GC': 'Cls_{:s}_PosPos.dat',
                            'root_WL': 'Cls_{:s}_ShearShear.dat',
                            'root_XC': 'Cls_{:s}_PosShear.dat',
                            'IA_model': 'zNLA',
                            'Fourier': True,
                            'photo_data': 'standard',
                            'root_mixing_matrix': 'fs2_mms_10zbins_32ellbins.fits',
                            'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npz',
                            'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npz',
                            'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npz',
                            'cov_model': 'Gauss'},
                         'ExternalBenchmark/cmbx': {
                                  'root_CMBlens': 'Cls_kCMB.dat',
                                  'root_CMBlensxWL': 'Cls_kCMBxWL.dat',
                                  'root_CMBlensxGC': 'Cls_kCMBxGC.dat',
                                  'root_CMBisw': 'Cls_{:s}_ISWxGC.dat',
                                  'ISW_model': 'zNLA',
                                  'cov_7x2pt': 'Cov_7x2pt_WL_GC_CMBX.npy'}
                    }
                    }}


model = get_model(info)


# Evaluate the likelihood on the fiducial cosmology
logposterior = model.logposterior({})

like = model.likelihood['Euclid']


photo = Photo(like.cosmo.cosmo_dic, like.likefinal.data_ins.nz_dict_WL, like.likefinal.data_ins.nz_dict_GC_Phot)
photo.update(like.cosmo.cosmo_dic)

cmbx = CMBX(photo)
cmbx.cmbx_update(photo)
### Compute all the Cls

# Elles used in the data (from OU-LE3 ?)
ells = like.likefinal.data_ins.data_dict['WL']['ells']
iswxgc_ells = like.likefinal.data_ins.data_dict['ISWxGC']['ells']

cl_kk = np.zeros(len(ells))

for i, ell in enumerate(ells):
    cl_kk[i] = cmbx.Cl_kCMB(ell)

ntomowl = photo.nz_WL.get_num_tomographic_bins()
ntomogc = photo.nz_GC.get_num_tomographic_bins()

cl_kgi = np.zeros([ntomogc, len(ells)])
cl_kwi = np.zeros([ntomowl, len(ells)])

cl_iswxgc = np.zeros([ntomogc, len(iswxgc_ells)])

for ibin in range(ntomogc):
    cl_kgi[ibin] = cmbx.Cl_kCMB_X_GC_phot(ells, ibin+1)
        
for ibin in range(ntomowl):
    cl_kwi[ibin] = cmbx.Cl_kCMB_X_WL(ells, ibin+1)

for ibin in range(ntomogc):
    cl_iswxgc[ibin] = cmbx.Cl_ISWxGC(iswxgc_ells, ibin+1)

#### Save the data 


savepath = opj(os.path.dirname(os.path.dirname(cloe.__file__)), 'data/ExternalBenchmark', 'cmbx')
np.savetxt(opj(savepath, 'Cls_kCMB.dat'), np.array([ells, cl_kk]).T, header='ells     kCMB-kCMB')


header = 'ells'
for i in range(10):
    header += '    kCMB-WL{}'.format(i+1)
    
np.savetxt(opj(savepath, 'Cls_kCMBxWL.dat'), np.c_[ells, cl_kwi.T], header=header)



header = 'ells'
for i in range(10):
    header += '    kCMB-GC{}'.format(i+1)
    
np.savetxt(opj(savepath, 'Cls_kCMBxGC.dat'), np.c_[ells, cl_kgi.T], header=header)


header = 'ells'
for i in range(10):
    header += '    n{}'.format(i+1)
    
np.savetxt(opj(savepath, 'Cls_zNLA_ISWxGC.dat'), np.c_[iswxgc_ells, cl_iswxgc.T], header=header)
# np.savetxt(opj(savepath, 'Cls_cloe_ISWxGC.dat'), np.c_[iswxgc_ells, cl_iswxgc.T], header=header)

### Make some plots for visual checks

def w(ells):
    return (ells*(ells+1))**2 /4

plt.figure()

plt.loglog(ells, cl_kk, label='CLOE')

plt.loglog(ells.astype('int'), w(ells.astype('int'))*like.cosmo.cosmo_dic['Cl']['pp'][ells.astype('int')], label='CAMB')


plt.legend()
plt.xlabel('$L$')
plt.ylabel('$C_L^{\kappa \kappa}$')


plt.figure()

for ibin in range(ntomogc):
    plt.loglog(ells, cl_kgi[ibin], label='Bin {}'.format(ibin+1))
plt.legend(ncol=2)

plt.xlabel('$L$')
plt.ylabel('$C_L^{\kappa, \mathrm{GC photo}}$')


plt.figure()

for ibin in range(ntomowl):
    plt.loglog(ells, cl_kwi[ibin], label='Bin {}'.format(ibin+1))
plt.legend(ncol=2)

plt.xlabel('$L$')
plt.ylabel('$C_L^{\kappa, \mathrm{WL}}$')

plt.figure()

for ibin in range(ntomogc):
    plt.loglog(iswxgc_ells, cl_iswxgc[ibin], label='Bin {}'.format(ibin+1))
plt.legend(ncol=2)

plt.xlabel('$L$')
plt.ylabel('$C_L^{\mathrm{ISW}, \mathrm{GC}}$')

plt.show()
