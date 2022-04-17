import numpy as np
import os, sys
from pathlib import Path
from likelihood.cobaya_interface import EuclidLikelihood
from likelihood.auxiliary.likelihood_yaml_handler import set_halofit_version
from cobaya.model import get_model
from likelihood.photometric_survey.photo import Photo

parent_path = str(Path(Path(__file__).resolve().parents[1]))
data_path = parent_path + '/data/ExternalBenchmark/Photometric/data/'

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
        'b1_spectro': 1.4614804,
        'b2_spectro': 1.6060949,
        'b3_spectro': 1.7464790,
        'b4_spectro': 1.8988660,
        'aia': 1.72,
        'nia': -0.41,
        'bia': 0.0,
        'dz_1_GCphot': 0.0, 'dz_1_WL': 0.0,
        'dz_2_GCphot': 0.0, 'dz_2_WL': 0.0,
        'dz_3_GCphot': 0.0, 'dz_3_WL': 0.0,
        'dz_4_GCphot': 0.0, 'dz_4_WL': 0.0,
        'dz_5_GCphot': 0.0, 'dz_5_WL': 0.0,
        'dz_6_GCphot': 0.0, 'dz_6_WL': 0.0,
        'dz_7_GCphot': 0.0, 'dz_7_WL': 0.0,
        'dz_8_GCphot': 0.0, 'dz_8_WL': 0.0,
        'dz_9_GCphot': 0.0, 'dz_9_WL': 0.0,
        'dz_10_GCphot': 0.0, 'dz_10_WL': 0.0},
    'theory': {'camb':
               {'stop_at_error': True,
                'extra_args':{'num_massive_neutrinos': 1,
                              'dark_energy_model': 'ppf'}}},
    'sampler': {'evaluate': None},
    'output': 'chains/my_euclid_experiment',
    'debug': False,
    'timing': True,
    'force': True,
    }

info['likelihood'] = {'Euclid':
                        {'external': EuclidLikelihood,
                        'observables_selection': {
                            'WL': {'WL': True, 'GCphot': False, 'GCspectro': False},
                            'GCphot': {'GCphot': True, 'GCspectro': False},
                            'GCspectro': {'GCspectro': True}
                        },
                     'plot_observables_selection': True,
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
                            'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npy',
                            'cov_model': 'Gauss'}},
                        }
                    }

set_halofit_version(info, info['likelihood']['Euclid']['NL_flag'])

model = get_model(info)
logposterior = model.logposterior({})

like = EuclidLikelihood()
like.initialize()
like.passing_requirements(model, info, **model.provider.params)
like.cosmo.update_cosmo_dic(like.cosmo.cosmo_dic['z_win'], 0.05)

like.likefinal.data_ins.compute_nz()
photo = Photo(like.cosmo.cosmo_dic,
              like.likefinal.data_ins.nz_dict_WL,
              like.likefinal.data_ins.nz_dict_GC_Phot)


ell_bins = np.linspace(np.log(10.0),np.log(5000.0),21)
ells = (ell_bins[:-1]+ell_bins[1:])/2.0
ells = np.exp(ells)

nbin = 10

matGC = []
matWL = []
matXC = []

for bin_i in range(1, nbin + 1):
    for bin_j in range(1, nbin + 1):
        if (bin_j >= bin_i):
            cl_GC = np.array([photo.Cl_GC_phot(cur_ell, bin_i, bin_j) for
                              cur_ell in ells])
            cl_WL = np.array([photo.Cl_WL(cur_ell, bin_i, bin_j) for
                              cur_ell in ells])
            matGC.append(cl_GC)
            matWL.append(cl_WL)
        cl_cross = np.array([photo.Cl_cross(cur_ell, bin_j, bin_i) for
                             cur_ell in ells])
        matXC.append(cl_cross)

matGC = np.array(matGC).T
matWL = np.array(matWL).T
matXC = np.array(matXC).T
matGC = np.c_[ells, matGC]
matWL = np.c_[ells, matWL]
matXC = np.c_[ells, matXC]

header_GC = 'ells'
header_WL = 'ells'
header_XC = 'ells'

for i in range(1, nbin + 1):
    for j in range(1, nbin + 1):
        if (j >= i):
            header_GC = header_GC + '\tP%s-P%s'%(i,j)
            header_WL = header_WL + '\tE%s-E%s'%(i,j)
        header_XC = header_XC + '\tP%s-E%s'%(i,j)

np.savetxt(data_path + 'Cls_zNLA_PosPos_new.dat',
           matGC, fmt='%.12e', delimiter='\t',
           newline='\n', header=header_GC)
np.savetxt(data_path + 'Cls_zNLA_ShearShear_new.dat',
           matWL, fmt='%.12e', delimiter='\t',
           newline='\n', header=header_WL)
np.savetxt(data_path + 'Cls_zNLA_PosShear_new.dat',
           matXC, fmt='%.12e', delimiter='\t',
           newline='\n', header=header_XC)
