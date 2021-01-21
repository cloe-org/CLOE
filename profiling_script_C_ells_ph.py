#General imports
import numpy as np
from scipy import integrate
from scipy import interpolate
import sys
from astropy import constants as const
import time

#Import cobaya -need to be installed
import cobaya
#Import external loglike from the Likelihood Package within cobaya interface module
from likelihood.cobaya_interface import EuclidLikelihood


print("****** running script: ", sys.argv[0])

#ATTENTION: CHANGE THIS TO YOUR LOCAL PATH where your external codes are installed: CAMB, polychord, likelihoods...

##modules_path = "/data2/cobaya_modules/"   ##Commented out, using local environment camb installation

h=0.67  ## SanCas: this h should enter directly the H0 below, otherwise can be misleading
# Info is the 'ini file' dictionary.
# You can translate this info-dictionary into a yaml file and run: cobaya-run info.yaml such as CosmoMC like

info = {
    #Which parameters would you like to sample?
    'params': {
        # Fixed
        'ombh2': 0.05*h**2, 'omch2': 0.268563*h**2, 'H0': 67, 'tau': 0.0925,
        'mnu': 0.06, 'nnu': 3.046,
        # Sampled - just as an example we assume we will be sampling over ns
        'ns': 0.96, #{'prior': {'min': 0.8, 'max': 1.2}},
        #To be passed to euclid which likelihood to use (preliminary)
        # 1: shear
        # 2: spec
        # 12: both
        'like_selection': 12,
        'As': 2.12605e-9},
    #Which theory code you want to use? HERE CAMB
    'theory': {'camb': {'stop_at_error': True, 'extra_args':{'num_massive_neutrinos': 1}}},
    #Which sample do you want to use? HERE I use MCMC for testing
    'sampler': {'mcmc': None},
    #Where have you installed your modules (i.e: CAMB, polychord, likelihoods...)
    ### 'packages_path': modules_path,  ##commented out, using local env camb installation
    #Where are the results going to be stored?
    'output': 'chains/my_euclid_experiment',
    #Likelihood: we load the likelihood as an external function
    'likelihood': {'Euclid': EuclidLikelihood},
    'debug': False,
    'force': True
    }


# Model wrapper of cobaya
from cobaya.model import get_model
model = get_model(info)
model.logposterior({})


# Import Cosmology module from the Likelihood Package to play with cosmological quantities
from likelihood.cosmo.cosmology import Cosmology
# Some of the theory needs require extra info (redshift, ks)...
z_min = 0.0
z_max = 4.0
z_samp = 100
z_win = np.linspace(z_min, z_max, z_samp)
# (SJ): log sampling
# z_win = np.logspace(-2, np.log10(4), 140)
# z_win[0] = 0
# z_win[1] = 1e-4
# z_win[2] = 1e-3
k_min_Boltzmannn = 0.002
k_max_Boltzmannn = 10.0
k_min_GC_phot_interp = 0.001
k_max_GC_phot_interp = 100.0
k_samp_GC = 100
k_win = np.logspace(np.log10(k_min_GC_phot_interp),
                    np.log10(k_max_GC_phot_interp),
                    k_samp_GC)


# Cobaya_interface save the cosmology parameters and the cosmology requirements
# from CAMB/CLASS via COBAYA to the cosmology class

# This dictionary collects info from cobaya
theory_dic = {'H0': model.provider.get_param('H0'),
              'H0_Mpc': model.provider.get_param('H0') / const.c.to('km/s').value,
              'omch2': model.provider.get_param('omch2'),
              'ombh2': model.provider.get_param('ombh2'),
              'Omc': model.provider.get_param('omch2') / (model.provider.get_param('H0') / 100.)**2.,
              'Omb': model.provider.get_param('ombh2') / (model.provider.get_param('H0') / 100.)**2.,
              'mnu': model.provider.get_param('mnu'),
              'omnuh2': model.provider.get_param('mnu') / 94.07 * (1./3)**0.75,
              'Omnu': (model.provider.get_param('mnu') / 94.07 * (1./3)**0.75) / (model.provider.get_param('H0') / 100.)**2.,
              'comov_dist': model.provider.get_comoving_radial_distance(z_win),
              'angular_dist': model.provider.get_angular_diameter_distance(z_win),
              'H': model.provider.get_Hubble(z_win),
              'H_Mpc': model.provider.get_Hubble(z_win, units='1/Mpc'),
              'Pk_interpolator': model.provider.get_Pk_interpolator(nonlinear=False),
              'Pk_delta': None,
              'fsigma8': None,
              'z_win': z_win,
              'k_win': k_win
              }

theory_dic['Omm'] = theory_dic['Omb'] + theory_dic['Omc'] + theory_dic['Omnu']
theory_dic['Pk_delta'] = model.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), nonlinear=False)
theory_dic['fsigma8'] = model.provider.get_fsigma8(z_win)
# Remember: h is hard-coded
R, z, sigma_R = model.provider.get_sigma_R()
#print(R)
theory_dic['sigma_8'] = sigma_R[:, 0]

# Initialize cosmology class from likelihood.cosmo.cosmology
# By default: LCDM
cosmology = Cosmology()
cosmology.cosmo_dic.update(theory_dic)
cosmology.update_cosmo_dic(z_win, 0.005)

from likelihood.photometric_survey.photo import Photo
from likelihood.data_reader.reader import Reader

test_reading = Reader()
test_reading.compute_nz()
nz_dic_WL = test_reading.nz_dict_WL
nz_dic_GC = test_reading.nz_dict_GC_Phot

photo = Photo(cosmology.cosmo_dic, nz_dic_WL, nz_dic_GC)

len_ell_max = 10
ell_min = 10
ell_max = 1000
C_ells_list = np.linspace(ell_min, ell_max, len_ell_max)

# These int_step values are for now chosen to achieve internal
# sub-percent precision for the Cls of the 1-1 tomographic bin
# combination within ell of 10 to 1000. The values can be modified
# further during the more rigorous benchmarking phase, where we
# will have decided on the precision required for Euclid.
int_step_GC = 0.05
int_step_WL = 0.05
int_step_cross = 0.02

print("Computing galaxy-galaxy C_ells")
print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
# Compute C_GC_11
a=time.time()
C_GC_11 = np.array([photo.Cl_GC_phot(ell, 1, 1, int_step=int_step_GC) for ell in C_ells_list])
b=time.time()
print(C_GC_11)
print("Time: ", b - a)

print("Computing shear-shear C_ells")
print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
# Compute C_LL_11
a=time.time()
C_LL_11 = np.array([photo.Cl_WL(ell, 1, 1, int_step=int_step_WL) for ell in C_ells_list])
b=time.time()
print(C_LL_11)
print("Time: ", b - a)

print("Computing shear-galaxy C_ells")
print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
# Compute C_cross_11
a=time.time()
C_cross_11 = np.array([photo.Cl_cross(ell, 1, 1, int_step=int_step_cross) for ell in C_ells_list])
b=time.time()
print(C_cross_11)
print("Time: ", b - a)

print("calculation finished")
