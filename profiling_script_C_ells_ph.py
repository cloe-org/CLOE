#General imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate 
import sys



plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('font',size=15)
#plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=17)
plt.rc('lines', linewidth=2)
plt.rc('lines', markersize=6)
plt.rc('legend', fontsize=20)
plt.rc('mathtext', fontset='stix')
plt.rc('font', family='STIXGeneral')
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


#Import Cosmology module from the Likelihood Package to play with cosmological quantities
from likelihood.cosmo.cosmology import Cosmology
# Some of the theory needs require extra info (redshift, ks)...
z_min = 0.0
z_max = 4.0
z_samp = 100
z_win = np.linspace(z_min, z_max, z_samp)
k_max = 0.2



# Cobaya_interface save the cosmology parameters and the cosmology requirements 
# from CAMB/CLASS via COBAYA to the cosmology class

# This dictionary collects info from cobaya
theory_dic = {'H0': model.provider.get_param('H0'),
                  'omch2': model.provider.get_param('omch2'),
                  'ombh2': model.provider.get_param('ombh2'),
                  'mnu': model.provider.get_param('mnu'),
                  'omnuh2': model.provider.get_param('mnu') / 94.07 * (1./3)**0.75, 
                  'comov_dist': model.provider.get_comoving_radial_distance(z_win),
                  'angular_dist': model.provider.get_angular_diameter_distance(z_win),
                  'H': model.provider.get_Hubble(z_win),
                  'Pk_interpolator': model.provider.get_Pk_interpolator(nonlinear=False),
                  'Pk_delta': None,
                  'fsigma8': None,
                  'z_win': z_win
                  }
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


C_GC_bench = np.genfromtxt("./data/ExternalBenchmark/Photometric/CijGG-LCDM-Lin-noIA.dat")
C_LL_bench = np.genfromtxt("./data/ExternalBenchmark/Photometric/CijLL-LCDM-Lin-noIA.dat")

from likelihood.photometric_survey.shear import Shear
shear = Shear(cosmology.cosmo_dic)


len_ell_max = int(sys.argv[1])
C_GC_bench = C_GC_bench[:len_ell_max]
C_GC_bench_ells = C_GC_bench[:,0]
ell_min = C_GC_bench_ells[0]
ell_max = C_GC_bench_ells[-1]

print("Computing gal-gal C_ells")
print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max) 
 

# Save C_GC_11
C_GC_11 = np.zeros(len_ell_max)

for y, l in enumerate(C_GC_bench_ells):
    print(l)
    C_GC_11[y]=shear.Cl_GC_phot(l, 1, 1, int_step=0.1)

ISTL_C_GC_11 = np.column_stack((C_GC_bench_ells, C_GC_11))
np.savetxt('./data/ISTL_C_GC_11.txt', ISTL_C_GC_11)


len_ell_max = int(sys.argv[1])
C_LL_bench = C_LL_bench[:len_ell_max]
C_LL_bench_ells = C_LL_bench[:, 0]
ell_min = C_LL_bench_ells[0]
ell_max = C_LL_bench_ells[-1]

print("Computing shear-shear C_ells")
print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max) 


C_LL_11 = np.zeros(len_ell_max)
for y, l in enumerate(C_LL_bench_ells):
    print(l)
    C_LL_11[y]=shear.Cl_WL(l, 1, 1, int_step=0.1)


ISTL_C_LL_11 = np.column_stack((C_LL_bench_ells, C_LL_11))
np.savetxt('./data/ISTL_C_LL_11.txt', ISTL_C_LL_11)

plt.figure(figsize=(10,6));
plt.loglog(C_GC_bench[:,0], C_GC_bench[:, 1], 'g--', label=r"Benchmark $(i,j=1,1)$")
plt.loglog(C_GC_bench_ells, C_GC_11, 'o', label="GCph Shear $(i,j=1,1)$")
plt.xlabel(r"$\ell$", fontsize=20);
plt.ylabel(r"$C_\ell^{GCphot}$", fontsize=20);
plt.legend()
plt.savefig("./data/plot_comparison_C_GC.png")
plt.close()

plt.figure(figsize=(10,6))
plt.plot(C_GC_bench[:, 0],  100*(C_GC_11-C_GC_bench[:, 1])/C_GC_bench[:, 1], 'k-')
plt.xlabel(r"$\ell$", fontsize=20);
plt.ylabel(r"$\Delta C_\ell^{GCphoto}/C_\ell^{GCphoto} (\%)$", fontsize=20);
plt.savefig("./data/plot_delta_C_GC.png")
plt.close()


plt.figure(figsize=(10,6));
plt.loglog(C_LL_bench[:,0], C_LL_bench[:, 1], 'g--', label=r"Benchmark $(i,j=1,1)$")
plt.loglog(C_LL_bench_ells, C_LL_11, 'o', label="WL Shear $(i,j=1,1)$")
plt.xlabel(r"$\ell$", fontsize=20);
plt.ylabel(r"$C_\ell^{WL}$", fontsize=20);
plt.legend()
plt.savefig("./data/plot_comparison_C_WL.png")
plt.close()

plt.figure(figsize=(10,6));
plt.plot(C_LL_bench[:, 0],  100*(C_LL_11-C_LL_bench[:, 1])/C_LL_bench[:, 1], 'k-', label=r"Benchmark $(i,j=1,1)$")
plt.xlabel(r"$\ell$", fontsize=20);
plt.ylabel(r"$\Delta C_\ell^{WL}/C_\ell^{WL} (\%)$", fontsize=20);
plt.savefig("./data/plot_delta_C_LL.png")
plt.close()





