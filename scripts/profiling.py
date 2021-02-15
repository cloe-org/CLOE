#General imports
import numpy as np
from scipy import integrate
from scipy import interpolate
import sys
from astropy import constants as const
import time
from pathlib import Path
from likelihood.photometric_survey import photo
from likelihood.spectroscopic_survey.spec import Spec

#Import cobaya -need to be installed
import cobaya
#Import external loglike from the Likelihood Package within cobaya interface module
from likelihood.cobaya_interface import EuclidLikelihood


print("****** running script: ", sys.argv[0])

#ATTENTION: CHANGE THIS TO YOUR LOCAL PATH where your external codes are installed: CAMB, polychord, likelihoods...

if len(sys.argv) > 1:
    runoption = sys.argv[1]
else:
    runoption = 0
runoption = int(runoption)
print('runoption = ', runoption)

if runoption == 0:
    print('Full likelihood evaluation!')
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
            'H0': 67, #Hubble parameter evaluated today (z=0) in km/s/Mpc
            'tau': 0.0925, #optical depth
            'mnu': 0.06, #  sum of the mass of neutrinos in eV
            'nnu': 3.046, #N_eff of relativistic species
            'As': 2.12605e-9, #Amplitude of the primordial scalar power spectrum
            'ns': 0.9674,  # primordial power spectrum tilt
            'w': -1, #Dark energy fluid model
            'wa': 0, #Dark energy fluid model
            'omk': 0.0, #curvature density
            'omegam': None, #DERIVED parameter: Omega matter density
            'omegab': None, #DERIVED parameter: Omega barion density
            'omeganu': None, #DERIVED parameter: Omega neutrino density
            'omnuh2': None, #DERIVED parameter: Omega neutrino density times de reduced Hubble parameter squared
            'omegac': None, #DERIVED parameter: Omega cold dark matter density
            'N_eff': None,
            # (UC): change 'like_selection' based on which observational probe you would like to use.
            # Choose among:
            # 1: photometric survey
            # 2: spectroscopic survey
            # 12: both surveys
            'like_selection': 12,
            # (UC): if you selected the photometric survey (1) or both (12) in 'like_selection'
            # you may want to choose between:
            # using Galaxy Clustering photometric and Weak Lensing probes combined assuming they are independent ('full_photo': False)
            # or Galaxy Clustering photometric, Weak Lensing and the cross-correlation between them ('full_photo': True)
            # This flag is not used if 'like_selection: 2'
            'full_photo': True,
            # (UC): galaxy bias parameters:
            # The bias parameters below are currently fixed to the
            # values used by the Inter Science Taskforce: Forcast (IST:F)
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
            # Spectroscopic bias parameters
            'b1_spec': 1.46,
            'b2_spec': 1.61,
            'b3_spec': 1.75,
            'b4_spec': 1.90},
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
        'theory': {'camb':
                   {'stop_at_error': True,
                    'extra_args':{'num_massive_neutrinos': 1,
                                  'dark_energy_model': 'ppf'}}},
        #'sampler': Cobaya's protected key of the input dictionary.
        # (UC): you can choose the sampler you want to use.
        # Check Cobaya's documentation to see the list of available samplers
        # In this DEMO, we use the 'evaluate' sampler to make a single computation of the posterior distributions
        # WARNING: at the moment, the only sampler that works is 'evaluate'
        'sampler': {'evaluate': None},
        #'output': Cobaya's protected key of the input dictionary.
        # Where are the results going to be stored, in case that the sampler produce output files?
        # For example: chains...
        # (UC): modify the path below within 'output' to choose a name and a directory for those files
        'output': 'chains/my_euclid_experiment',
        #'likelihood': Cobaya's protected key of the input dictionary.
        # (UC): The user can select which data wants to use for the analysis.
        # Check Cobaya's documentation to see the list of the current available data experiments
        # In this DEMO, we load the Euclid-Likelihood as an external function, and name it 'Euclid'
        'likelihood': {'Euclid': EuclidLikelihood},
        #'debug': Cobaya's protected key of the input dictionary.
        # (UC): how much information you want Cobaya to print? If debug: True, it prints every single detail
        # that is going on internally in Cobaya
        'debug': True,
        #'timing': Cobaya's protected key of the input dictionary.
        # (UC): if timing: True, Cobaya returns how much time it took it to make a computation of the posterior
        # and how much time take each of the modules to perform their tasks
        'timing': True,
        #'force': Cobaya's protected key of the input dictionary.
        # (UC): if 'force': True, Cobaya forces deleting the previous output files, if found, with the same name
        'force': True
        }

    # GCH: THIS IS JUST A CALL TO THE LIKELIHOOD
    # FULL CALCULATION PHOTO + SPEC
    from cobaya.model import get_model
    model = get_model(info)
    model.logposterior({})


#########################

if runoption == 1:
    print('Computation of photometric and spectrscopic observables only!')
    print("Initializing the photometric calculation.")

    def mock_MG_func(z, k):
        return 1.0

    class mock_P_obj:
        def __init__(self, p_interp):
            self.P = p_interp

    cur_dir = Path(__file__).resolve().parents[0]

    cmov_file = np.loadtxt(str(cur_dir) +
                           '/likelihood/tests/test_input/ComDist-LCDM-Lin-zNLA.dat')
    zs_r = cmov_file[:, 0]
    rs = cmov_file[:, 1]
    ang_dists = rs / (1.0 + zs_r)
    rz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r, y=rs,
                                                         ext=0)
    dz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r,
                                                         y=ang_dists,
                                                         ext=0)
    Hz_file = np.loadtxt(str(cur_dir) + '/likelihood/tests/test_input/Hz.dat')
    zs_H = Hz_file[:, 0]
    Hs = Hz_file[:, 1]
    Hs_mpc = Hz_file[:, 1] / const.c.to('km/s').value
    Hz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H, y=Hs,
                                                         ext=0)
    Hmpc_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                           y=Hs_mpc,
                                                           ext=0)
    f_sig_8_arr = np.load(str(cur_dir) +
                          '/likelihood/tests/test_input/f_sig_8_arr.npy',
                          allow_pickle=True)
    sig_8_arr = np.load(str(cur_dir) +
                        '/likelihood/tests/test_input/sig_8_arr.npy',
                        allow_pickle=True)
    sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                   x=np.linspace(0.0, 5.0, 50),
                   y=sig_8_arr[::-1], ext=0)
    f_sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                     x=np.linspace(0.0, 5.0, 50),
    y=f_sig_8_arr[::-1], ext=0)

    MG_interp = mock_MG_func

    pdd = np.load(str(cur_dir) + '/likelihood/tests/test_input/pdd.npy')
    pdi = np.load(str(cur_dir) + '/likelihood/tests/test_input/pdi.npy')
    pgd = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgd.npy')
    pgg = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgg.npy')
    pgi_phot = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgi_phot.npy')
    pgi_spec = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgi_spec.npy')
    pii = np.load(str(cur_dir) + '/likelihood/tests/test_input/pii.npy')

    zs_base = np.linspace(0.0, 4.0, 100)
    ks_base = np.logspace(-3.0, 1.0, 100)

    mock_cosmo_dic = {'ombh2': 0.022445, 'omch2': 0.121203, 'H0': 67.0,
                      'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046,
                      'omkh2': 0.0, 'omnuh2': 0.0, 'ns': 0.96,
                      'w': -1.0, 'sigma_8_0': 0.816,
                      'As': 2.115e-9, 'sigma8_z_func': sig_8_interp,
                      'fsigma8_z_func': f_sig_8_interp,
                      'r_z_func': rz_interp, 'd_z_func': dz_interp,
                      'H_z_func_Mpc': Hmpc_interp,
                      'H_z_func': Hz_interp,
                      'z_win': np.linspace(0.0, 4.0, 100),
                      'k_win': np.linspace(0.001, 10.0, 100),
                      'MG_sigma': MG_interp, 'c': const.c.to('km/s').value}
    mock_cosmo_dic['H0_Mpc'] = \
        mock_cosmo_dic['H0'] / const.c.to('km/s').value
    mock_cosmo_dic['Omb'] = \
        mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.)**2.
    mock_cosmo_dic['Omc'] = \
        mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.)**2.
    mock_cosmo_dic['Omnu'] = \
        mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.)**2.
    mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                             mock_cosmo_dic['Omc'] +
                             mock_cosmo_dic['Omb'])

    p_matter = mock_P_obj(interpolate.RectBivariateSpline(zs_base,
                                                          ks_base,
                                                          pdd))
    mock_cosmo_dic['Pk_interpolator'] = p_matter
    mock_cosmo_dic['Pk_delta'] = p_matter
    mock_cosmo_dic['Pgg_phot'] = \
        interpolate.RectBivariateSpline(zs_base,
                                        ks_base,
                                        pgg,
                                        kx=1, ky=1)
    mock_cosmo_dic['Pgdelta_phot'] = \
        interpolate.RectBivariateSpline(zs_base,
                                        ks_base,
                                        pgd,
                                        kx=1, ky=1)
    mock_cosmo_dic['Pii'] = \
        interpolate.RectBivariateSpline(zs_base,
                                        ks_base,
                                        pii,
                                        kx=1, ky=1)
    mock_cosmo_dic['Pdeltai'] = \
        interpolate.RectBivariateSpline(zs_base,
                                        ks_base,
                                        pdi,
                                        kx=1, ky=1)
    mock_cosmo_dic['Pgi_phot'] = \
        interpolate.RectBivariateSpline(zs_base,
                                        ks_base,
                                        pgi_phot,
                                        kx=1, ky=1)
    mock_cosmo_dic['Pgi_spec'] = \
        interpolate.RectBivariateSpline(zs_base,
                                        ks_base,
                                        pgi_spec,
                                        kx=1, ky=1)

    nz_dic_WL = np.load(str(cur_dir) +
                        '/likelihood/tests/test_input/nz_dict_WL.npy',
                        allow_pickle=True).item()
    nz_dic_GC = np.load(str(cur_dir) +
                        '/likelihood/tests/test_input/nz_dict_GC_phot.npy',
                        allow_pickle=True).item()
    phot = photo.Photo(mock_cosmo_dic,
                       nz_dic_WL, nz_dic_GC)

    # from likelihood.photometric_survey.photo import Photo
    # photo = Photo(cosmology.cosmo_dic, nz_dic_WL, nz_dic_GC)
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
    C_GC_11 = np.array([phot.Cl_GC_phot(ell, 1, 1, int_step=int_step_GC) for ell in C_ells_list])
    b=time.time()
    print(C_GC_11)
    print("Time: ", b - a)

    print("Computing shear-shear C_ells")
    print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
    # Compute C_LL_11
    a=time.time()
    C_LL_11 = np.array([phot.Cl_WL(ell, 1, 1, int_step=int_step_WL) for ell in C_ells_list])
    b=time.time()
    print(C_LL_11)
    print("Time: ", b - a)

    print("Computing shear-galaxy C_ells")
    print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
    # Compute C_cross_11
    a=time.time()
    C_cross_11 = np.array([phot.Cl_cross(ell, 1, 1, int_step=int_step_cross) for ell in C_ells_list])
    b=time.time()
    print(C_cross_11)
    print("Time: ", b - a)

    ##############################
    # SPEC
    print("Initializing the spectroscopic calculation.")
    spec_zkm = np.load(str(cur_dir) + '/likelihood/tests/test_input/spec_zkm.npy')
    pgg_spec = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgg_spec.npy')
    mock_cosmo_dic['Pgg_spec'] = interpolate.LinearNDInterpolator(spec_zkm,
                                                                  pgg_spec)
    mock_cosmo_dic['Pgi_spec'] = interpolate.interp2d(zs_base, ks_base,
                                                      pgi_spec.T,
                                                      fill_value=0.0)
    fid_H_arr = np.load(str(cur_dir) + '/likelihood/tests/test_input/spec_fid_HZ.npy')
    fid_d_A_arr = np.load(str(cur_dir) + '/likelihood/tests/test_input/spec_fid_d_A.npy')

    fid_H_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                            y=fid_H_arr,
                                                            ext=0)
    fid_dA_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                             y=fid_d_A_arr,
                                                             ext=0)
    fid_mock_dic = {'H0': 67.5,
                    'omch2': 0.122,
                    'ombh2': 0.022,
                    'omnuh2': 0.00028,
                    'omkh2': 0.0,
                    'w': -1.0,
                    'mnu': 0.06,
                    'tau': 0.07,
                    'nnu': 3.046,
                    'ns': 0.9674,
                    'As': 2.1e-9,
                    'c': const.c.to('km/s').value,
                    'd_z_func': fid_dA_interp,
                    'H_z_func': fid_H_interp,
                    'z_win': zs_base,
                    'k_win': ks_base,
                    'MG_sigma': MG_interp}

    fiducial_dict = fid_mock_dic
    test_dict = mock_cosmo_dic

    spec = Spec(test_dict,
                fiducial_dict)

    print("Computing multipole spectrum P0")
    a=time.time()
    p0_spec = spec.multipole_spectra(1.0, 0.1, 0)
    b=time.time()
    print('P0 = ', p0_spec)
    print("Time: ", b - a)

    print("Computing multipole spectrum P1")
    a=time.time()
    p1_spec = spec.multipole_spectra(1.0, 0.1, 1)
    b=time.time()
    print('P1 = ', p1_spec)
    print("Time: ", b - a)

    #(SJ): uncomment P2/P3/P4 once speed issue is resolved.
    # print("Computing multipole spectrum P2")
    # a=time.time()
    # p2_spec = spec.multipole_spectra(1.0, 0.1, 2)
    # b=time.time()
    # print('P2 = ', p2_spec)
    # print("Time: ", b - a)

    # print("Computing multipole spectrum P3")
    # a=time.time()
    # p3_spec = spec.multipole_spectra(1.0, 0.1, 3)
    # b=time.time()
    # print('P3 = ', p3_spec)
    # print("Time: ", b - a)

    # print("Computing multipole spectrum P4")
    # a=time.time()
    # p4_spec = spec.multipole_spectra(1.0, 0.1, 4)
    # b=time.time()
    # print('P4 = ', p4_spec)
    # print("Time: ", b - a)
    ##############################

print("calculation finished")

# GCH: THE REST IS NOT NEEDED and this is why is commented out

# Import Cosmology module from the Likelihood Package to play with cosmological quantities
#from likelihood.cosmo.cosmology import Cosmology
# Some of the theory needs require extra info (redshift, ks)...
#z_min = 0.0
#z_max = 4.0
#z_samp = 100
#z_win = np.linspace(z_min, z_max, z_samp)
# (SJ): log sampling
# z_win = np.logspace(-2, np.log10(4), 140)
# z_win[0] = 0
# z_win[1] = 1e-4
# z_win[2] = 1e-3
#k_min_Boltzmannn = 0.002
#k_max_Boltzmannn = 10.0
#k_min_GC_phot_interp = 0.001
#k_max_GC_phot_interp = 100.0
#k_samp_GC = 100
#k_win = np.logspace(np.log10(k_min_GC_phot_interp),
#                    np.log10(k_max_GC_phot_interp),
#                    k_samp_GC)


# Cobaya_interface save the cosmology parameters and the cosmology requirements
# from CAMB/CLASS via COBAYA to the cosmology class

# This dictionary collects info from cobaya
#theory_dic = {'H0': model.provider.get_param('H0'),
#              'H0_Mpc': model.provider.get_param('H0') / const.c.to('km/s').value,
#              'omch2': model.provider.get_param('omch2'),
#              'ombh2': model.provider.get_param('ombh2'),
#              'Omc': model.provider.get_param('omch2') / (model.provider.get_param('H0') / 100.)**2.,
#              'Omb': model.provider.get_param('ombh2') / (model.provider.get_param('H0') / 100.)**2.,
#              'mnu': model.provider.get_param('mnu'),
#              'omnuh2': model.provider.get_param('mnu') / 94.07 * (1./3)**0.75,
#              'Omnu': (model.provider.get_param('mnu') / 94.07 * (1./3)**0.75) / (model.provider.get_param('H0') / 100.)**2.,
#              'comov_dist': model.provider.get_comoving_radial_distance(z_win),
#              'angular_dist': model.provider.get_angular_diameter_distance(z_win),
#              'H': model.provider.get_Hubble(z_win),
#              'H_Mpc': model.provider.get_Hubble(z_win, units='1/Mpc'),
#              'Pk_interpolator': model.provider.get_Pk_interpolator(nonlinear=False),
#              'Pk_delta': None,
#              'fsigma8': None,
#              'z_win': z_win,
#              'k_win': k_win
#              }

#theory_dic['Omm'] = theory_dic['Omb'] + theory_dic['Omc'] + theory_dic['Omnu']
#theory_dic['Pk_delta'] = model.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), nonlinear=False)
#theory_dic['fsigma8'] = model.provider.get_fsigma8(z_win)
# Remember: h is hard-coded
#R, z, sigma_R = model.provider.get_sigma_R()
#print(R)
#theory_dic['sigma_8'] = sigma_R[:, 0]

# Initialize cosmology class from likelihood.cosmo.cosmology
# By default: LCDM
#cosmology = Cosmology()
#cosmology.cosmo_dic.update(theory_dic)
#cosmology.update_cosmo_dic(z_win, 0.005)
#
#from likelihood.photometric_survey.photo import Photo
#from likelihood.data_reader.reader import Reader
#
#test_reading = Reader()
#test_reading.compute_nz()
#nz_dic_WL = test_reading.nz_dict_WL
#nz_dic_GC = test_reading.nz_dict_GC_Phot
#
#photo = Photo(cosmology.cosmo_dic, nz_dic_WL, nz_dic_GC)
#
#len_ell_max = 10
#ell_min = 10
#ell_max = 1000
#C_ells_list = np.linspace(ell_min, ell_max, len_ell_max)
#
# These int_step values are for now chosen to achieve internal
# sub-percent precision for the Cls of the 1-1 tomographic bin
# combination within ell of 10 to 1000. The values can be modified
# further during the more rigorous benchmarking phase, where we
# will have decided on the precision required for Euclid.
#int_step_GC = 0.05
#int_step_WL = 0.05
#int_step_cross = 0.02
#
#print("Computing galaxy-galaxy C_ells")
#print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
# Compute C_GC_11
#a=time.time()
#C_GC_11 = np.array([photo.Cl_GC_phot(ell, 1, 1, int_step=int_step_GC) for ell in C_ells_list])
#b=time.time()
#print(C_GC_11)
#print("Time: ", b - a)
#
#print("Computing shear-shear C_ells")
#print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
# Compute C_LL_11
#a=time.time()
#C_LL_11 = np.array([photo.Cl_WL(ell, 1, 1, int_step=int_step_WL) for ell in C_ells_list])
#b=time.time()
#print(C_LL_11)
#print("Time: ", b - a)
#
#print("Computing shear-galaxy C_ells")
#print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
# Compute C_cross_11
#a=time.time()
#C_cross_11 = np.array([photo.Cl_cross(ell, 1, 1, int_step=int_step_cross) for ell in C_ells_list])
#b=time.time()
#print(C_cross_11)
#print("Time: ", b - a)
#
#print("calculation finished")
