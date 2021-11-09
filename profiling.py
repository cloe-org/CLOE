# General imports
import numpy as np
from scipy import interpolate
import sys
from astropy import constants as const
import time
from pathlib import Path
from likelihood.photometric_survey import photo
from likelihood.spectroscopic_survey.spectro import Spectro
from unittest import TestCase
# Import cobaya -need to be installed
import cobaya
# Import external loglike from the Likelihood Package within cobaya interface module
from likelihood.cobaya_interface import EuclidLikelihood
# Generate likelihood params yaml file
from likelihood.auxiliary.likelihood_yaml_handler import write_params_yaml_from_cobaya_dict
from likelihood.auxiliary.likelihood_yaml_handler import write_data_yaml_from_data_dict

print("Running script: ", sys.argv[0])

# Attention: If working outside of the likelihood environment, change this to your
# local path where your external codes are installed (CAMB, polychord, likelihoods, etc).

if len(sys.argv) > 1:
    runoption = sys.argv[1]
else:
    runoption = 0
runoption = int(runoption)
print('runoption = ', runoption)

if runoption == 0:
    print('Full likelihood evaluation!')
    info = {
        # 'params': Cobaya's protected key of the input dictionary.
        # Includes the parameters that the user would like to sample over:
        'params': {
            # Each parameter below (which is a 'key' of another sub-dictionary) can contain a dictionary
            # with the key 'prior', 'latex', etc.
            # If the prior dictionary is not passed to a parameter, this parameter is fixed.
            # In this example, we are sampling the parameter ns
            #  For more information see: https://cobaya.readthedocs.io/en/latest/example.html
            'ombh2': 0.022445,  # Omega density of baryons times the reduced Hubble parameter squared
            'omch2': 0.1205579307,  # Omega density of cold dark matter times the reduced Hubble parameter squared
            'H0': 67,  # Hubble parameter evaluated today (z=0) in km/s/Mpc
            'tau': 0.0925,  # optical depth
            'mnu': 0.06,  # sum of the mass of neutrinos in eV
            'nnu': 3.046,  # N_eff of relativistic species
            'As': 2.12605e-9,  # Amplitude of the primordial scalar power spectrum
            'ns': 0.9674,  # primordial power spectrum tilt
            'w': -1,  # Dark energy fluid model
            'wa': 0,  # Dark energy fluid model
            'omk': 0.0,  # curvature density
            'omegam': None,  # DERIVED parameter: Omega matter density
            'omegab': None,  # DERIVED parameter: Omega barion density
            'omeganu': None,  # DERIVED parameter: Omega neutrino density
            'omnuh2': None,  # DERIVED parameter: Omega neutrino density times de reduced Hubble parameter squared
            'omegac': None,  # DERIVED parameter: Omega cold dark matter density
            'N_eff': None,
            'NL_flag': 2,
            # Galaxy bias parameters:
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
            # Spectroscopic bias parameters
            'b1_spectro': 1.46,
            'b2_spectro': 1.61,
            'b3_spectro': 1.75,
            'b4_spectro': 1.90,
            # Intrinsic alignment parameters
            'aia': 1.72,
            'nia': -0.41,
            'bia': 0.0},
        # 'theory': Cobaya's protected key of the input dictionary.
        # Cobaya needs to ask some minimum theoretical requirements to a Boltzman Solver
        # You can choose between CAMB or CLASS
        # In this DEMO, we use CAMB and specify some CAMB arguments
        # such as the number of massive neutrinos
        # and the dark energy model
        #
        # Attention: If you have CAMB/CLASS already installed and
        # you are not using the likelihood conda environment
        # or option (2) in cell (3) (Cobaya modules), you can add an extra key called 'path' within the camb dictionary
        # to point to your already installed CAMB code
        'theory': {'camb':
                   {'stop_at_error': True,
                    'extra_args': {'num_massive_neutrinos': 1,
                                   'dark_energy_model': 'ppf',
                                   'halofit_version': 'mead2020'}}},
        # 'sampler': Cobaya's protected key of the input dictionary.
        # You can choose the sampler you want to use.
        # Check Cobaya's documentation to see the list of available samplers
        # In this DEMO, we use the 'evaluate' sampler to make a single computation of the posterior distributions
        # Note: at the moment, the only sampler that works is 'evaluate'
        'sampler': {'evaluate': None},
        # 'output': Cobaya's protected key of the input dictionary.
        # Where are the results going to be stored, in case that the sampler produce output files?
        #  For example: chains...
        # Modify the path below within 'output' to choose a name and a directory for those files
        'output': 'chains/my_euclid_experiment',
        # 'likelihood': Cobaya's protected key of the input dictionary.
        # The user can select which data wants to use for the analysis.
        # Check Cobaya's documentation to see the list of the current available data experiments
        # In this DEMO, we load the Euclid-Likelihood as an external function, and name it 'Euclid'
        'likelihood': {'Euclid': EuclidLikelihood},
        # 'debug': Cobaya's protected key of the input dictionary.
        # How much information you want Cobaya to print. If debug: True, it prints every detail
        # executed internally in Cobaya
        'debug': True,
        # 'timing': Cobaya's protected key of the input dictionary.
        # If timing: True, Cobaya returns how much time it took it to make a computation of the posterior
        # and how much time take each of the modules to perform their tasks
        'timing': True,
        # 'force': Cobaya's protected key of the input dictionary.
        # If 'force': True, Cobaya forces deleting the previous output files, if found, with the same name
        'force': True,
        'data': {
            # 'sample' specifies the first folder below the main data folder
            'sample': 'ExternalBenchmark',
            # 'spectro' and 'photo' specify paths to data files.
            'spectro': {
                # GC Spectro root name should contain z{:s} string
                # to enable iteration over bins
                'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
                'redshifts': ["1.", "1.2", "1.4", "1.65"]
                },
            'photo': {
                'ndens_GC': 'niTab-EP10-RB00.dat',
                'ndens_WL': 'niTab-EP10-RB00.dat',
                # Photometric root names should contain z{:s} string
                # to specify IA model
                'root_GC': 'Cls_{:s}_PosPos.dat',
                'root_WL': 'Cls_{:s}_ShearShear.dat',
                'root_XC': 'Cls_{:s}_PosShear.dat',
                'IA_model': 'zNLA',
                # Photometric covariances root names should contain z{:s} string
                # to specify how the covariance was calculated
                'cov_GC': 'CovMat-PosPos-{:s}-20Bins.dat',
                'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.dat',
                'cov_3x2': 'CovMat-3x2pt-{:s}-20Bins.dat',
                'cov_model': 'Gauss'
                }
            }
        }

    write_data_yaml_from_data_dict(info['data'])
    write_params_yaml_from_cobaya_dict(info)

    # This is just a call to the likelihood
    # Full calculation photo + spectro
    from cobaya.model import get_model
    model = get_model(info)
    model.logposterior({})


#########################

if runoption == 1:
    print('Computation of photometric and spectrscopic observables only!')
    print("Initializing the photometric calculation.")

    cur_dir = Path(__file__).resolve().parents[0]

    def mock_MG_func(z, k):
        return 1.0

    class mock_P_obj:
        def __init__(self, p_interp):
            self.P = p_interp

    class cosmoinitTestCase(TestCase):

        def __init__(self):
            self.jaja = 1

        def setUp(self):
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
            f_z_arr = np.load(str(cur_dir) +
                              '/likelihood/tests/test_input/f_z_arr.npy',
                              allow_pickle=True)

            sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                x=np.linspace(0.0, 5.0, 50),
                y=sig_8_arr[::-1], ext=0)
            f_sig_8_interp = interpolate.InterpolatedUnivariateSpline(
                x=np.linspace(0.0, 5.0, 50),
                y=f_sig_8_arr[::-1], ext=0)
            f_z_interp = interpolate.InterpolatedUnivariateSpline(
                x=np.linspace(0.0, 5.0, 50),
                y=f_z_arr[::-1], ext=0)

            MG_interp = mock_MG_func

            pdd = np.load(str(cur_dir) + '/likelihood/tests/test_input/pdd.npy')
            pdi = np.load(str(cur_dir) + '/likelihood/tests/test_input/pdi.npy')
            pgd = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgd.npy')
            pgg = np.load(str(cur_dir) + '/likelihood/tests/test_input/pgg.npy')
            pgi_phot = np.load(str(cur_dir)
                               + '/likelihood/tests/test_input/pgi_phot.npy')
            pgi_spectro = np.load(str(cur_dir)
                                  + '/likelihood/tests/test_input/pgi_spectro.npy')
            pii = np.load(str(cur_dir) + '/likelihood/tests/test_input/pii.npy')

            zs_base = np.linspace(0.0, 4.0, 100)
            ks_base = np.logspace(-3.0, 1.0, 100)

            mock_cosmo_dic = {'ombh2': 0.022445, 'omch2': 0.121203, 'H0': 67.0,
                              'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046,
                              'omkh2': 0.0, 'omnuh2': 0.0, 'ns': 0.96,
                              'w': -1.0, 'sigma_8_0': 0.816,
                              'As': 2.115e-9, 'sigma8_z_func': sig_8_interp,
                              'fsigma8_z_func': f_sig_8_interp,
                              'f_z': f_z_interp,
                              'r_z_func': rz_interp, 'd_z_func': dz_interp,
                              'H_z_func_Mpc': Hmpc_interp,
                              'H_z_func': Hz_interp,
                              'z_win': zs_base,
                              'k_win': ks_base,
                              'MG_sigma': MG_interp, 'c': const.c.to('km/s').value,
                              'nuisance_parameters': {
                                  'NL_flag': 1,
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
                                  'aia': 1.72,
                                  'nia': -0.41,
                                  'bia': 0.0}
                              }

            mock_cosmo_dic['H0_Mpc'] = \
                mock_cosmo_dic['H0'] / const.c.to('km/s').value
            mock_cosmo_dic['Omb'] = \
                mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
            mock_cosmo_dic['Omc'] = \
                mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
            mock_cosmo_dic['Omnu'] = \
                mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
            mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                                     mock_cosmo_dic['Omc'] +
                                     mock_cosmo_dic['Omb'])
            p_matter = interpolate.RectBivariateSpline(zs_base,
                                                       ks_base,
                                                       pdd,
                                                       kx=1, ky=1)
            mock_cosmo_dic['Pk_delta'] = mock_P_obj(p_matter)
            mock_cosmo_dic['Pmm_phot'] = p_matter
            mock_cosmo_dic['Pgg_phot'] = \
                interpolate.RectBivariateSpline(zs_base, ks_base,
                                                pgg.T, kx=1, ky=1)
            mock_cosmo_dic['Pgdelta_phot'] = \
                interpolate.RectBivariateSpline(zs_base, ks_base,
                                                pgd.T, kx=1, ky=1)
            mock_cosmo_dic['Pii'] = \
                interpolate.RectBivariateSpline(zs_base, ks_base,
                                                pii.T, kx=1, ky=1)
            mock_cosmo_dic['Pdeltai'] = \
                interpolate.RectBivariateSpline(zs_base, ks_base,
                                                pdi.T, kx=1, ky=1)

            mock_cosmo_dic['Pgi_phot'] = \
                interpolate.RectBivariateSpline(zs_base, ks_base,
                                                pgi_phot.T, kx=1, ky=1)

            mock_cosmo_dic['Pgi_spectro'] = \
                interpolate.RectBivariateSpline(zs_base, ks_base,
                                                pgi_spectro.T, kx=1, ky=1)

            mock_cosmo_dic['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)

            fid_H_arr = np.load(str(cur_dir) + '/likelihood/tests/test_input/spectro_fid_HZ.npy')
            fid_d_A_arr = np.load(str(cur_dir) + '/likelihood/tests/test_input/spectro_fid_d_A.npy')

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
                            'MG_sigma': MG_interp,
                            'nuisance_parameters': {
                                'NL_flag': 1,
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
                                'aia': 1.72,
                                'nia': -0.41,
                                'bia': 0.0}
                            }

            self.fiducial_dict = fid_mock_dic
            self.test_dict = mock_cosmo_dic

            self.spectro = Spectro(self.test_dict, self.fiducial_dict)

            nz_dic_WL = np.load(str(cur_dir) +
                                '/likelihood/tests/test_input/nz_dict_WL.npy',
                                allow_pickle=True).item()
            nz_dic_GC = np.load(str(cur_dir) +
                                '/likelihood/tests/test_input/nz_dict_GC_phot.npy',
                                allow_pickle=True).item()

            self.phot = photo.Photo(self.test_dict, nz_dic_WL, nz_dic_GC)

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
            print("C_ells_list: ", C_ells_list)
            # Compute C_GC_11
            a = time.time()
            C_GC_11 = np.array([self.phot.Cl_GC_phot(ell, 1, 1, int_step=int_step_GC) for ell in C_ells_list])
            b = time.time()
            print('C_{G_phot-G_phot} = ', C_GC_11)
            print("Time: ", b - a)

            print("Computing shear-shear C_ells")
            print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
            # Compute C_LL_11
            a = time.time()
            C_LL_11 = np.array([self.phot.Cl_WL(ell, 1, 1, int_step=int_step_WL) for ell in C_ells_list])
            b = time.time()
            print('C_{shear-shear} = ', C_LL_11)
            print("Time: ", b - a)

            print("Computing shear-galaxy C_ells")
            print("len_ell_max: ", len_ell_max, "  ell_min: ", ell_min, "  ell_max:", ell_max)
            # Compute C_cross_11
            a = time.time()
            C_cross_11 = np.array([self.phot.Cl_cross(ell, 1, 1, int_step=int_step_cross) for ell in C_ells_list])
            b = time.time()
            print('C_{G_phot-shear} = ', C_cross_11)
            print("Time: ", b - a)

            ##############################
            # SPECTRO
            print("Initializing the spectroscopic calculation.")

            print("All examples are here at z = 1, k = 0.1/Mpc.")
            print("Computing multipole spectrum P0")
            a = time.time()
            p0_spectro = self.spectro.multipole_spectra(1.0, 0.1, ms=[0])
            b = time.time()
            print('P0 = ', p0_spectro)
            print("Time: ", b - a)

            print("Computing multipole spectrum P1")
            a = time.time()
            p1_spectro = self.spectro.multipole_spectra(1.0, 0.1, ms=[1])
            b = time.time()
            print('P1 = ', p1_spectro)
            print("Time: ", b - a)

            print("Computing multipole spectrum P2")
            a = time.time()
            p2_spectro = self.spectro.multipole_spectra(1.0, 0.1, ms=[2])
            b = time.time()
            print('P2 = ', p2_spectro)
            print("Time: ", b - a)

            print("Computing multipole spectrum P3")
            a = time.time()
            p3_spectro = self.spectro.multipole_spectra(1.0, 0.1, ms=[3])
            b = time.time()
            print('P3 = ', p3_spectro)
            print("Time: ", b - a)

            print("Computing multipole spectrum P4")
            a = time.time()
            p4_spectro = self.spectro.multipole_spectra(1.0, 0.1, ms=[4])
            b = time.time()
            print('P4 = ', p4_spectro)
            print("Time: ", b - a)

            print("Computing ALL multipole spectra")
            a = time.time()
            pall_spectro = self.spectro.multipole_spectra(1.0, 0.1)
            b = time.time()
            print('P024 = ', pall_spectro)
            print("Time: ", b - a)

        def istf_spectro_galbias(self, redshift, bin_edge_list=None):

            if bin_edge_list is None:
                bin_edge_list = [0.90, 1.10, 1.30, 1.50, 1.80]

            istf_bias_list = [self.test_dict['nuisance_parameters']['b1_spectro'],
                              self.test_dict['nuisance_parameters']['b2_spectro'],
                              self.test_dict['nuisance_parameters']['b3_spectro'],
                              self.test_dict['nuisance_parameters']['b4_spectro']]

            if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
                for i in range(len(bin_edge_list) - 1):
                    if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                        bi_val = istf_bias_list[i]
            elif redshift >= bin_edge_list[-1]:
                bi_val = istf_bias_list[-1]
            elif redshift < bin_edge_list[0]:
                bi_val = istf_bias_list[0]
            return bi_val

        def Pgg_spectro_def(self, redshift, k_scale, mu_rsd):
            bias = self.istf_spectro_galbias(redshift)
            growth = self.test_dict['f_z'](redshift)
            power = self.test_dict['Pk_delta'].P(redshift, k_scale)
            pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
            return pval

    cosmoinitTestCase().setUp()

print("calculation finished")
