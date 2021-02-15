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
print(p0_spec)
print("Time: ", b - a)

print("Computing multipole spectrum P1")
a=time.time()
p1_spec = spec.multipole_spectra(1.0, 0.1, 1)
b=time.time()
print(p1_spec)
print("Time: ", b - a)

print("Computing multipole spectrum P2")
a=time.time()
p2_spec = spec.multipole_spectra(1.0, 0.1, 2)
b=time.time()
print(p2_spec)
print("Time: ", b - a)

print("Computing multipole spectrum P3")
a=time.time()
p3_spec = spec.multipole_spectra(1.0, 0.1, 3)
b=time.time()
print(p3_spec)
print("Time: ", b - a)

print("Computing multipole spectrum P4")
a=time.time()
p4_spec = spec.multipole_spectra(1.0, 0.1, 4)
b=time.time()
print(p4_spec)
print("Time: ", b - a)
##############################

print("calculation finished")
