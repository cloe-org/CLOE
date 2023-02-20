import sys
import os
from cloe.auxiliary.likelihood_yaml_handler \
	import write_params_yaml_from_info_dict

script_path = os.path.realpath(os.getcwd())
if script_path.endswith('mcmc_scripts'):
    sys.path.append(os.path.realpath(os.path.join(script_path, os.pardir)))
else:
    sys.path.append(script_path)

from cobaya.run import run
from cloe.cobaya_interface import EuclidLikelihood

info = {
        'debug': True,
        'force': True,
        'likelihood':
        {
            'Euclid':
            {
                'aliases': ['euclid'],
                'external': EuclidLikelihood,
                'speed': 500,
                'k_max_extrap': 500.0,
                'k_min_extrap': 1e-05,
                'k_samp': 1000,
                'z_min': 0.0,
                'z_max': 4.0,
                'z_samp': 100,
                'solver': 'camb',
                'NL_flag': 0,
                'add_phot_RSD': False,
                'use_gamma_MG': False,
                'f_out_z_dep': False,
                'plot_observables_selection': False,
                'data':
                {
                    'photo':
                    {
                        'luminosity_ratio': 'luminosity_ratio.dat',
                        'IA_model': 'zNLA',
                        'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npy',
                        'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npy',
                        'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npy',
                        'cov_model': 'Gauss',
                        'ndens_GC': 'niTab-EP10-RB00.dat',
                        'ndens_WL': 'niTab-EP10-RB00.dat',
                        'root_GC': 'Cls_{:s}_PosPos.dat',
                        'root_WL': 'Cls_{:s}_ShearShear.dat',
                        'root_XC': 'Cls_{:s}_PosShear.dat',
                    },
                    'sample': 'ExternalBenchmark',
                    'spectro':
                    {
                        'redshifts': ['1.', '1.2', '1.4', '1.65'],
                        'edges': [0.9, 1.1, 1.3, 1.5, 1.8],
                        'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
                    },
                },
                'observables_selection':
                {
                    'WL':
                    {
                        'WL': True,
                        'GCphot': True,
                        'GCspectro': False,
                    },
                    'GCphot':
                    {
                        'GCphot': True,
                        'GCspectro': False,
                    },
                    'GCspectro':
                    {
                        'GCspectro': True,
                    },
                },
                'observables_specifications':
                {
                    'GCphot':
                    {
                        'statistics': 'angular_power_spectrum',
                        'bins':
                        {
                            'n1':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n2':
                            {
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n3':
                            {
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n4':
                            {
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n5':
                            {
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n6':
                            {
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n7':
                            {
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n8':
                            {
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n9':
                            {
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n10':
                            {
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                        },
                    },
                    'GCspectro':
                    {
                        'statistics': 'legendre_multipole_power_spectrum',
                        'bins':
                        {
                            'n1':
                            {
                                'n1':
                                {
                                    'multipoles':
                                    {
                                        0:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        2:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        4:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                    },
                                },
                            },
                            'n2':
                            {
                                'n2':
                                {
                                    'multipoles':
                                    {
                                        0:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        2:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        4:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                    },
                                },
                            },
                            'n3':
                            {
                                'n3':
                                {
                                    'multipoles':
                                    {
                                        0:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        2:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        4:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                    },
                                },
                            },
                            'n4':
                            {
                                'n4':
                                {
                                    'multipoles':
                                    {
                                        0:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        2:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                        4:
                                        {
                                            'k_range': [[0.01, 0.5]],
                                        },
                                    },
                                },
                            },
                        },
                    },
                    'WL':
                    {
                        'statistics': 'angular_power_spectrum',
                        'bins':
                        {
                            'n1':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n2':
                            {
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n3':
                            {
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n4':
                            {
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n5':
                            {
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n6':
                            {
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n7':
                            {
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n8':
                            {
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n9':
                            {
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n10':
                            {
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                        },
                    },
                    'WL-GCphot':
                    {
                        'statistics': 'angular_power_spectrum',
                        'bins':
                        {
                            'n1':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n2':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n3':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n4':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n5':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n6':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n7':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n8':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n9':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                            'n10':
                            {
                                'n1':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n2':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n3':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n4':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n5':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n6':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n7':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n8':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n9':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                                'n10':
                                {
                                    'ell_range': [[20, 4000]],
                                },
                            },
                        },
                    },
                    'GCphot-GCspectro': None,
                    'WL-GCspectro': None,
                },
            },
        },
        'output': './chains/chain_3x2pt+GCsp_LCDM_nonflat',
        'params':
        {
            'As':
            {
                'latex': 'A_\mathrm{s}',
                'value': 'lambda logA: 1e-10*np.exp(logA)',
            },
            'H0':
            {
                'latex': 'H_0',
                'prior':
                {
                    'max': 100.0,
                    'min': 40.0,
                },
                'proposal': 0.5,
                'ref':
                {
                    'dist': 'norm',
                    'loc': 67.0,
                    'scale': 1.0,
                },
            },
            'logA':
            {
                'drop': True,
                'latex': '\log(10^{10} A_\mathrm{s})',
                'prior':
                {
                    'max': 7.0,
                    'min': 1.6,
                },
                'proposal': 0.001,
                'ref':
                {
                    'dist': 'norm',
                    'loc': 3.05,
                    'scale': 0.001,
                },
            },
            'mnu': 0.06,
            'nnu': 3.046,
            'ns':
            {
                'latex': 'n_\mathrm{s}',
                'prior':
                {
                    'max': 1.2,
                    'min': 0.6,
                },
                'proposal': 0.002,
                'ref':
                {
                    'dist': 'norm',
                    'loc': 0.96,
                    'scale': 0.004,
                },
            },
            'ombh2':
            {
                'latex': '\Omega_\mathrm{b} h^2',
                'prior':
                {
                    'max': 0.1,
                    'min': 0.005,
                },
                'proposal': 0.0001,
                'ref':
                {
                    'dist': 'norm',
                    'loc': 0.0224,
                    'scale': 0.0001,
                },
            },
            'omch2':
            {
                'latex': '\Omega_\mathrm{c} h^2',
                'prior':
                {
                    'max': 0.99,
                    'min': 0.001,
                },
                'proposal': 0.0005,
                'ref':
                {
                    'dist': 'norm',
                    'loc': 0.12,
                    'scale': 0.001,
                },
            },
            'omegam':
            {
                'latex': '\Omega_\mathrm{m}',
            },
            'omk':
            {
                'latex': '\Omega_k',
                'prior':
                {
                    'max': 0.1,
                    'min': -0.1,
                },
                'proposal': 0.05,
                'ref':
                {
                    'dist': 'norm',
                    'loc': 0,
                    'scale': 0.05,
                },
            },
            'sigma8':
            {
                'latex': '\sigma_8',
            },
            'tau': 0.0925,
            'w': -1.0,
            'wa': 0.0,
            'gamma_MG': 0.55,
            'b10_photo': 1.7429859437184225,
            'b1_photo': 1.0997727037892875,
            'b1_spectro': 1.46,
            'b2_photo': 1.220245876862528,
            'b2_spectro': 1.61,
            'b3_photo': 1.2723993083933989,
            'b3_spectro': 1.75,
            'b4_photo': 1.316624471897739,
            'b4_spectro': 1.9,
            'b5_photo': 1.35812370570578,
            'b6_photo': 1.3998214171814918,
            'b7_photo': 1.4446452851824907,
            'b8_photo': 1.4964959071110084,
            'b9_photo': 1.5652475842498528,
            'aia': 1.72,
            'nia': -0.41,
            'bia': 0.0,
            'dz_1_GCphot': 0.0,
            'dz_1_WL': 0.0,
            'dz_2_GCphot': 0.0,
            'dz_2_WL': 0.0,
            'dz_3_GCphot': 0.0,
            'dz_3_WL': 0.0,
            'dz_4_GCphot': 0.0,
            'dz_4_WL': 0.0,
            'dz_5_GCphot': 0.0,
            'dz_5_WL': 0.0,
            'dz_6_GCphot': 0.0,
            'dz_6_WL': 0.0,
            'dz_7_GCphot': 0.0,
            'dz_7_WL': 0.0,
            'dz_8_GCphot': 0.0,
            'dz_8_WL': 0.0,
            'dz_9_GCphot': 0.0,
            'dz_9_WL': 0.0,
            'dz_10_GCphot': 0.0,
            'dz_10_WL': 0.0,
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
            'f_out': 0.0,
            'f_out_1': 0.0,
            'f_out_2': 0.0,
            'f_out_3': 0.0,
            'f_out_4': 0.0,
        },
        'sampler':
        {
            'mcmc':
            {
                'max_tries': 100000,
            },
        },
        'theory':
        {
            'camb':
            {
                'extra_args':
                {
                    'num_nu_massive': 1,
                    'dark_energy_model': 'ppf',
                },
                'stop_at_error': True,
            },
        },
        'timing': True,
}

write_params_yaml_from_info_dict(info)

updated_info, sampler = run(info)
