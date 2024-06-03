"""MOCK OBSERVABLES

Default observables dictionary for unit tests.

"""

from cloe.data_reader.reader import Reader
from cloe.tests.test_input.data import mock_data


def build_mock_observables(reader=None):
    r"""Builds mock observables dictionary for unit tests.
    """
    if reader is None:
        reader = Reader(mock_data)
        reader.compute_nz()
        reader.read_GC_spectro()
        reader.read_GC_spectro_scale_cuts()
        reader.read_phot()

    ell_range = [5, 10000]
    k_range = [0.001, 2.0]
    r_range = [1, 200]

    wl_bins = {}
    for i in range(1, reader.numtomo_wl + 1):
        inner_bins = {}
        for j in range(i, reader.numtomo_wl + 1):
            inner_bins[f'n{j}'] = {'ell_range': [ell_range]}
        wl_bins[f'n{i}'] = inner_bins
    wl_specs = {'statistics': 'angular_power_spectrum',
                'angular_power_spectrum': {'bins': wl_bins}}

    xc_phot_bins = {}
    for i in range(1, reader.numtomo_wl + 1):
        inner_bins = {}
        for j in range(1, reader.numtomo_gcphot + 1):
            inner_bins[f'n{j}'] = {'ell_range': [ell_range]}
        xc_phot_bins[f'n{i}'] = inner_bins
    xc_phot_specs = {'statistics': 'angular_power_spectrum',
                     'angular_power_spectrum': {'bins': xc_phot_bins}}

    gc_phot_bins = {}
    for i in range(1, reader.numtomo_gcphot + 1):
        inner_bins = {}
        for j in range(i, reader.numtomo_gcphot + 1):
            inner_bins[f'n{j}'] = {'ell_range': [ell_range]}
        gc_phot_bins[f'n{i}'] = inner_bins
    gc_phot_specs = {'statistics': 'angular_power_spectrum',
                     'angular_power_spectrum': {'bins': xc_phot_bins}}

    redshifts = reader.data_dict['GC-Spectro'].keys()
    gc_spectro_configuration_space_bins = {}
    for redshift_index, redshift in enumerate(redshifts):
        multipoles = (
            [key for key in
             reader.data_dict['GC-Spectro'][f'{redshift}'].keys()
             if key.startswith('pk')])
        multipole_bins_configuration_space = {}
        for multipole in multipoles:
            multipole_bins_configuration_space[int(multipole[2:])] =\
                {'r_range': [r_range]}
        gc_spectro_configuration_space_bins[f'n{redshift_index+1}'] = {
            f'n{redshift_index+1}': {
                'multipoles': multipole_bins_configuration_space}}
    gc_spectro_specs = {'statistics': 'multipole_power_spectrum',
                        'multipole_correlation_function':
                        {'bins': gc_spectro_configuration_space_bins}}

    observables = {}
    # Note all the probes are set to True
    observables['selection'] = {
        'WL': {'WL': True, 'GCphot': True, 'GCspectro': False},
        'GCphot': {'GCphot': True, 'GCspectro': False},
        'GCspectro': {'GCspectro': True},
        'add_phot_RSD': False,
        'matrix_transform_phot': False,
        'CG': {'CG': False}
    }
    observables['specifications'] = {
        'WL': wl_specs,
        'WL-GCphot': xc_phot_specs,
        'GCphot': gc_phot_specs,
        'GCspectro': gc_spectro_specs,
        'CG': {'CG_probe': 'CC',
               'CG_xi2_cov_selection': 'CG_nonanalytic_cov',
               'neutrino_cdm': 'cb',
               'external_richness_selection_function': 'non_CG_ESF'}
    }

    return observables
