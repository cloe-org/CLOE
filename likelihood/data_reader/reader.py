# -*- coding: utf-8 -*-
"""Reader

Contains class to read external data
"""

import numpy as np
from astropy.io import fits
from pathlib import Path


class Reader:
    """
    Class to read external data files.
    """
    def __init__(self, data_subdirectory='/ExternalBenchmark', no_bins_WL=10,
                 no_bins_GC_Phot=10):
        """
        Parameters
        ----------
        data_subdirectory: str
            Location of subdirectory within which desired files are stored.
        no_bins_WL: int
            Number of redshift bins for photometric WL probe.
        no_bins_GC_Phot: int
            Number of redshift bins for photometric GC probe.
        """
        root_dir = Path(__file__).resolve().parents[2]
        self.dat_dir_main = str(root_dir) + '/data' + data_subdirectory
        self.data_dict = {'GC-Spec': None, 'GC-Phot': None, 'WL': None,
                          'XC-Phot': None}
        self.nz_dict_WL = {}
        self.nz_dict_GC_Phot = {}

        for bin_WL in range(1, no_bins_WL + 1):
            self.nz_dict_WL[bin_WL] = None

        for bin_GC_Phot in range(1, no_bins_GC_Phot + 1):
            self.nz_dict_GC_Phot[bin_GC_Phot] = None
        return

    def read_GC_spec(self,
                     file_dest='/Spectroscopic/data/Sefusatti_multipoles_pk/',
                     file_names='cov_power_galaxies_dk0p004_z%s.fits',
                     zstr=["1.", "1.2", "1.4", "1.65"]):
        """
        Function to read OU-LE3 spectroscopic galaxy clustering files, based
        on location provided to class. Adds contents to the data dictionary
        (self.data_dict).

        Parameters
        ----------
        file_dest: str
            Sub-folder of self.data_subdirectory within which to find
            spectroscopic data.
        file_names: str
            General structure of file names. Note: must contain 'z%s' to
            enable iteration over redshifts.
        zstr: list
            List of strings denoting spectroscopic redshift bins.
        """
        if 'z%s' not in file_names:
            raise Exception('GC Spec file names should contain z%s string to '
                            'enable iteration over bins.')

        full_path = self.dat_dir_main + file_dest + file_names
        GC_spec_dict = {}

        for z_label in zstr:
            fits_file = fits.open(full_path % z_label)
            average = fits_file[1].data
            kk = average["SCALE_1DIM"]
            pk0 = average["AVERAGE0"]
            pk2 = average["AVERAGE2"]
            pk4 = average["AVERAGE4"]

            cov = fits_file[2].data["COVARIANCE"]
            cov_k_i = fits_file[2].data["SCALE_1DIM-I"]
            cov_k_j = fits_file[2].data["SCALE_1DIM-J"]
            cov_l_i = fits_file[2].data["MULTIPOLE-I"]
            cov_l_j = fits_file[2].data["MULTIPOLE-J"]

            nk = len(kk)
            cov = np.reshape(cov, newshape=(3 * nk, 3 * nk))
            cov_k_i = np.reshape(cov_k_i, newshape=(3 * nk, 3 * nk))
            cov_k_j = np.reshape(cov_k_j, newshape=(3 * nk, 3 * nk))
            cov_l_i = np.reshape(cov_l_i, newshape=(3 * nk, 3 * nk))
            cov_l_j = np.reshape(cov_l_j, newshape=(3 * nk, 3 * nk))

            GC_spec_dict['z={:s}'.format(z_label)] = {'k_pk': kk,
                                                      'pk0': pk0,
                                                      'pk2': pk2,
                                                      'pk4': pk4,
                                                      'cov': cov,
                                                      'cov_k_i': cov_k_i,
                                                      'cov_k_j': cov_k_j,
                                                      'cov_l_i': cov_l_i,
                                                      'cov_l_j': cov_l_j}
        self.data_dict['GC-Spec'] = GC_spec_dict
        return
