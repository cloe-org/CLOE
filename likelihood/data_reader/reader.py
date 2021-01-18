# -*- coding: utf-8 -*-
"""Reader

Contains class to read external data
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from scipy import interpolate
from scipy import integrate


class ReaderError(Exception):
    r"""

    """
    pass


class Reader:
    """
    Class to read external data files.
    """

    def __init__(self, data_subdirectory='/ExternalBenchmark'):
        """
        Parameters
        ----------
        data_subdirectory: str
            Location of subdirectory within which desired files are stored.
        """
        root_dir = Path(__file__).resolve().parents[2]
        self.dat_dir_main = str(root_dir) + '/data' + data_subdirectory
        self.data_dict = {'GC-Spec': None, 'GC-Phot': None, 'WL': None,
                          'XC-Phot': None}

        # (GCH): Added dictionaries for n(z)
        # Both raw data and interpolated data
        self.nz_dict_WL = {}
        self.nz_dict_GC_Phot = {}
        self.nz_dict_WL_raw = {}
        self.nz_dict_GC_Phot_raw = {}

        # (GCH): Added empty dict to fill in
        # fiducial cosmology data from Spec OU-level3 files
        self.data_spec_fiducial_cosmo = {}

        return

    def reader_raw_nz(self, file_dest, file_name):
        """
        General routine to read the galaxy density
        distribution n(z) files

        Parameters
        ----------
        file_dest: str
            Sub-folder of Reader.data_subdirectory within which to find
            the n(z) data.
        file_name: str
            Name of the n(z) files

        Return
        ------
        nz_dict: dict
            dictionary containing raw n(z) data
        """
        try:
            # (GCH): open file and read the content
            f = open(self.dat_dir_main + file_dest + file_name, "r")
            content = f.read()
            # (GCH): get the arbitrary header and save in a dict
            nz = np.genfromtxt(content.splitlines(), names=True)
            nz_dict = {x: nz[x] for x in nz.dtype.names}
            f.close()
            return nz_dict
        except BaseException:
            raise Exception(
                'n(z) files not found. Please, check out the files')

    def compute_nz(self,
                   file_dest='/Photometric/',
                   file_name_GC='niTab-EP10-RB00.dat',
                   file_name_WL='niTab-EP10-RB00.dat'):
        """
        Function to save n(z) dictionaries as attributes of the Reader class
        It saves the interpolators of the raw data.

        Parameters
        ----------
        file_dest: str
            Sub-folder of Reader.data_subdirectory within which to find
            the n(z) data.
        file_name_GC: str
            Name of the n(z) files for GC
        file_name_WL: str
            Name of the n(z) files for WL
        """
        # (GCH): GC n(z) data
        self.nz_dict_GC_Phot_raw.update(
            self.reader_raw_nz(
                file_dest, file_name_GC))
        self.nz_dict_GC_Phot.update(
            {
                x: interpolate.InterpolatedUnivariateSpline(
                    self.nz_dict_GC_Phot_raw['z'],
                    self.nz_dict_GC_Phot_raw[x] / integrate.trapz(
                        self.nz_dict_GC_Phot_raw[x],
                        self.nz_dict_GC_Phot_raw['z']), ext=2) for x in list(
                    self.nz_dict_GC_Phot_raw.keys())[1:]})
        # (GCH): WL n(z) data
        self.nz_dict_WL_raw.update(
            self.reader_raw_nz(
                file_dest, file_name_WL))
        self.nz_dict_WL.update({x: interpolate.InterpolatedUnivariateSpline(
                                self.nz_dict_WL_raw['z'],
                                self.nz_dict_WL_raw[x] / integrate.trapz(
                                    self.nz_dict_WL_raw[x],
                                    self.nz_dict_WL_raw['z']), ext=2) for x in
                                list(self.nz_dict_WL_raw.keys())[1:]})

    def read_GC_spec(self,
                     file_dest='/Spectroscopic/data/Sefusatti_multipoles_pk/',
                     file_names='cov_power_galaxies_dk0p004_z%s.fits',
                     zstr=["1.", "1.2", "1.4", "1.65"]):
        """
        Function to read OU-LE3 spectroscopic galaxy clustering files, based
        on location provided to Reader class. Adds contents to the data
        dictionary (Reader.data_dict).

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
        fid_cosmo_file = fits.open(full_path % zstr[0])
        try:
            self.data_spec_fiducial_cosmo = {
                'H0': fid_cosmo_file[1].header['HUBBLE'] *
                100,
                'omch2': (fid_cosmo_file[1].header['OMEGA_M'] -
                          fid_cosmo_file[1].header['OMEGA_B']) *
                fid_cosmo_file[1].header['HUBBLE']**2,
                'ombh2': fid_cosmo_file[1].header['OMEGA_B'] *
                fid_cosmo_file[1].header['HUBBLE']**2,
                'ns': fid_cosmo_file[1].header['INDEX_N'],
                'sigma_8_0': fid_cosmo_file[1].header['SIGMA_8'],
                'w': fid_cosmo_file[1].header['W_STATE'],
                'omkh2': fid_cosmo_file[1].header['OMEGA_K'] *
                fid_cosmo_file[1].header['HUBBLE']**2,
                'omnuh2': 0}
        # GCH: remember, for the moment we ignore Omega_R and
        # neutrinos
        except ReaderError:
            print('There was an error when reading the fiducial '
                  'data from OU-level3 files')

        fid_cosmo_file.close()

        k_fac = (self.data_spec_fiducial_cosmo['H0'] / 100.0)
        p_fac = 1.0 / ((self.data_spec_fiducial_cosmo['H0'] / 100.0) ** 3.0)
        cov_fac = p_fac ** 2.0

        for z_label in zstr:
            fits_file = fits.open(full_path % z_label)
            average = fits_file[1].data
            kk = average["SCALE_1DIM"] * k_fac
            pk0 = average["AVERAGE0"] * p_fac
            pk2 = average["AVERAGE2"] * p_fac
            pk4 = average["AVERAGE4"] * p_fac

            cov = fits_file[2].data["COVARIANCE"] * cov_fac
            cov_k_i = fits_file[2].data["SCALE_1DIM-I"] * k_fac
            cov_k_j = fits_file[2].data["SCALE_1DIM-J"] * k_fac
            cov_l_i = fits_file[2].data["MULTIPOLE-I"]
            cov_l_j = fits_file[2].data["MULTIPOLE-J"]

            nk = len(kk)
            cov = np.reshape(cov, newshape=(3 * nk, 3 * nk))
            cov_k_i = np.reshape(cov_k_i, newshape=(3 * nk, 3 * nk))
            cov_k_j = np.reshape(cov_k_j, newshape=(3 * nk, 3 * nk))
            cov_l_i = np.reshape(cov_l_i, newshape=(3 * nk, 3 * nk))
            cov_l_j = np.reshape(cov_l_j, newshape=(3 * nk, 3 * nk))

            GC_spec_dict['{:s}'.format(z_label)] = {'k_pk': kk,
                                                    'pk0': pk0,
                                                    'pk2': pk2,
                                                    'pk4': pk4,
                                                    'cov': cov,
                                                    'cov_k_i': cov_k_i,
                                                    'cov_k_j': cov_k_j,
                                                    'cov_l_i': cov_l_i,
                                                    'cov_l_j': cov_l_j}

            fits_file.close()

        self.data_dict['GC-Spec'] = GC_spec_dict
        return

    def read_phot(self, file_dest='/Photometric/data/', IA_model_str='zNLA',
                  cov_model_str='Gauss'):
        """
        Function to read OU-LE3 photometric galaxy clustering and weak lensing
        files, based on location provided to Reader class. Adds contents to
        the data dictionary (Reader.data_dict).

        Parameters
        ----------
        file_dest: str
            Sub-folder of self.data_subdirectory within which to find
            photometric data.
        IA_model_str: str
            String used to denote particular intrinsic alignment model used.
        cov_model_str: str
            String used to denote type of covariance matrices constructed.
            E.g. 'Gauss' for Gaussian, 'GaussSSC' for Gaussian + Super Sampled
            Covariance.
        """

        GC_phot_dict = {}
        WL_dict = {}
        XC_phot_dict = {}

        full_path = self.dat_dir_main + file_dest

        GC_file = fits.open(full_path + 'Cls_{:s}_PosPos.fits'.format(
            IA_model_str))
        WL_file = fits.open(full_path + 'Cls_{:s}_ShearShear.fits'.format(
            IA_model_str))
        XC_file = fits.open(full_path + 'Cls_{:s}_PosShear.fits'.format(
            IA_model_str))

        GC_phot_dict['ells'] = GC_file[1].data
        WL_dict['ells'] = WL_file[1].data
        XC_phot_dict['ells'] = XC_file[1].data

        for i in range(2, len(GC_file)):
            cur_ind = GC_file[i].header['EXTNAME']
            cur_comb = GC_file[i].header['BIN_COMB']
            if len(cur_comb) > 3:
                if cur_comb[:2] == '10':
                    left_digit = '10'
                else:
                    left_digit = cur_comb[0]
                if cur_comb[-2:] == '10':
                    right_digit = '10'
                else:
                    right_digit = cur_comb[-1]
            else:
                left_digit = cur_comb[0]
                right_digit = cur_comb[-1]
            cur_lab = cur_ind[0] + left_digit + '-' + cur_ind[2] + right_digit
            GC_phot_dict[cur_lab] = GC_file[i].data

        for j in range(2, len(WL_file)):
            cur_ind = WL_file[j].header['EXTNAME']
            cur_comb = WL_file[j].header['BIN_COMB']
            if len(cur_comb) > 3:
                if cur_comb[:2] == '10':
                    left_digit = '10'
                else:
                    left_digit = cur_comb[0]
                if cur_comb[-2:] == '10':
                    right_digit = '10'
                else:
                    right_digit = cur_comb[-1]
            else:
                left_digit = cur_comb[0]
                right_digit = cur_comb[-1]
            cur_lab = cur_ind[0] + left_digit + '-' + cur_ind[2] + right_digit
            WL_dict[cur_lab] = WL_file[j].data

        for k in range(2, len(XC_file)):
            cur_ind = XC_file[k].header['EXTNAME']
            cur_comb = XC_file[k].header['BIN_COMB']
            if len(cur_comb) > 3:
                if cur_comb[:2] == '10':
                    left_digit = '10'
                else:
                    left_digit = cur_comb[0]
                if cur_comb[-2:] == '10':
                    right_digit = '10'
                else:
                    right_digit = cur_comb[-1]
            else:
                left_digit = cur_comb[0]
                right_digit = cur_comb[-1]
            cur_lab = cur_ind[0] + left_digit + '-' + cur_ind[2] + right_digit
            XC_phot_dict[cur_lab] = XC_file[k].data

        GC_cov = np.loadtxt(full_path + 'CovMat-PosPos-{:s}-20Bins.dat'.format(
            cov_model_str))
        WL_cov = np.loadtxt(full_path +
                            'CovMat-ShearShear-{:s}-20Bins.dat'.format(
                                cov_model_str))
        tx2_cov = np.loadtxt(full_path + 'CovMat-3x2pt-{:s}-20Bins.dat'.format(
            cov_model_str))
        XC_cov = tx2_cov[WL_cov.shape[0]:-GC_cov.shape[0],
                         WL_cov.shape[1]:-GC_cov.shape[1]]

        GC_phot_dict['cov'] = GC_cov
        WL_dict['cov'] = WL_cov
        XC_phot_dict['cov'] = tx2_cov
        XC_phot_dict['cov_XC_only'] = XC_cov

        self.data_dict['GC-Phot'] = GC_phot_dict
        self.data_dict['WL'] = WL_dict
        self.data_dict['XC-Phot'] = XC_phot_dict

        GC_file.close()
        WL_file.close()
        XC_file.close()
        return
