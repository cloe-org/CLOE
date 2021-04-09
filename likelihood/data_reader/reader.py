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
    ReaderError
    """

    pass


class Reader:
    """
    Class to read external data files.
    """

    def __init__(self, data_subdirectory='ExternalBenchmark'):
        """Initialize

        Parameters
        ----------
        data_subdirectory: str
            Location of subdirectory within which desired files are stored.
        """
        root_dir = Path(__file__).resolve().parents[2]
        self.dat_dir_main = Path(root_dir, Path('data'),
                                 Path(data_subdirectory))
        self.data_dict = {'GC-Spec': None, 'GC-Phot': None, 'WL': None,
                          'XC-Phot': None}

        # Added dictionaries for n(z)
        # Both raw data and interpolated data
        self.nz_dict_WL = {}
        self.nz_dict_GC_Phot = {}
        self.nz_dict_WL_raw = {}
        self.nz_dict_GC_Phot_raw = {}
        self.numtomo_gcphot = {}
        self.numtomo_wl = {}

        # Added empty dict to fill in fiducial
        # cosmology data from Spec OU-level3 files
        self.data_spec_fiducial_cosmo = {}

        return

    def reader_raw_nz(self, file_dest, file_name):
        """Reader Raw Nz

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
            # Open file and read the content
            f = open(Path(self.dat_dir_main, Path(file_dest),
                          Path(file_name)), "r")
            content = f.read()
            f.close()
            # Obtain the arbitrary header and save in a dict
            nz = np.genfromtxt(content.splitlines(), names=True)
            nz_dict = {x: nz[x] for x in nz.dtype.names}

            return nz_dict
        except BaseException:
            raise Exception(
                'n(z) files not found. Please, check out the files')

    def compute_nz(self,
                   file_dest='Photometric',
                   file_name_GC='niTab-EP10-RB00.dat',
                   file_name_WL='niTab-EP10-RB00.dat'):
        """Compute Nz

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
        # GC-Phot n(z) data
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
        # WL n(z) data
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
                     file_dest='Spectroscopic/data/Sefusatti_multipoles_pk',
                     file_names='cov_power_galaxies_dk0p004_z%s.fits',
                     zstr=["1.", "1.2", "1.4", "1.65"]):
        """Read GC Spec

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
        cur_fname = file_names % zstr[0]
        full_path = Path(self.dat_dir_main, file_dest, cur_fname)
        GC_spec_dict = {}
        fid_cosmo_file = fits.open(full_path)
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
                'sigma8_0': fid_cosmo_file[1].header['SIGMA_8'],
                'w': fid_cosmo_file[1].header['W_STATE'],
                'omkh2': fid_cosmo_file[1].header['OMEGA_K'] *
                fid_cosmo_file[1].header['HUBBLE']**2,
                # OU-LE3 spec files always with omnuh2 = 0
                'omnuh2': 0.000644201,
                'Omnu': 0.001435066}
            # Omega_radiation is ignored here
            fid_cosmo_file.close()

        except ReaderError:
            print('There was an error when reading the fiducial '
                  'data from OU-level3 files')

        k_fac = (self.data_spec_fiducial_cosmo['H0'] / 100.0)
        p_fac = 1.0 / ((self.data_spec_fiducial_cosmo['H0'] / 100.0) ** 3.0)
        cov_fac = p_fac ** 2.0

        for z_label in zstr:
            cur_it_fname = file_names % z_label
            cur_full_path = Path(self.dat_dir_main, file_dest, cur_it_fname)
            fits_file = fits.open(cur_full_path)
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

    def read_phot(self, file_dest='Photometric/data', IA_model_str='zNLA',
                  cov_model_str='Gauss'):
        """Read Phot

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

        full_path = Path(self.dat_dir_main, file_dest)

        GC_file = fits.open(Path(full_path, 'Cls_{:s}_PosPos.fits'.format(
            IA_model_str)))
        WL_file = fits.open(Path(full_path, 'Cls_{:s}_ShearShear.fits'.format(
            IA_model_str)))
        XC_file = fits.open(Path(full_path, 'Cls_{:s}_PosShear.fits'.format(
            IA_model_str)))

        GC_phot_dict['ells'] = GC_file[1].data
        WL_dict['ells'] = WL_file[1].data
        XC_phot_dict['ells'] = XC_file[1].data

        self.numtomo_wl = len(self.nz_dict_WL)
        self.numtomo_gcphot = len(self.nz_dict_GC_Phot)

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

        GC_cov = np.loadtxt(Path(full_path,
                                 'CovMat-PosPos-{:s}-20Bins.dat'.format(
                                     cov_model_str)))
        WL_cov = np.loadtxt(Path(full_path,
                                 'CovMat-ShearShear-{:s}-20Bins.dat'.format(
                                     cov_model_str)))
        tx2_cov = np.loadtxt(Path(full_path,
                                  'CovMat-3x2pt-{:s}-20Bins.dat'.format(
                                      cov_model_str)))

        tot_XC_bins = self.numtomo_wl * self.numtomo_gcphot
        tot_GC_bins = len(GC_cov) / len(GC_phot_dict['ells'])
        tot_WL_bins = len(WL_cov) / len(WL_dict['ells'])
        total_bins = tot_WL_bins + tot_XC_bins + tot_GC_bins
        XC_side = tot_XC_bins * len(XC_phot_dict['ells'])
        XC_cov = np.zeros((XC_side, XC_side))

        for i in range(len(XC_phot_dict['ells'])):
            cur_i = int(i * tot_XC_bins)
            next_i = int((i + 1) * tot_XC_bins)
            XC_i_low = int((i * total_bins) + tot_WL_bins)
            XC_i_hi = int(XC_i_low + tot_XC_bins)
            for j in range(len(XC_phot_dict['ells'])):
                cur_j = int(j * tot_XC_bins)
                next_j = int((j + 1) * tot_XC_bins)
                XC_j_low = int((j * total_bins) + tot_WL_bins)
                XC_j_hi = int(XC_j_low + tot_XC_bins)
                main_extract = tx2_cov[XC_i_low:XC_i_hi, XC_j_low:XC_j_hi]
                XC_cov[cur_i:next_i, cur_j:next_j] = main_extract

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
