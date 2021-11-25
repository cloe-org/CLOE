# -*- coding: utf-8 -*-
"""Reader

Contains class to read external data
"""

import numpy as np
from astropy.io import fits, ascii
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

    def __init__(self, data):
        """Initialize

        Parameters
        ----------
        data: dict
            Dictionary containing specifications for data loading and handling.
        """
        self.data = data

        root_dir = Path(__file__).resolve().parents[2]
        self.dat_dir_main = Path(root_dir, Path('data'),
                                 Path(self.data['sample']))
        self.data_dict = {'GC-Spectro': None, 'GC-Phot': None, 'WL': None,
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
        # cosmology data from Spectro OU-level3 files
        self.data_spectro_fiducial_cosmo = {}

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

    def compute_nz(self, file_dest='Photometric'):
        """Compute Nz

        Function to save n(z) dictionaries as attributes of the Reader class
        It saves the interpolators of the raw data.

        Parameters
        ----------
        file_dest: str
            Sub-folder of Reader.data_subdirectory within which to find
            the n(z) data.
        """
        # GC-Phot n(z) data
        file_name_GC = self.data['photo']['ndens_GC']
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
        file_name_WL = self.data['photo']['ndens_WL']
        self.nz_dict_WL_raw.update(
            self.reader_raw_nz(
                file_dest, file_name_WL))
        self.nz_dict_WL.update({x: interpolate.InterpolatedUnivariateSpline(
                                self.nz_dict_WL_raw['z'],
                                self.nz_dict_WL_raw[x] / integrate.trapz(
                                    self.nz_dict_WL_raw[x],
                                    self.nz_dict_WL_raw['z']), ext=2) for x in
                                list(self.nz_dict_WL_raw.keys())[1:]})

    def read_GC_spectro(self, file_dest='Spectroscopic/data'):
        """Read GC Spectro

        Function to read OU-LE3 spectroscopic galaxy clustering files, based
        on location provided to Reader class. Adds contents to the data
        dictionary (Reader.data_dict).

        Parameters
        ----------
        file_dest: str
            Sub-folder of self.data_subdirectory within which to find
            spectroscopic data.
        """
        root = self.data['spectro']['root']
        redshifts = self.data['spectro']['redshifts']

        if 'z{:s}' not in root:
            raise ValueError('GC Spectro file names should contain z{:s} '
                             'string to enable iteration over bins.')
        cur_fname = root.format(redshifts[0])
        full_path = Path(self.dat_dir_main, file_dest, cur_fname)
        GC_spectro_dict = {}
        fid_cosmo_file = fits.open(full_path)
        try:
            self.data_spectro_fiducial_cosmo = {
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
                # OU-LE3 spectro files always with omnuh2 = 0
                'omnuh2': 0.000644201,
                'Omnu': 0.001435066}
            # Omega_radiation is ignored here
            fid_cosmo_file.close()

        except ReaderError:
            print('There was an error when reading the fiducial '
                  'data from OU-level3 files')

        k_fac = (self.data_spectro_fiducial_cosmo['H0'] / 100.0)
        p_fac = 1.0 / ((self.data_spectro_fiducial_cosmo['H0'] / 100.0) ** 3.0)
        cov_fac = p_fac ** 2.0

        for z_label in redshifts:
            cur_it_fname = root.format(z_label)
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

            GC_spectro_dict['{:s}'.format(z_label)] = {'k_pk': kk,
                                                       'pk0': pk0,
                                                       'pk2': pk2,
                                                       'pk4': pk4,
                                                       'cov': cov,
                                                       'cov_k_i': cov_k_i,
                                                       'cov_k_j': cov_k_j,
                                                       'cov_l_i': cov_l_i,
                                                       'cov_l_j': cov_l_j}

            fits_file.close()

        self.data_dict['GC-Spectro'] = GC_spectro_dict
        return

    def read_phot(self, file_dest='Photometric/data'):
        """Read Phot

        Function to read OU-LE3 photometric galaxy clustering and weak lensing
        files, based on location provided to Reader class. Adds contents to
        the data dictionary (Reader.data_dict).

        Parameters
        ----------
        file_dest: str
            Sub-folder of self.data_subdirectory within which to find
            photometric data.
        """
        root_GC = self.data['photo']['root_GC']
        root_WL = self.data['photo']['root_WL']
        root_XC = self.data['photo']['root_XC']
        IA_model = self.data['photo']['IA_model']

        GC_phot_dict = {}
        WL_dict = {}
        XC_phot_dict = {}

        full_path = Path(self.dat_dir_main, file_dest)

        GC_file = ascii.read(
            Path(full_path, root_GC.format(IA_model)),
            encoding='utf-8',
        )
        WL_file = ascii.read(
            Path(full_path, root_WL.format(IA_model)),
            encoding='utf-8',
        )
        XC_file = ascii.read(
            Path(full_path, root_XC.format(IA_model)),
            encoding='utf-8',
        )

        self.numtomo_wl = len(self.nz_dict_WL)
        self.numtomo_gcphot = len(self.nz_dict_GC_Phot)

        header_GC = GC_file.colnames
        for i in range(len(header_GC)):
            GC_phot_dict[header_GC[i]] = GC_file[header_GC[i]].data

        header_WL = WL_file.colnames
        for i in range(len(header_WL)):
            WL_dict[header_WL[i]] = WL_file[header_WL[i]].data

        header_XC = XC_file.colnames
        for i in range(len(header_XC)):
            XC_phot_dict[header_XC[i]] = XC_file[header_XC[i]].data

        tx2_cov_str = self.data['photo']['cov_3x2'].format(self.data[
            'photo']['cov_model'])
        tx2_cov = np.load(Path(full_path, tx2_cov_str))

        self.data_dict['GC-Phot'] = GC_phot_dict
        self.data_dict['WL'] = WL_dict
        self.data_dict['XC-Phot'] = XC_phot_dict
        self.data_dict['cov_3x2'] = tx2_cov

        del(GC_file)
        del(WL_file)
        del(XC_file)
        return
