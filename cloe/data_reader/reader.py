# -*- coding: utf-8 -*-
"""READER

Contains class to read external data.
"""

import numpy as np
import yaml
import logging
from astropy.io import fits, ascii
from pathlib import Path
from scipy import interpolate
from scipy import integrate
from euclidlib.photo._le3_pk_wl import angular_power_spectra
from euclidlib.photo import mixing_matrices


class ReaderError(Exception):
    r"""
    ReaderError.
    """

    pass


class Reader:
    """
    Class to read external data files.
    """

    def __init__(self, data):
        """Initializes the :obj:`Reader` class.

        Parameters
        ----------
        data: dict
            Dictionary containing specifications for data loading and handling.
        """
        self.data = data

        self.root_dir = Path(__file__).resolve().parents[2]
        self.dat_dir_main = Path(self.root_dir, Path('data'),
                                 Path(self.data['sample']))
        self.data_dict = {'GC-Spectro': None, 'GC-Phot': None, 'WL': None,
                          'XC-Phot': None, 'CG': None}

        # Added dictionaries for n(z)
        # Both raw data and interpolated data
        self.nz_dict_WL = {}
        self.nz_dict_GC_Phot = {}
        self.nz_dict_WL_raw = {}
        self.nz_dict_GC_Phot_raw = {}
        self.numtomo_gcphot = {}
        self.numtomo_wl = {}
        self.GC_spectro_scale_cuts = {}
        self.luminosity_ratio_interpolator = None

        # Added empty dict to fill in fiducial
        # cosmology data from Spectro OU-level3 files
        self.data_spectro_fiducial_cosmo = {}

        # (ZS): Added empty dict to fill in
        # fiducial cosmology data from Spec OU-level3 files
        self.data_CG_fiducial_cosmo = {}

        return

    def reader_raw_nz(self, file_dest, file_name):
        """Reads in the raw redshift distributions.

        General routine to read the redshift distribution files.

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`Reader.data_subdirectory` within which
            to find the redshift distribution data
        file_name: str
            Name of the redshift distribution files

        Return
        ------
        Raw redshift distribution: dict
            Dictionary containing raw redshift distribution data
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

    def reader_luminosity_ratio(self, file_dest, file_name):
        """Reads in the luminosity ratio file.

        General routine to read the the luminosity ratio
        used to compute the IA as a function of redshift
        data.

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`Reader.data_subdirectory` within which
            to find the luminosity ratio as a function of redshift
        file_name: str
            Name of the luminosity file

        Return
        ------
        Raw luminosity ratio: dict
            Dictionary containing raw luminosity ratio data
        """
        try:
            path = Path(self.dat_dir_main, Path(file_dest), Path(file_name))
            luminosity_ratio = np.genfromtxt(path, names=True)
            luminosity_ratio_dict = {x: luminosity_ratio[x] for x
                                     in luminosity_ratio.dtype.names}

            return luminosity_ratio_dict
        except BaseException:
            raise Exception(
                'Luminosity ratio file not found. Please, check out the file')

    def compute_nz(self, file_dest='Photometric'):
        """Stores the redshift distributions.

        Function to save the redshift distribution dictionaries
        as attributes of the :obj:`Reader` class.

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`Reader.data_subdirectory` within which
            to find the redshift distribution data
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

    def compute_luminosity_ratio(self, file_dest='Photometric'):
        """Stores luminosity ratio.

        Function to save luminosty ratio dict as attributes of the
        :obj:`Reader` class.
        It saves the interpolator of the raw data.

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`Reader.data_subdirectory` within which
            to find the luminosity ratio data.
        """
        file_name_lum = self.data['photo']['luminosity_ratio']
        self.luminosity_ratio = self.reader_luminosity_ratio(
            file_dest, file_name_lum)
        self.luminosity_ratio_interpolator = \
            interpolate.InterpolatedUnivariateSpline(
                self.luminosity_ratio['z'],
                self.luminosity_ratio['luminosity'])

    def read_GC_spectro_scale_cuts(self, config_folder='configs',
                                   data_folder='Spectroscopic/data'):
        r"""Reads the spectroscopic scale cuts and converts units.

        Function to read the scale cuts specified in the configuration data.
        The input is being read and saved as a dictionary within the
        :obj:`Reader` class.
        The units from the input file are converted from :math:`\frac{h}{Mpc}`
        to :math:`\frac{1}{Mpc}` to be consistent with the input of the data
        vectors. In the rest of the code :math:`\frac{1}{Mpc}` is used.

        Parameters
        ----------
        config_folder: str
            Sub-folder of :obj:`Reader.root_dir` which is the highest-level
            folder of the likelihood code
        data_folder: str
            Sub_folder of :obj:`Reader.dat_dir_main` within which to find
            the spectroscopic data

        """

        # get file name from data.yaml
        fname_scale_cuts = self.data['spectro']['scale_cuts_fourier']

        # construct path to the scale cut file
        path = str(self.root_dir / config_folder / fname_scale_cuts)

        with open(path, 'r') as file:
            GC_sp_scale_cuts = yaml.load(file.read(), Loader=yaml.FullLoader)

        # read the fiducial cosmological parameters
        self.read_GC_spectro()
        h = self.data_spectro_fiducial_cosmo['H0'] / 100.0

        # create the scale cuts dictionary
        redshifts = self.data_dict['GC-Spectro'].keys()
        GC_sp_scale_cuts_h = GC_sp_scale_cuts

        for redshift_index, redshift in enumerate(redshifts):
            k_pk = self.data_dict['GC-Spectro'][f'{redshift}']['k_pk']
            multipoles = (
                [key for key in
                    self.data_dict['GC-Spectro'][f'{redshift}'].keys()
                    if key.startswith('pk')])
            for multipole in multipoles:
                # conversion from h/Mpc to 1/Mpc units (multiply by h)
                (GC_sp_scale_cuts_h['bins'][f'n{redshift_index+1}']
                    [f'n{redshift_index+1}']['multipoles'][int(multipole[2:])]
                    ['k_range'][0]) = [i * h for i in
                                       (GC_sp_scale_cuts['bins']
                                        [f'n{redshift_index+1}']
                                        [f'n{redshift_index+1}']
                                        ['multipoles'][int(multipole[2:])]
                                        ['k_range'][0])]

        self.GC_spectro_scale_cuts = GC_sp_scale_cuts_h

        return

    def read_GC_spectro(self, file_dest='Spectroscopic/data'):
        """Reads in the spectroscopic data.

        Function to read OU-LE3 spectroscopic galaxy clustering files, based
        on location provided to Reader class. Adds contents to the data
        dictionary (:obj:`Reader.data_dict`).

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`self.data_subdirectory` within which to find
            spectroscopic data
        """
        root = self.data['spectro']['root']
        redshifts = self.data['spectro']['redshifts']

        if 'cov_is_num' not in self.data['spectro'].keys():
            self.data['spectro']['cov_is_num'] = False

        if self.data['spectro']['cov_is_num']:
            if 'cov_nsim' not in self.data['spectro'].keys():
                raise Exception('The parameter cov_nsim for spectro data '
                                'must be set when cov_is_num = True')
            if not isinstance(self.data['spectro']['cov_nsim'], int):
                raise TypeError('The parameter cov_nsim for spectro data must '
                                'be set to an integer number when '
                                'cov_is_num = True')
            if self.data['spectro']['cov_nsim'] <= 0:
                raise ValueError('The parameter cov_nsim for spectro data '
                                 'must be strictly positive')

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
                'omch2': ((fid_cosmo_file[1].header['OMEGA_M'] -
                          fid_cosmo_file[1].header['OMEGA_B']) *
                          fid_cosmo_file[1].header['HUBBLE']**2 -
                          0.0006451438915397982),
                'ombh2': fid_cosmo_file[1].header['OMEGA_B'] *
                fid_cosmo_file[1].header['HUBBLE']**2,
                'ns': fid_cosmo_file[1].header['INDEX_N'],
                'sigma8_0': fid_cosmo_file[1].header['SIGMA_8'],
                'w': fid_cosmo_file[1].header['W_STATE'],
                'omkh2': fid_cosmo_file[1].header['OMEGA_K'] *
                fid_cosmo_file[1].header['HUBBLE']**2,
                # OU-LE3 spectro files always with omnuh2 = 0
                'omnuh2': 0.0006451438915397982,
                'Omnu': 0.0014214235118735832}
            # Omega_radiation is ignored here
            fid_cosmo_file.close()

        except ReaderError:
            log = logging.getLogger('CLOE')
            log.critical('There was an error when reading the fiducial '
                         'data from OU-level3 files in read_GC_spectro')

        if self.data['spectro']['Fourier']:
            k_fac = (self.data_spectro_fiducial_cosmo['H0'] / 100.0)
            p_fac = 1.0 / (k_fac ** 3.0)
            cov_fac = p_fac ** 2.0

            for z_label in redshifts:
                cur_it_fname = root.format(z_label)
                cur_full_path = Path(self.dat_dir_main, file_dest,
                                     cur_it_fname)
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

        else:
            for z_label in redshifts:
                cur_it_fname = root.format(z_label)
                cur_full_path = Path(self.dat_dir_main, file_dest,
                                     cur_it_fname)
                fits_file = fits.open(cur_full_path)
                average = fits_file[1].data
                rr = average["SCALE_1DIM"]
                xi0 = average["AVERAGE0"]
                xi2 = average["AVERAGE2"]
                xi4 = average["AVERAGE4"]

                cov = fits_file[2].data["COVARIANCE"]
                cov_r_i = fits_file[2].data["SCALE_1DIM-I"]
                cov_r_j = fits_file[2].data["SCALE_1DIM-J"]
                cov_l_i = fits_file[2].data["MULTIPOLE-I"]
                cov_l_j = fits_file[2].data["MULTIPOLE-J"]

                nr = len(rr)
                cov = np.reshape(cov, newshape=(3 * nr, 3 * nr))
                cov_r_i = np.reshape(cov_r_i, newshape=(3 * nr, 3 * nr))
                cov_r_j = np.reshape(cov_r_j, newshape=(3 * nr, 3 * nr))
                cov_l_i = np.reshape(cov_l_i, newshape=(3 * nr, 3 * nr))
                cov_l_j = np.reshape(cov_l_j, newshape=(3 * nr, 3 * nr))

                GC_spectro_dict['{:s}'.format(z_label)] = {'r_xi': rr,
                                                           'xi0': xi0,
                                                           'xi2': xi2,
                                                           'xi4': xi4,
                                                           'cov': cov,
                                                           'cov_r_i': cov_r_i,
                                                           'cov_r_j': cov_r_j,
                                                           'cov_l_i': cov_l_i,
                                                           'cov_l_j': cov_l_j}

                fits_file.close()

        self.data_dict['GC-Spectro'] = GC_spectro_dict
        return

    def read_GC_spectro_mixing_matrix(self, file_dest='Spectroscopic/data'):
        """Reads in the spectroscopic mixing matrix.

        Function to read the OU-LE3 spectroscopic mixing matrices, based
        on location provided to Reader class. Adds contents to the data
        dictionary (:obj:`Reader.data_dict`).

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`self.data_subdirectory` within which to find
            spectroscopic data
        """
        root = self.data['spectro']['root_mixing_matrix']
        full_path = Path(self.dat_dir_main, file_dest, root)
        fits_file = fits.open(full_path)

        fid_h = self.data_spectro_fiducial_cosmo['H0'] / 100.0
        kin0 = fits_file['BINS_INPUT'].data['kp0'] * fid_h
        kin2 = fits_file['BINS_INPUT'].data['kp2'] * fid_h
        kin4 = fits_file['BINS_INPUT'].data['kp4'] * fid_h
        kout = fits_file['BINS_OUTPUT'].data['k'] * fid_h
        mixing_matrix = fits_file['MIXING_MATRIX'].data

        mixing_matrix_dict = {}
        mixing_matrix_dict['kout'] = kout
        mixing_matrix_dict['kin0'] = kin0
        mixing_matrix_dict['kin2'] = kin2
        mixing_matrix_dict['kin4'] = kin4
        for i in [0, 2, 4]:
            for j in [0, 2, 4]:
                mm = f'W{i}{j}'
                mixing_matrix_dict[mm] = mixing_matrix[mm]

        return mixing_matrix_dict

    def read_CG(self, file_dest='Clusters/'):
        """Read CG

        Function to read OU-LE3 clusters of galaxies files, based
        on location provided to Reader class. Adds contents to the data
        dictionary (Reader.data_dict).

        Parameters
        ----------
        file_dest: str
            Sub-folder of self.data_subdirectory within which to find
            clusters data.
        file_names: str
            General structure of file names. Note: must contain 'z%s' to
            enable iteration over redshifts.
        zstr: list
            List of strings denoting clusters redshift bins.
        """
        file_names_CC = self.data['CG']['file_names_CC']
        file_cov_names_CC = self.data['CG']['file_cov_names_CC']
        file_names_MoR = self.data['CG']['file_names_MoR']
        file_cov_names_MoR = self.data['CG']['file_cov_names_MoR']
        file_names_xi2 = self.data['CG']['file_names_xi2']
        file_cov_names_xi2 = self.data['CG']['file_cov_names_xi2']

        cur_fname = file_names_CC
        full_path = Path(self.dat_dir_main, file_dest, cur_fname)
        cur_cov_fname = file_cov_names_CC
        full_cov_path = Path(self.dat_dir_main, file_dest, cur_cov_fname)
        CG_dict = np.loadtxt(Path(full_path))
        CG_dict_cov = np.loadtxt(Path(full_cov_path))
        self.data_dict['CG_CC'] = CG_dict
        self.data_dict['CG_cov_CC'] = CG_dict_cov

        cur_fname = file_names_MoR
        full_path = Path(self.dat_dir_main, file_dest, cur_fname)
        cur_cov_fname = file_cov_names_MoR
        full_cov_path = Path(self.dat_dir_main, file_dest, cur_cov_fname)
        CG_dict = np.loadtxt(Path(full_path))
        CG_dict_cov = np.loadtxt(Path(full_cov_path))
        self.data_dict['CG_MoR'] = CG_dict
        self.data_dict['CG_cov_MoR'] = CG_dict_cov

        cur_fname = file_names_xi2
        full_path = Path(self.dat_dir_main, file_dest, cur_fname)
        cur_cov_fname = file_cov_names_xi2
        full_cov_path = Path(self.dat_dir_main, file_dest, cur_cov_fname)
        CG_dict = np.load(Path(full_path))
        CG_dict_cov = np.load(Path(full_cov_path))
        self.data_dict['CG_xi2'] = CG_dict
        self.data_dict['CG_cov_xi2'] = CG_dict_cov

        return

    def read_phot_mixing_matrix(self, file_dest='Photometric/data'):
        """Reads in the photometric mixing matrix.

        Function to read the OU-LE3 photometric mixing matrices, based
        on location provided to Reader class. Adds contents to the data
        dictionary (:obj:`Reader.data_dict`).

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`self.data_subdirectory` within which to find
            photometric data
        """
        root = self.data['photo']['root_mixing_matrix']
        full_path = Path(self.dat_dir_main, file_dest, root)
        mixing_matrix = mixing_matrices(full_path)

        return mixing_matrix

    def read_phot(self, file_dest='Photometric/data'):
        """Reads in the photometric data.

        Function to read OU-LE3 photometric galaxy clustering and weak lensing
        files, based on location provided to Reader class. Adds contents to
        the data dictionary the data dictionary (:obj:`Reader.data_dict`).

        Parameters
        ----------
        file_dest: str
            Sub-folder of :obj:`self.data_subdirectory` within which to find
            photometric data
        """

        root_GC = self.data['photo']['root_GC']
        root_WL = self.data['photo']['root_WL']
        root_XC = self.data['photo']['root_XC']
        IA_model = self.data['photo']['IA_model']

        if 'cov_is_num' not in self.data['photo'].keys():
            self.data['photo']['cov_is_num'] = False

        if self.data['photo']['cov_is_num']:
            if 'cov_nsim' not in self.data['photo'].keys():
                raise Exception('The parameter cov_nsim for photo data '
                                'must be set when cov_is_num = True')
            if not isinstance(self.data['photo']['cov_nsim'], int):
                raise TypeError('The parameter cov_nsim for photo data must '
                                'be set to an integer number when '
                                'cov_is_num=True')
            if self.data['photo']['cov_nsim'] <= 0:
                raise ValueError('The parameter cov_nsim for photo data '
                                 'must be positive')

        if self.data['photo']['Fourier']:
            scale_var_str = 'ells'
        else:
            scale_var_str = 'thetas'

        self.numtomo_wl = len(self.nz_dict_WL)
        self.numtomo_gcphot = len(self.nz_dict_GC_Phot)
        self.num_bins_wl = int(self.numtomo_wl * (self.numtomo_wl + 1) / 2)
        self.num_bins_xcphot = self.numtomo_wl * self.numtomo_gcphot
        self.num_bins_gcphot = int(self.numtomo_gcphot *
                                   (self.numtomo_gcphot + 1) / 2)

        GC_phot_dict = {}
        WL_dict = {}
        XC_phot_dict = {}

        full_path = Path(self.dat_dir_main, file_dest)

        if self.data['photo']['photo_data'] == 'standard':

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

            header_GC = GC_file.colnames
            for i in range(len(header_GC)):
                GC_phot_dict[header_GC[i]] = GC_file[header_GC[i]].data

            header_WL = WL_file.colnames
            for i in range(len(header_WL)):
                WL_dict[header_WL[i]] = WL_file[header_WL[i]].data

            header_XC = XC_file.colnames
            for i in range(len(header_XC)):
                XC_phot_dict[header_XC[i]] = XC_file[header_XC[i]].data

            del (GC_file)
            del (WL_file)
            del (XC_file)

        elif self.data['photo']['photo_data'] == 'LE3':

            root_fits = self.data['photo']['root_fits'].format(
                self.numtomo_wl)

            self.loaded_cls = angular_power_spectra(
                path=f'{full_path}/{root_fits}',
                include=None,
                exclude=None,
            )

            for zi in range(self.numtomo_gcphot):
                for zj in range(self.numtomo_wl):
                    XC_phot_dict[f'P{zi + 1}-E{zj + 1}'] = \
                        self.loaded_cls[('P', 'G_E', zi, zj)]['CL'].astype(
                            'float64')

            for zi in range(self.numtomo_wl):
                for zj in range(zi, self.numtomo_wl):
                    WL_dict[f'E{zi + 1}-E{zj + 1}'] = \
                        self.loaded_cls[('G_E', 'G_E', zi, zj)]['CL'].astype(
                            'float64')

            for zi in range(self.numtomo_gcphot):
                for zj in range(zi, self.numtomo_gcphot):
                    GC_phot_dict[f'P{zi + 1}-P{zj + 1}'] = \
                        self.loaded_cls[('P', 'P', zi, zj)]['CL'].astype(
                            'float64')

            WL_dict['ells'] = \
                self.loaded_cls[('G_E', 'G_E', 5, 5)]['L'].astype('float64')
            XC_phot_dict['ells'] = \
                self.loaded_cls[('P', 'G_E', 5, 5)]['L'].astype('float64')
            GC_phot_dict['ells'] = \
                self.loaded_cls[('P', 'P', 5, 5)]['L'].astype('float64')

        else:
            raise ValueError(
                'photo_data must be either "standard" or "LE3"')

        self.num_scales_wl = len(WL_dict[scale_var_str])
        self.num_scales_xcphot = len(XC_phot_dict[scale_var_str])
        self.num_scales_gcphot = len(GC_phot_dict[scale_var_str])

        self.data_dict['WL'] = WL_dict
        self.data_dict['XC-Phot'] = XC_phot_dict
        self.data_dict['GC-Phot'] = GC_phot_dict

        tx2_cov_str = self.data['photo']['cov_3x2pt'].format(self.data[
            'photo']['cov_model'])
        if tx2_cov_str.endswith('npz'):
            tx2_cov = np.load(Path(full_path, tx2_cov_str))['arr_0']
        else:
            tx2_cov = np.load(Path(full_path, tx2_cov_str))

        self.data_dict['3x2pt_cov'] = tx2_cov
        return

    def read_cmbx(self, file_dest='cmbx'):
        """Read Phot

        Function to read CMB lensing files, based on
        location provided to Reader class. Adds contents to
        the data dictionary (Reader.data_dict).

        Parameters
        ----------
        file_dest: str
            Sub-folder of self.data_subdirectory within which to find
            the CMB lensing data.
        """

        root_dir = Path(__file__).resolve().parents[2]
        cmbx_dir = Path(self.dat_dir_main, file_dest)

        if 'cmbx' not in self.data:
            self.data['cmbx'] = {'root_CMBlens': 'Cls_kCMB.dat',
                                 'root_CMBlensxWL': 'Cls_kCMBxWL.dat',
                                 'root_CMBlensxGC': 'Cls_kCMBxGC.dat',
                                 'root_CMBisw': 'Cls_{:s}_ISWxGC.dat',
                                 'ISW_model': 'zNLA',
                                 'cov_7x2pt': 'Cov_7x2pt_WL_GC_CMBX.npy'}
        else:
            defaults = {
                'root_CMBlens': 'Cls_kCMB.dat',
                'root_CMBlensxWL': 'Cls_kCMBxWL.dat',
                'root_CMBlensxGC': 'Cls_kCMBxGC.dat',
                'root_CMBisw': 'Cls_{:s}_ISWxGC.dat',
                'ISW_model': 'zNLA',
                'cov_7x2pt': 'Cov_7x2pt_WL_GC_CMBX.npy'
            }

            self.data.setdefault('cmbx', {}).update(
                {k: v for k, v in defaults.items()
                 if k not in self.data['cmbx']})

        KK_file = ascii.read(
            Path(cmbx_dir, self.data['cmbx']['root_CMBlens']),
            encoding='utf-8',
        )

        KWL_file = ascii.read(
            Path(cmbx_dir, self.data['cmbx']['root_CMBlensxWL']),
            encoding='utf-8',
        )

        KGC_file = ascii.read(
            Path(cmbx_dir, self.data['cmbx']['root_CMBlensxGC']),
            encoding='utf-8',
        )

        ISWxGC_file = ascii.read(
            Path(cmbx_dir, self.data['cmbx']['root_CMBisw'].format(
                self.data['cmbx']['ISW_model'])),
            encoding='utf-8',
        )

        kCMB_dict = {}
        kCMBxWL_dict = {}
        kCMBxGC_dict = {}
        ISWxGC_dict = {}

        for dico, datafile in zip(
                [kCMB_dict, kCMBxWL_dict, kCMBxGC_dict, ISWxGC_dict],
                [KK_file, KWL_file, KGC_file, ISWxGC_file]
        ):
            header = datafile.colnames
            for i in range(len(header)):
                dico[header[i]] = datafile[header[i]].data

        self.data_dict['kCMB'] = kCMB_dict
        self.data_dict['kCMBxWL'] = kCMBxWL_dict
        self.data_dict['kCMBxGC'] = kCMBxGC_dict
        self.data_dict['ISWxGC'] = ISWxGC_dict

        cov_7x2_str = self.data['cmbx']['cov_7x2pt']
        cov_7x2 = np.load(Path(cmbx_dir, cov_7x2_str))
        self.data_dict['7x2pt_cov'] = cov_7x2

        del (KK_file)
        del (KWL_file)
        del (KGC_file)
        del (ISWxGC_file)

        return
