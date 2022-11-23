# -*- coding: utf-8 -*-
"""Reader

Contains class to read external data
"""

import numpy as np
from astropy.io import fits, ascii
from pathlib import Path
from scipy import interpolate
from scipy import integrate
from cloe.auxiliary.logger import log_critical


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
        self.luminosity_ratio_interpolator = None

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

    def reader_luminosity_ratio(self, file_dest, file_name):
        """Reader luminosity ratio file

        General routine to read the the luminosity ratio
        used to compute the IA as a function of redshift
        data

        Parameters
        ----------
        file_dest: str
            Sub-folder of Reader.data_subdirectory within which to find
            the luminosity ratio as a function of redshift.
        file_name: str
            Name of the luminosity file

        Return
        ------
        luminosity_ratio_dict: dict
            dictionary containing raw luminosity ratio data
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
        """Compute Nz

        Function to save n(z) dictionaries as attributes of the Reader class
        It saves the interpolator of the raw data.

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

    def compute_luminosity_ratio(self, file_dest='Photometric'):
        """Compute luminosity ratio

        Function to save luminosty ratio dict as attributes of the Reader class
        It saves the interpolator of the raw data.

        Parameters
        ----------
        file_dest: str
            Sub-folder of Reader.data_subdirectory within which to find
            the luminosity ratio data.
        """
        print(self.data)
        file_name_lum = self.data['photo']['luminosity_ratio']
        self.luminosity_ratio = self.reader_luminosity_ratio(
            file_dest, file_name_lum)
        self.luminosity_ratio_interpolator = \
            interpolate.InterpolatedUnivariateSpline(
                self.luminosity_ratio['z'],
                self.luminosity_ratio['luminosity'])

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
            log_critical('There was an error when reading the fiducial '
                         'data from OU-level3 files in read_GC_spectro')

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

        header_GC = GC_file.colnames
        for i in range(len(header_GC)):
            GC_phot_dict[header_GC[i]] = GC_file[header_GC[i]].data

        header_WL = WL_file.colnames
        for i in range(len(header_WL)):
            WL_dict[header_WL[i]] = WL_file[header_WL[i]].data

        header_XC = XC_file.colnames
        for i in range(len(header_XC)):
            XC_phot_dict[header_XC[i]] = XC_file[header_XC[i]].data

        self.numtomo_wl = len(self.nz_dict_WL)
        self.numtomo_gcphot = len(self.nz_dict_GC_Phot)
        self.num_bins_wl = int(self.numtomo_wl * (self.numtomo_wl + 1) / 2)
        self.num_bins_xcphot = self.numtomo_wl * self.numtomo_gcphot
        self.num_bins_gcphot = int(self.numtomo_gcphot *
                                   (self.numtomo_gcphot + 1) / 2)
        self.num_ells_wl = len(WL_dict['ells'])
        self.num_ells_xcphot = len(XC_phot_dict['ells'])
        self.num_ells_gcphot = len(GC_phot_dict['ells'])

        tx2_cov_str = self.data['photo']['cov_3x2pt'].format(self.data[
            'photo']['cov_model'])
        tx2_cov = np.load(Path(full_path, tx2_cov_str))
        new_tx2_cov = self._unpack_3x2pt_cov(tx2_cov)

        self.data_dict['WL'] = WL_dict
        self.data_dict['XC-Phot'] = XC_phot_dict
        self.data_dict['GC-Phot'] = GC_phot_dict
        self.data_dict['3x2pt_cov'] = new_tx2_cov

        del (GC_file)
        del (WL_file)
        del (XC_file)
        return

    def _unpack_3x2pt_cov(self, tx2_cov):
        """Unpack 3x2pt covariance matrix

        Unpacks the full 3x2pt covariance matrix and reshapes it in order to
        have the different probes as the outermost variable, the
        tomographic bin combination as the intermediate variable, and the
        multipole as innermost variable.

        Parameters
        ----------
        tx2_cov: numpy.ndarray
            2-dimensional array containing the 3x2pt covariance matrix.

        Returns
        -------
        new_tx2_cov: numpy.ndarray
            2-dimensional array containing the reshaped 3x2pt covariance
            matrix.
        """
        nbins_wl = self.num_bins_wl
        nbins_xc = self.num_bins_xcphot
        nbins_gc = self.num_bins_gcphot
        nbins_tot = nbins_wl + nbins_xc + nbins_gc

        wl_side = nbins_wl * self.num_ells_wl
        xc_side = nbins_xc * self.num_ells_xcphot
        gc_side = nbins_gc * self.num_ells_gcphot

        new_tx2_cov = np.zeros((tx2_cov.shape[0], tx2_cov.shape[1]))

        for i in range(nbins_wl):
            cur_i = int(i * self.num_ells_wl)
            next_i = int((i + 1) * self.num_ells_wl)
            for j in range(nbins_wl):
                cur_j = int(j * self.num_ells_wl)
                next_j = int((j + 1) * self.num_ells_wl)
                new_tx2_cov[cur_i:next_i, cur_j:next_j] = \
                    tx2_cov[i::nbins_tot, j::nbins_tot]
            for j in range(nbins_xc):
                cur_j = int(wl_side + j * self.num_ells_xcphot)
                next_j = int(wl_side + (j + 1) * self.num_ells_xcphot)
                new_tx2_cov[cur_i:next_i, cur_j:next_j] = \
                    tx2_cov[i::nbins_tot, (nbins_wl + j)::nbins_tot]
            for j in range(nbins_gc):
                cur_j = int(wl_side + xc_side + j * self.num_ells_gcphot)
                next_j = int(wl_side + xc_side +
                             (j + 1) * self.num_ells_gcphot)
                new_tx2_cov[cur_i:next_i, cur_j:next_j] = \
                    tx2_cov[i::nbins_tot,
                            (nbins_wl + nbins_xc + j)::nbins_tot]

        for i in range(nbins_xc):
            cur_i = int(wl_side + i * self.num_ells_xcphot)
            next_i = int(wl_side + (i + 1) * self.num_ells_xcphot)
            for j in range(nbins_xc):
                cur_j = int(wl_side + j * self.num_ells_xcphot)
                next_j = int(wl_side + (j + 1) * self.num_ells_xcphot)
                new_tx2_cov[cur_i:next_i, cur_j:next_j] = \
                    tx2_cov[(nbins_wl + i)::nbins_tot,
                            (nbins_wl + j)::nbins_tot]
            for j in range(nbins_gc):
                cur_j = int(wl_side + xc_side + j * self.num_ells_gcphot)
                next_j = int(wl_side + xc_side +
                             (j + 1) * self.num_ells_gcphot)
                new_tx2_cov[cur_i:next_i, cur_j:next_j] = \
                    tx2_cov[(nbins_wl + i)::nbins_tot,
                            (nbins_wl + nbins_xc + j)::nbins_tot]

        for i in range(nbins_gc):
            cur_i = int(wl_side + xc_side + i * self.num_ells_gcphot)
            next_i = int(wl_side + xc_side + (i + 1) * self.num_ells_gcphot)
            for j in range(nbins_gc):
                cur_j = int(wl_side + xc_side + j * self.num_ells_gcphot)
                next_j = int(wl_side + xc_side +
                             (j + 1) * self.num_ells_gcphot)
                new_tx2_cov[cur_i:next_i, cur_j:next_j] = \
                    tx2_cov[(nbins_wl + nbins_xc + i)::nbins_tot,
                            (nbins_wl + nbins_xc + j)::nbins_tot]

        new_tx2_cov[wl_side:(wl_side + xc_side), :wl_side] = \
            new_tx2_cov[:wl_side, wl_side:(wl_side + xc_side)].T
        new_tx2_cov[(wl_side + xc_side):(wl_side + xc_side + gc_side),
                    :wl_side] = \
            new_tx2_cov[:wl_side,
                        (wl_side + xc_side):(wl_side + xc_side + gc_side)].T
        new_tx2_cov[(wl_side + xc_side):(wl_side + xc_side + gc_side),
                    wl_side:(wl_side + xc_side)] = \
            new_tx2_cov[wl_side:(wl_side + xc_side),
                        (wl_side + xc_side):(wl_side + xc_side + gc_side)].T

        return new_tx2_cov
