# -*- coding: utf-8 -*-
r"""DATA HANDLER MODULE

Module for rearranging data vectors and covariance matrices into
the needed final form and for obtaining the masking vector.
"""

import numpy as np
from cloe.auxiliary.matrix_manipulator import merge_matrices
from warnings import warn
from cloe.data_reader import reader


class Data_handler:
    r"""Reshapes data vectors and covariances and defines masking vectors.

    Build data vectors, covariance matrices and masking vectors
    starting from an initialized instance of a Reader object, and from
    a dictionary containing the user's specifications about the probes to be
    included in the likelihood evaluation. The data vector, covariance matrix
    and masking vector are built for the photometric and spectroscopic probes
    separately and for the combined case, where they are built as the
    concatenation of the following probes (in this order, from lower indices to
    higher indices):

     - WL: Weak Lensing

     - XCphot: (Weak Lensing) x (Photometric Galaxy Clustering) cross
       correlation

     - GCphot: Photometric Galaxy Clustering

     - GCspectro: Spectroscopic Galaxy Clustering

    The size of the data vector, covariance matrix and masking vector
    is inferred from the input :obj:`Reader` object.
    """

    def __init__(self, data_dict, cov_dict, obs_dict, data_reader):
        r"""Initialises the class.

        Constructor of the class :obj:`Data_handler`.
        The data vectors, covariance matrices and masking vectors are
        built starting from the data_reader, and from the observables
        dictionary.

        Parameters
        ----------
        data_dict: dict
            Dictionary containing the data vectors
        cov_dict: dict
            Dictionary containing the covariance matrices
        obs_dict: dict
            Dictionary containing the user-selected probes
        data_reader: :obj:`Reader`
            Instance of an already initialized :obj:`Reader` object
        """

        self._obs = obs_dict

        # boolean representing whether a given probe has been selected
        self._use_wl = self._obs['selection']['WL']['WL']
        self._use_gc_phot = self._obs['selection']['GCphot']['GCphot']
        self._use_gc_spectro = (
            self._obs['selection']['GCspectro']['GCspectro'])
        try:
            self._use_xc_phot = self._obs['selection']['WL']['GCphot']
        except KeyError:
            try:
                self._use_xc_phot = self._obs['selection']['GCphot']['WL']
                warn('WL X GCPhot cross-correlation selection has not been '
                     'found in WL dictionary. Using the value defined in '
                     'GCPhot dictionary.')
            except KeyError:
                self._use_xc_phot = False
                warn('WL X GCPhot cross-correlation selection has not been '
                     'found in either WL and GCPhot dictionary. Setting it '
                     'to False.')
        self._use_phot = (self._use_wl or self._use_gc_phot or
                          self._use_xc_phot)

        self._data = data_dict
        self._cov = cov_dict

        # Check whether any cross-correlation with kCMB has been asked. Set to
        # false when key is missing
        try:
            self._use_kCMB = self._obs['selection']['CMBlens']['CMBlens']
        except KeyError:
            self._use_kCMB = False
        try:
            self._use_kCMB_wl = self._obs['selection']['CMBlens']['WL']
        except KeyError:
            self._use_kCMB_wl = False
        try:
            self._use_kCMB_gc = self._obs['selection']['CMBlens']['GCphot']
        except KeyError:
            self._use_kCMB_gc = False
        try:
            self._use_iswxgc = self._obs['selection']['CMBisw']['GCphot']
        except KeyError:
            self._use_iswxgc = False

        self._use_cmbx = (self._use_kCMB or self._use_kCMB_wl or
                          self._use_kCMB_gc or self._use_iswxgc)

        if self._use_gc_spectro:
            self._gc_spectro_size = len(self._data['GC-Spectro'])
            self._create_data_vector_spectro()
            self._create_cov_matrix_spectro()
            self._create_masking_vector_spectro(data_reader)
        if self._use_phot:
            self._wl_size = len(self._data['WL'])
            self._xc_phot_size = len(self._data['XC-Phot'])
            self._gc_phot_size = len(self._data['GC-Phot'])
            if self._use_cmbx:
                self._kCMB_size = len(self._data['kCMB'])
                self._kCMBxWL_size = len(self._data['kCMBxWL'])
                self._kCMBxGC_size = len(self._data['kCMBxGC'])
                self._iswxgc_size = len(self._data['ISWxGC'])
            self._create_data_vector_phot()
            self._create_cov_matrix_phot()
            self._create_masking_vector_phot(data_reader)

        if self._use_phot and self._use_gc_spectro:
            self._create_masking_vector_full(data_reader)

    def get_data_and_masking_vector(self):
        r"""Gets data and masking vector.

        Returns the final unmasked data vector, unmasked
        covariance matrix, and the masking vector
        for the combination of photometric and spectroscopic probes.

        Returns
        -------
        Data vector, covariance, masking vector: numpy.ndarray
            Final unmasked data vector, Final unmasked covariance matrix,
            Masking vector
        """
        return (self._data_vector,
                self._cov_matrix,
                self._masking_vector)

    def get_data_and_masking_vector_phot(self):
        r"""Gets photometric data and masking vector.

        Returns the final unmasked data vector, unmasked
        covariance matrix, and the masking vector
        for the photometric probes.

        Returns
        -------
        Data vector, covariance, masking vector: numpy.ndarray
            Final unmasked data vector, Final unmasked covariance matrix,
            Masking vector
        """
        return (self._data_vector_phot,
                self._cov_matrix_phot,
                self._masking_vector_phot)

    def get_data_and_masking_vector_spectro(self):
        r"""Gets spectroscopic data and masking vector.

        Returns the final unmasked data vector, unmasked
        covariance matrix, and the masking vector
        for the spectroscopic probes.

        Returns
        -------
        Data vector, covariance, masking vector numpy.ndarray
            Final unmasked data vector, Final unmasked covariance matrix,
            Masking vector
        """
        return (self._data_vector_spectro,
                self._cov_matrix_spectro,
                self._masking_vector_spectro)

    @property
    def use_wl(self):
        r"""Gets whether the WL probe should be used."""
        return self._use_wl

    @property
    def use_xc_phot(self):
        r"""Gets whether the XCphot probe should be used."""
        return self._use_xc_phot

    @property
    def use_gc_phot(self):
        r"""Gets whether the protometric probe should be used."""
        return self._use_gc_phot

    @property
    def use_iswxgc(self):
        r"""Get whether the GC-Phot probe should be used"""
        return self._use_iswxgc

    @property
    def use_gc_spectro(self):
        r"""Gets whether spectroscopic probe should be used."""
        return self._use_gc_spectro

    @property
    def use_kCMB(self):
        r"""Get whether the kCMB probe should be used"""
        return self._use_kCMB

    @property
    def use_kCMB_wl(self):
        r"""Get whether the kCMBxWL probe should be used"""
        return self._use_kCMB_wl

    @property
    def use_kCMB_gc(self):
        r"""Get whether the kCMBxGC probe should be used"""
        return self._use_kCMB_gc

    @property
    def gc_spectro_size(self):
        r"""Gets the size of the spectroscopic part of the data vector."""
        return self._gc_spectro_size

    def _create_data_vector(self):
        r"""Creates the final unmasked data vector.

        Concatenates data vectors from the four individual probes,
        and assigns the result to the internal attribute
        :obj:`self._data_vector`.
        """
        data_vector = np.concatenate((self._data['WL'],
                                      self._data['XC-Phot'],
                                      self._data['GC-Phot'],
                                      self._data['GC-Spectro'],
                                      ))
        self._data_vector = data_vector

    def _create_data_vector_phot(self):
        r"""Creates the final unmasked photometric data vector.

        Concatenates data vectors from the three photometric probes,
        and assigns the result to the internal attribute
        :obj:`self._data_vector_phot`.
        """
        data_vector = np.concatenate((self._data['WL'],
                                      self._data['XC-Phot'],
                                      self._data['GC-Phot']))

        if self._use_cmbx:
            data_vector_xcmb = np.concatenate((self._data['kCMB'],
                                               self._data['kCMBxWL'],
                                               self._data['kCMBxGC'],
                                               self._data['ISWxGC']))
            data_vector = np.concatenate((data_vector, data_vector_xcmb))

        self._data_vector_phot = data_vector

    def _create_data_vector_spectro(self):
        r"""Creates the final unmasked spectroscopic data vector.

        Assigns the spectroscopic data vector to the
        internal attribute :obj:`self._data_vector_spectro`.
        """

        self._data_vector_spectro = self._data['GC-Spectro']

    def _create_cov_matrix(self):
        r"""Creates the final unmasked covariance matrix.

        Merges the individual covariance matrices for the four individual
        probes into a single matrix (with zero cross-terms),
        and assigns the result to the internal attribute
        :obj:`self._cov_matrix`.
        """
        cov_matrix = merge_matrices(self._cov['3x2pt'],
                                    self._cov['GC-Spectro'])
        self._cov_matrix = cov_matrix

    def _create_cov_matrix_phot(self):
        r"""Creates the final unmasked photometric covariance matrix.

        Assigns the 3x2pt covariance to the
        internal attribute :obj:`self._cov_matrix_phot`.

        if self.use_cmbx:
            We use the 7x2pt covariance matrix generated using the
            make_CMBX_covmat.py script and then add the former 3x2pt
            covariance matrix at the right position, so that the value
            of the likelihood is not affected for Euclid's probes only.

        """
        self._cov_matrix_phot = self._cov['3x2pt']

        if self._use_cmbx:
            self._cov_matrix_phot = self._cov['7x2pt']
            self._cov_matrix_phot[0:len(self._cov['3x2pt']), 0:len(
                self._cov['3x2pt'])] = self._cov['3x2pt']

    def _create_cov_matrix_spectro(self):
        r"""Creates the final unmasked spectroscopic covariance matrix.

        Assigns the GC Spectro covariance
        to the internal attribute :obj:`self._cov_matrix_spectro`.
        """
        self._cov_matrix_spectro = self._cov['GC-Spectro']

    def _create_masking_vector_full(self, data):
        r"""Builds the masking vector from the observables specification

        Builds a masking vector made of 1's and 0's, used to mask the data and
        theory vectors, starting from the user specifications contained in the
        :obj:`self._obs dictionary`.

        Parameters
        ----------
        data: class
          An instance of the :obj:`Reader class` specifying the structure and
          organization of the data

        Notes
        -----
        The created array contains 1's and 0's and has the same size of the
        data and theory vector, with 1's in correspondence of the data/theory
        elements to be included in the likelihood, and 0's in correspondence
        of the elements that are not to be included.
        The size of the masking vectors is inferred from the input data.
        """
        wl_vec = []
        # define variables for better readability
        zpairs_wl = data.numtomo_wl * (data.numtomo_wl + 1) // 2
        zpairs_xc = data.numtomo_wl * data.numtomo_gcphot
        zpairs_gcphot = data.numtomo_gcphot * \
            (data.numtomo_gcphot + 1) // 2

        if self._use_wl:
            stat_wl = self._obs['specifications']['WL']['statistics']
            if stat_wl in ('angular_power_spectrum', 'pseudo_cl'):
                scale_var = 'ells'
                scale_range_var = 'ell_range'
                num_repetitions_wl = 1
            elif stat_wl == 'angular_correlation_function':
                scale_var = 'thetas'
                scale_range_var = 'theta_range'
                num_repetitions_wl = 2
            zpair = 0
            scales = data.data_dict['WL'][scale_var]
            wl_vec = np.zeros((len(scales), zpairs_wl))
            for i in range(1, data.numtomo_wl + 1):
                for j in range(i, data.numtomo_wl + 1):
                    accepted_scales = np.array(
                        self._obs['specifications']['WL'][stat_wl]['bins']
                        [f'n{i}'][f'n{j}'][scale_range_var])
                    wl_vec[:, zpair] = self._get_masking(scales,
                                                         accepted_scales)
                    zpair += 1
            wl_vec = wl_vec.flatten()
            # the default for np.ndarray.flatten()
            # is leftmost axis as outermost for loop
            # in this and in the following, flatten(),
            # or equivalently, flatten(order='C') will flatten the array
            # using the first axis of wl_vec - i.e.,
            # the one on the multipoles - as the outermost for loop.
            # Setting flatten(order='F') will flatten the array
            # using the second axis
            # - i.e., the one on the redshift pairs -
            # as the outermost for loop.
        else:
            wl_vec = np.full(self._wl_size, self._use_wl, dtype=int)

        if self._use_xc_phot:
            stat_xc = self._obs['specifications']['WL-GCphot']['statistics']
            if stat_xc in ('angular_power_spectrum', 'pseudo_cl'):
                scale_var = 'ells'
                scale_range_var = 'ell_range'
            elif stat_xc == 'angular_correlation_function':
                scale_var = 'thetas'
                scale_range_var = 'theta_range'
            zpair = 0
            scales = data.data_dict['XC-Phot'][scale_var]
            xc_phot_vec = np.zeros((len(scales), zpairs_xc))
            for i in range(1, data.numtomo_wl + 1):
                for j in range(1, data.numtomo_gcphot + 1):
                    accepted_scales = np.array(
                        self._obs['specifications']['WL-GCphot'][stat_xc]
                        ['bins'][f'n{i}'][f'n{j}'][scale_range_var])
                    xc_phot_vec[:, zpair] = \
                        self._get_masking(scales, accepted_scales)
                    zpair += 1
            xc_phot_vec = xc_phot_vec.flatten()
        else:
            xc_phot_vec = (
                np.full(self._xc_phot_size, self._use_xc_phot, dtype=int))

        if self._use_gc_phot:
            stat_gcphot = self._obs['specifications']['GCphot']['statistics']
            if stat_gcphot in ('angular_power_spectrum', 'pseudo_cl'):
                scale_var = 'ells'
                scale_range_var = 'ell_range'
            elif stat_gcphot == 'angular_correlation_function':
                scale_var = 'thetas'
                scale_range_var = 'theta_range'
            zpair = 0
            scales = data.data_dict['GC-Phot'][scale_var]
            gc_phot_vec = np.zeros((len(scales), zpairs_gcphot))
            for i in range(1, data.numtomo_gcphot + 1):
                for j in range(i, data.numtomo_gcphot + 1):
                    accepted_scales = np.array(
                        self._obs['specifications']['GCphot'][stat_gcphot]
                        ['bins'][f'n{i}'][f'n{j}'][scale_range_var])
                    gc_phot_vec[:, zpair] = \
                        self._get_masking(scales, accepted_scales)
                    zpair += 1
            gc_phot_vec = gc_phot_vec.flatten()
        else:
            gc_phot_vec = (
                np.full(self._gc_phot_size, self._use_gc_phot, dtype=int))

        gc_spectro_vec = []
        redshifts = data.data_dict['GC-Spectro'].keys()

        if self._obs['specifications']['GCspectro']['statistics'] in \
                ('multipole_power_spectrum',
                 'convolved_multipole_power_spectrum'):
            data.read_GC_spectro_scale_cuts()
            for redshift_index, redshift in enumerate(redshifts):
                k_pk = data.data_dict['GC-Spectro'][f'{redshift}']['k_pk']
                multipoles = (
                    [key for key in
                     data.data_dict['GC-Spectro'][f'{redshift}'].keys()
                     if key.startswith('pk')])
                for multipole in multipoles:
                    accepted_k_pk = np.array(
                        data.GC_spectro_scale_cuts['bins']
                        [f'n{redshift_index+1}'][f'n{redshift_index+1}']
                        ['multipoles'][int(multipole[2:])]
                        ['k_range'])
                    gc_spectro_vec = np.concatenate(
                        (gc_spectro_vec,
                         self._get_masking(k_pk, accepted_k_pk)), axis=None)

        elif self._obs['specifications']['GCspectro']['statistics'] == \
                'multipole_correlation_function':
            for redshift_index, redshift in enumerate(redshifts):
                r_xi = data.data_dict['GC-Spectro'][f'{redshift}']['r_xi']
                multipoles = (
                    [key for key in
                     data.data_dict['GC-Spectro'][f'{redshift}'].keys()
                     if key.startswith('xi')])
                for multipole in multipoles:
                    accepted_r_xi = np.array(
                        self._obs['specifications']['GCspectro']
                        ['multipole_correlation_function']['bins']
                        [f'n{redshift_index+1}'][f'n{redshift_index+1}']
                        ['multipoles'][int(multipole[2:])]
                        ['r_range'])
                    gc_spectro_vec = np.concatenate(
                        (gc_spectro_vec,
                         self._get_masking(r_xi, accepted_r_xi)), axis=None)

        self._masking_vector = np.concatenate(
            (wl_vec, xc_phot_vec, gc_phot_vec, gc_spectro_vec), axis=None)

    def _create_masking_vector_phot(self, data):
        r"""Builds the photometric masking vector.

        Builds a masking vector made of 1's and 0's, used to mask the data and
        theory vectors, starting from the user specifications contained in the
        :obj:`self._obs dictionary`.

        Parameters
        ----------
        data: class
          Instance of the Reader class specifying the structure and
          organization of the data

        Notes
        -----
        The created array contains 1's and 0's and has the same size of the
        data and theory vector, with 1's in correspondence of the data/theory
        elements to be included in the likelihood, and 0's in correspondence
        of the elements that are not to be included.
        The size of the masking vectors is inferred from the input data.
        """
        wl_vec = []
        # define variables for better readability
        zpairs_wl = data.numtomo_wl * (data.numtomo_wl + 1) // 2
        zpairs_xc = data.numtomo_wl * data.numtomo_gcphot
        zpairs_gcphot = data.numtomo_gcphot * \
            (data.numtomo_gcphot + 1) // 2

        if self._use_wl:
            stat_wl = self._obs['specifications']['WL']['statistics']
            if stat_wl in ('angular_power_spectrum', 'pseudo_cl'):
                scale_var = 'ells'
                scale_range_var = 'ell_range'
                num_repetitions_wl = 1
            elif stat_wl == 'angular_correlation_function':
                scale_var = 'thetas'
                scale_range_var = 'theta_range'
                num_repetitions_wl = 2
            zpair = 0
            scales = data.data_dict['WL'][scale_var]
            wl_vec = np.zeros((len(scales), num_repetitions_wl * zpairs_wl))
            for i in range(1, data.numtomo_wl + 1):
                for j in range(i, data.numtomo_wl + 1):
                    accepted_scales = np.array(
                        self._obs['specifications']['WL'][stat_wl]['bins']
                        [f'n{i}'][f'n{j}'][scale_range_var])
                    wl_vec[:, zpair] = self._get_masking(scales,
                                                         accepted_scales)
                    zpair += 1
            wl_vec = wl_vec.flatten()
            # the default for np.ndarray.flatten()
            # is leftmost axis as outermost for loop
            # in this and in the following, flatten(),
            # or equivalently, flatten(order='C') will flatten the array
            # using the first axis of wl_vec - i.e.,
            # the one on the multipoles - as the outermost for loop.
            # Setting flatten(order='F') will flatten the array
            # using the second axis
            # - i.e., the one on the redshift pairs -
            # as the outermost for loop.
        else:
            wl_vec = np.full(self._wl_size, self._use_wl, dtype=int)

        if self._use_xc_phot:
            stat_xc = self._obs['specifications']['WL-GCphot']['statistics']
            if stat_xc in ('angular_power_spectrum', 'pseudo_cl'):
                scale_var = 'ells'
                scale_range_var = 'ell_range'
            elif stat_xc == 'angular_correlation_function':
                scale_var = 'thetas'
                scale_range_var = 'theta_range'
            zpair = 0
            scales = data.data_dict['XC-Phot'][scale_var]
            xc_phot_vec = np.zeros((len(scales), zpairs_xc))
            for i in range(1, data.numtomo_wl + 1):
                for j in range(1, data.numtomo_gcphot + 1):
                    accepted_scales = np.array(
                        self._obs['specifications']['WL-GCphot'][stat_xc]
                        ['bins'][f'n{i}'][f'n{j}'][scale_range_var])
                    xc_phot_vec[:, zpair] = \
                        self._get_masking(scales, accepted_scales)
                    zpair += 1
            xc_phot_vec = xc_phot_vec.flatten()
        else:
            xc_phot_vec = (
                np.full(self._xc_phot_size, self._use_xc_phot, dtype=int))

        if self._use_gc_phot:
            stat_gcphot = self._obs['specifications']['GCphot']['statistics']
            if stat_gcphot in ('angular_power_spectrum', 'pseudo_cl'):
                scale_var = 'ells'
                scale_range_var = 'ell_range'
            elif stat_gcphot == 'angular_correlation_function':
                scale_var = 'thetas'
                scale_range_var = 'theta_range'
            zpair = 0
            scales = data.data_dict['GC-Phot'][scale_var]
            gc_phot_vec = np.zeros((len(scales), zpairs_gcphot))
            for i in range(1, data.numtomo_gcphot + 1):
                for j in range(i, data.numtomo_gcphot + 1):
                    accepted_scales = np.array(
                        self._obs['specifications']['GCphot'][stat_gcphot]
                        ['bins'][f'n{i}'][f'n{j}'][scale_range_var])
                    gc_phot_vec[:, zpair] = \
                        self._get_masking(scales, accepted_scales)
                    zpair += 1
            gc_phot_vec = gc_phot_vec.flatten()
        else:
            gc_phot_vec = (
                np.full(self._gc_phot_size, self._use_gc_phot, dtype=int))

        self._masking_vector_phot = np.concatenate(
            (wl_vec, xc_phot_vec, gc_phot_vec),
            axis=None)

        if self._use_cmbx:
            kCMB_vec = []
            if self._use_kCMB:
                ells = data.data_dict['kCMB']['ells']
                accepted_ells = np.array(
                    self._obs['specifications']['CMBlens']['ell_range'])
                kCMB_vec = self._get_masking(ells, accepted_ells)
            else:
                kCMB_vec = np.full(self._kCMB_size, self._use_kCMB, dtype=int)

            if self._use_kCMB_wl:
                ells = data.data_dict['kCMBxWL']['ells']
                kCMBxWL_vec = np.zeros((len(ells), data.numtomo_wl))
                for i in range(1, data.numtomo_wl + 1):
                    accepted_ells = np.array(
                            self._obs['specifications']['CMBlens-WL']['bins']
                            [f'n{i}']['ell_range'])
                    kCMBxWL_vec[:, i -
                                1] = self._get_masking(ells, accepted_ells)
                kCMBxWL_vec = kCMBxWL_vec.flatten()
            else:
                kCMBxWL_vec = np.full(
                    self._kCMBxWL_size, self._use_kCMB_wl, dtype=int)

            if self._use_kCMB_gc:
                ells = data.data_dict['kCMBxGC']['ells']
                kCMBxGC_vec = np.zeros((len(ells), data.numtomo_gcphot))
                for i in range(1, data.numtomo_gcphot + 1):
                    accepted_ells = np.array(
                            self._obs['specifications']['CMBlens-GCphot']
                            ['bins'][f'n{i}']['ell_range'])
                    kCMBxGC_vec[:, i -
                                1] = self._get_masking(ells, accepted_ells)
                kCMBxGC_vec = kCMBxGC_vec.flatten()
            else:
                kCMBxGC_vec = np.full(
                    self._kCMBxGC_size, self._use_kCMB_gc, dtype=int)

            if self._use_iswxgc:
                ells = data.data_dict['ISWxGC']['ells']
                iswxgc_vec = np.zeros((len(ells), data.numtomo_gcphot))
                for i in range(1, data.numtomo_gcphot + 1):
                    accepted_ells = np.array(
                        self._obs['specifications']['ISW-GCphot']['bins']
                        [f'n{i}']['ell_range'])
                    iswxgc_vec[:, i -
                               1] = self._get_masking(ells, accepted_ells)
                iswxgc_vec = iswxgc_vec.flatten()
            else:
                iswxgc_vec = (
                    np.full(self._iswxgc_size, self._use_iswxgc, dtype=int))

            self._masking_vector_phot = np.concatenate(
                (wl_vec,
                 xc_phot_vec,
                 gc_phot_vec,
                 kCMB_vec,
                 kCMBxWL_vec,
                 kCMBxGC_vec,
                 iswxgc_vec),
                axis=None)

    def _create_masking_vector_spectro(self, data):
        r"""Builds the spectroscopic masking vector.

        Builds a masking vector made of 1's and 0's, used to mask the data and
        theory vectors, starting from the user specifications contained in the
        :obj:`self._obs dictionary`.

        Parameters
        ----------
        data: class
          Instance of the Reader class specifying the structure and
          organization of the data

        Notes
        -----
        The created array contains 1's and 0's and has the same size of the
        data and theory vector, with 1's in correspondence of the data/theory
        elements to be included in the likelihood, and 0's in correspondence
        of the elements that are not to be included.
        The size of the masking vectors is inferred from the input data.
        """
        gc_spectro_vec = []
        redshifts = data.data_dict['GC-Spectro'].keys()

        if self._obs['specifications']['GCspectro']['statistics'] in \
                ('multipole_power_spectrum',
                 'convolved_multipole_power_spectrum'):
            data.read_GC_spectro_scale_cuts()
            for redshift_index, redshift in enumerate(redshifts):
                k_pk = data.data_dict['GC-Spectro'][f'{redshift}']['k_pk']
                multipoles = (
                    [key for key in
                     data.data_dict['GC-Spectro'][f'{redshift}'].keys()
                     if key.startswith('pk')])
                for multipole in multipoles:
                    accepted_k_pk = np.array(
                        data.GC_spectro_scale_cuts['bins']
                        [f'n{redshift_index+1}'][f'n{redshift_index+1}']
                        ['multipoles'][int(multipole[2:])]
                        ['k_range'])
                    gc_spectro_vec = np.concatenate(
                        (gc_spectro_vec,
                         self._get_masking(k_pk, accepted_k_pk)), axis=None)

        elif self._obs['specifications']['GCspectro']['statistics'] == \
                'multipole_correlation_function':
            for redshift_index, redshift in enumerate(redshifts):
                r_xi = data.data_dict['GC-Spectro'][f'{redshift}']['r_xi']
                multipoles = (
                    [key for key in
                     data.data_dict['GC-Spectro'][f'{redshift}'].keys()
                     if key.startswith('xi')])
                for multipole in multipoles:
                    accepted_r_xi = np.array(
                        self._obs['specifications']['GCspectro']
                        ['multipole_correlation_function']['bins']
                        [f'n{redshift_index+1}'][f'n{redshift_index+1}']
                        ['multipoles'][int(multipole[2:])]
                        ['r_range'])
                    gc_spectro_vec = np.concatenate(
                        (gc_spectro_vec,
                         self._get_masking(r_xi, accepted_r_xi)), axis=None)

        self._masking_vector_spectro = gc_spectro_vec

    def _get_masking(self, arr, acceptance_intervals):
        r"""Get a 1/0 mask for arr elements contained in acceptance_intervals

        Get an array of 1's and 0's indicating which elements of the input
        array are contained in the specified acceptance intervals

        Parameters
        ----------
        arr: numpy.ndarray
          The array of values for which the mask is to be evaluated
        acceptance_intervals: ndarray
          An array of 2-element arrays, each defining an interval of values
          to be kept in arr

        Returns
        -------
        numpy.ndarray:
          An array of 0's and 1's of the same size of the input arr, with 1's
          in correspondence of the arr elements contained in the acceptance
          intervals, and 0's in correspondence of the arr elements not
          contained in the acceptance intervals.
        """
        return np.any((arr[:, None] >= acceptance_intervals[:, 0]) &
                      (arr[:, None] <= acceptance_intervals[:, 1]),
                      axis=1).astype(int)
