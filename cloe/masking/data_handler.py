# -*- coding: utf-8 -*-
r"""DATA HANLDER MODULE

Module for rearranging data vectors and covariance matrices into final
form, and for obtaining the masking vector
"""

import numpy as np
from cloe.auxiliary.matrix_manipulator import merge_matrices
from warnings import warn
from cloe.data_reader import reader


class Data_handler:
    r"""Reshape data and covariances vectors and define masking vector.

    Build the data vector, the total covariance matrix and the masking
    vector starting from an initialized instance of a Reader object, and from
    a dictionary containing the user's specifications about the probes to be
    included in the likelihood evaluation.
    The data vector, covariance matrix and masking vector have the
    same structure and dimension, as it should be in order to use them
    consistently in the likelihood evaluation.
    They are built as the concatenation of the following probes (in this
    order, from lower indices to higher indices):
     - WL: Weak Lensing
     - XC-Phot: (Weak Lensing) x (Photometric Galaxy Clustering) cross
       correlation
     - GC-Phot: Photometric Galaxy Clustering
     - GC-Spectro: Spectroscopic Galaxy Clustering

    The size of the data vector, covariance matrix and masking vector
    is inferred from the input Reader object.
    """

    def __init__(self, data_dict, cov_dict, obs_dict, data_reader):
        r"""Constructor.

        Constructor of the class Data_handler.
        The data vector, covariance matrix and masking vector are
        built starting from the data_reader, and from the observables
        dictionary.

        Parameters
        ----------
        data_dict: dict
            Dictionary containing the data vectors.
        cov_dict: dict
            Dictionary containing the covariance matrices.
        obs_dict: dict
            Dictionary containing the user-selected probes.
        data_reader: class
            Instance of an already initialized Reader object
        """

        self._data = data_dict
        self._cov = cov_dict
        self._obs = obs_dict

        # size of each individual data vector
        self._wl_size = len(self._data['WL'])
        self._xc_phot_size = len(self._data['XC-Phot'])
        self._gc_phot_size = len(self._data['GC-Phot'])
        self._gc_spectro_size = len(self._data['GC-Spectro'])

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

        self._create_data_vector()
        self._create_cov_matrix()
        self._create_masking_vector(data_reader)

    def get_data_and_masking_vector(self):
        r"""Getter.

        Returns the final unmasked data vector, unmasked
        covariance matrix, and the masking vector.

        Returns
        -------
        self._data_vector: np.ndarray
            Final unmasked data vector
        self._cov_matrix: np.ndarray
            Final unmasked covariance matrix
        self._masking_vector: np.ndarray
            Masking vector
        """
        return (self._data_vector,
                self._cov_matrix,
                self._masking_vector)

    @property
    def use_wl(self):
        r"""Get whether the WL probe should be used"""
        return self._use_wl

    @property
    def use_xc_phot(self):
        r"""Get whether the XC-Phot probe should be used"""
        return self._use_xc_phot

    @property
    def use_gc_phot(self):
        r"""Get whether the GC-Phot probe should be used"""
        return self._use_gc_phot

    @property
    def use_gc_spectro(self):
        r"""Get whether GC-Spectro probe should be used"""
        return self._use_gc_spectro

    @property
    def gc_spectro_size(self):
        r"""Get the size of the GC-Spectro part of the data vector"""
        return self._gc_spectro_size

    def _create_data_vector(self):
        r"""Creates the final unmasked data vector.

        Concatenates data vectors from the four individual probes,
        and assigns the result to the internal attribute self._data_vector.
        """
        data_vector = np.concatenate((self._data['WL'],
                                      self._data['XC-Phot'],
                                      self._data['GC-Phot'],
                                      self._data['GC-Spectro']))

        self._data_vector = data_vector

    def _create_cov_matrix(self):
        r"""Creates the final unmasked covariance matrix.

        Merges the individual covariance matrices for the four individual
        probes into a single matrix (with zero cross-terms),
        and assigns the result to the internal attribute self._cov_matrix.
        """
        cov_matrix = merge_matrices(self._cov['3x2pt'],
                                    self._cov['GC-Spectro'])
        self._cov_matrix = cov_matrix

    def _create_masking_vector(self, data):
        r"""Build the masking vector from the observables specification

        Build a masking vector made of 1's and 0's, used to mask the data and
        theory vectors, starting from the user specifications contained in the
        self._obs dictionary.

        Parameters
        ----------
        data: class
          an instance of the Reader class specifying the structure and
          organization of the data

        Notes
        -----
        The created array contains 1's and 0's and has the same size of the
        data and theory vector, with 1's in correspondence of the data/theory
        elements to be included in the likelihood, and 0's in correspondence
        of the elements that are not to be included.
        The size of the masking vectors is inferred from the input data.
        """

        # define variables for better readability
        zpairs_wl = data.numtomo_wl * (data.numtomo_wl + 1) // 2
        zpairs_xc = data.numtomo_wl * data.numtomo_gcphot
        zpairs_gcphot = data.numtomo_gcphot * \
            (data.numtomo_gcphot + 1) // 2

        if self._use_wl:
            zpair = 0
            ells = data.data_dict['WL']['ells']
            wl_vec = np.zeros((len(ells), zpairs_wl))
            for i in range(1, data.numtomo_wl + 1):
                for j in range(i, data.numtomo_wl + 1):
                    accepted_ells = np.array(
                        self._obs['specifications']['WL']['bins']
                        [f'n{i}'][f'n{j}']['ell_range'])
                    wl_vec[:, zpair] = self._get_masking(ells, accepted_ells)
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
            zpair = 0
            ells = data.data_dict['XC-Phot']['ells']
            xc_phot_vec = np.zeros((len(ells), zpairs_xc))
            for i in range(1, data.numtomo_wl + 1):
                for j in range(1, data.numtomo_gcphot + 1):
                    accepted_ells = np.array(
                        self._obs['specifications']['WL-GCphot']['bins']
                        [f'n{i}'][f'n{j}']['ell_range'])
                    xc_phot_vec[:, zpair] = \
                        self._get_masking(ells, accepted_ells)
                    zpair += 1
            xc_phot_vec = xc_phot_vec.flatten()
        else:
            xc_phot_vec = (
                np.full(self._xc_phot_size, self._use_xc_phot, dtype=int))

        # gc_phot_vec = []
        if self._use_gc_phot:
            zpair = 0
            ells = data.data_dict['GC-Phot']['ells']
            gc_phot_vec = np.zeros((len(ells), zpairs_gcphot))
            for i in range(1, data.numtomo_gcphot + 1):
                for j in range(i, data.numtomo_gcphot + 1):
                    accepted_ells = np.array(
                        self._obs['specifications']['GCphot']['bins']
                        [f'n{i}'][f'n{j}']['ell_range'])
                    gc_phot_vec[:, zpair] = \
                        self._get_masking(ells, accepted_ells)
                    zpair += 1
            gc_phot_vec = gc_phot_vec.flatten()
        else:
            gc_phot_vec = (
                np.full(self._gc_phot_size, self._use_gc_phot, dtype=int))

        gc_spectro_vec = []
        if self._use_gc_spectro:
            redshifts = data.data_dict['GC-Spectro'].keys()
            for redshift_index, redshift in enumerate(redshifts):
                k_pk = data.data_dict['GC-Spectro'][f'{redshift}']['k_pk']
                multipoles = (
                    [key for key in
                     data.data_dict['GC-Spectro'][f'{redshift}'].keys()
                     if key.startswith('pk')])
                for multipole in multipoles:
                    accepted_k_pk = np.array(
                        self._obs['specifications']['GCspectro']['bins']
                        [f'n{redshift_index+1}'][f'n{redshift_index+1}']
                        ['multipoles'][int(multipole[2:])]
                        ['k_range'])
                    gc_spectro_vec = np.concatenate(
                        (gc_spectro_vec,
                         self._get_masking(k_pk, accepted_k_pk)),
                        axis=None)

        else:
            gc_spectro_vec = (
                np.full(
                    self._gc_spectro_size,
                    self._use_gc_spectro,
                    dtype=int))

        self._masking_vector = np.concatenate(
            (wl_vec, xc_phot_vec, gc_phot_vec, gc_spectro_vec),
            axis=None)

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
