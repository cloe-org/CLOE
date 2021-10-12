# -*- coding: utf-8 -*-
r"""DATA HANLDER MODULE

Module for rearranging data vectors and covariance matrices into final
form, and for obtaining the masking vector
"""

import numpy as np
from likelihood.auxiliary.matrix_manipulator import merge_matrices
from warnings import warn


class Data_handler:
    r"""Reshape data and covariances vectors and define masking vector

    This class interprets the content of the masking vector to tell whether a
    given probe is enabled.
    It assumes that the masking vector is obtained as the concatenation of
    the following probes (from lower to higher array indices)
    - WL: Photometric Weak Lensing
    - XC-Phot: (Weak Lensing) x (Photometric Galaxy Clustering) cross
      correlation
    - GC-Phot: Photometric Galaxy Clustering
    - GC-Spectro: Spectroscopic Galaxy Clustering

    For the fiducial benchmark, each probe corresponds to:
    -  WL: 1100 elements (55 bins combination X 20 ells),
           from index 0 to index 1100-1
    -  XC-Phot: 2000 elements (100 bins combination X 20 ells),
           from index 1100 to index 3100-1
    -  GC-Phot: 1100 elements (55 bins combination X 20 ells),
           from index 3100 to index 4200-1
    -  GC-Spectro: 1500 elements (4 redshifts X 3 multipoles X 125 scales),
           from index 4200 to index 5700-1
    """

    def __init__(self, data_dict, cov_dict, obs_dict):
        r"""Constructor.

        Constructor of the class Data_handler. Data vectors and covariance
        matrices are rearranged into their final format, and the masking vector
        is defined based on the size of the data vectors and on the choice
        of observables selected by the user.

        Parameters
        ----------
        data_dict: dict
            Dictionary containing the data vectors.
        cov_dict: dict
            Dictionary containing the covariance matrices.
        obs_dict: dict
            Dictionary containing the user-selected probes.
        """

        self._data = data_dict
        self._cov = cov_dict
        self._obs = obs_dict

        # size of each individual data vector
        self._wl_size = len(self._data['WL'])
        self._xc_phot_size = len(self._data['XC-Phot'])
        self._gc_phot_size = len(self._data['GC-Phot'])
        self._gc_spectro_size = len(self._data['GC-Spectro'])

        self._tot_size = (
            self._wl_size +
            self._xc_phot_size +
            self._gc_phot_size +
            self._gc_spectro_size
        )

        # indices representing the first entry of each probe
        self._wl_start = 0
        self._xc_phot_start = self._wl_start + self._wl_size
        self._gc_phot_start = self._xc_phot_start + self._xc_phot_size
        self._gc_spectro_start = self._gc_phot_start + self._gc_phot_size

        # boolean representing whether a given probe has been selected
        self._use_wl = self._obs['WL']['WL']
        self._use_gc_phot = self._obs['GCphot']['GCphot']
        self._use_gc_spectro = self._obs['GCspectro']['GCspectro']
        try:
            self._use_xc_phot = self._obs['WL']['GCphot']
        except KeyError:
            try:
                self._use_xc_phot = self._obs['GCphot']['WL']
                warn('WL X GCPhot cross-correlation selection has not been '
                     'found in WL dictionary. Using the value defined in '
                     'GCPhot dictionary.')
            except KeyError:
                self._use_xc_phot = False
                warn('WL X GCPhot cross-correlation selection has not been '
                     'found in either WL and GCPhot dictionary. Setting it '
                     'to False.')

        self._create_data_vector()
        self._create_invcov_matrix()
        self._create_masking_vector()

    def get_data_and_masking_vector(self):
        r"""Getter.

        Returns the final unmasked data vector, unmasked inverse
        covariance matrix, and the masking vector.

        Returns
        -------
        self._data_vector: np.ndarray
            Final unmasked data vector
        self._invcov: np.ndarray
            Final unmasked inverse covariance matrix
        self._data_vector: np.ndarray
            Masking vector
        """
        return (self._data_vector,
                self._invcov_matrix,
                self._masking_vector)

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

    def _create_invcov_matrix(self):
        r"""Creates the final unmasked inverse covariance matrix.

        Computes the inevrse covariance matrix for the four individual probes,
        merges them into a single matrix (with zero cross-terms),
        and assigns the result to the internal attribute self._invcov_matrix.
        """
        invcov_wl = np.linalg.inv(self._cov['WL'])
        invcov_xc_phot = np.linalg.inv(self._cov['XC-Phot'])
        invcov_gc_phot = np.linalg.inv(self._cov['GC-Phot'])
        invcov_gc_spectro = np.linalg.inv(self._cov['GC-Spectro'])

        invcov_matrix = merge_matrices(invcov_wl, invcov_xc_phot)
        invcov_matrix = merge_matrices(invcov_matrix, invcov_gc_phot)
        invcov_matrix = merge_matrices(invcov_matrix, invcov_gc_spectro)

        self._invcov_matrix = invcov_matrix

    def _create_masking_vector(self):
        r"""Creates the masking vector.

        Creates the masking vector based on the size of the data vectors and
        on the choice of observables of the user.
        """
        masking_vector = np.concatenate(
            (
                np.full(self._wl_size, self._use_wl, dtype=int),
                np.full(self._xc_phot_size, self._use_xc_phot, dtype=int),
                np.full(self._gc_phot_size, self._use_gc_phot, dtype=int),
                np.full(self._gc_spectro_size, self._use_gc_spectro, dtype=int)
            )
        )

        self._masking_vector = masking_vector
