# -*- coding: utf-8 -*-
r"""MASKING MODULE

Module for masking the data vector and the covariance matrix.
"""

import numpy


class Masking:
    r"""Masking of the data vector and of the covariance matrix.
    """

    def __init__(self):
        r"""Constructor.

        All the data members are initialized to None.
        """
        self._data_vector = None
        self._theory_vector = None
        self._covariance_matrix = None
        self._masking_vector = None
        self._masked_data_vector = None
        self._masked_theory_vector = None
        self._masked_covariance_matrix = None

    def set_covariance_matrix(self, matrix):
        r"""Sets the covariance matrix.

        Parameters
        ----------
        matrix: numpy.ndarray
          The input matrix should be a 2-d array representing a square matrix,
          whose number of rows and columns must match the size of the data
          vector, of the theory vector and of the masking vector.
          It is assumed that the covariance matrix is already unrolled
          and stacked with the same arrangement used for the data and masking.
          vectors

        Raises
        ------
        TypeError
          If the matrix is not a 2-d array
        TypeError
          If the matrix is not a square matrix
        """
        if len(matrix.shape) != 2:
            raise TypeError(f'The size of the input array is'
                            f' {len(matrix.shape)}, while a 2-d array was'
                            f' expected')
        if matrix.shape[0] != matrix.shape[1]:
            raise TypeError(f'the input matrix is not a square matrix, it has'
                            f' size {matrix.shape[0]}x{matrix.shape[1]}')
        self._covariance_matrix = matrix
        self._masked_covariance_matrix = None

    def set_masking_vector(self, vector):
        r"""Sets the masking vector.

        Parameters
        ----------
        vector: numpy.ndarray
          The masking vector is a 1-d array of int and its size must match
          that of the data vector and of the theory vector
        """
        self._masking_vector = vector.astype(bool)
        self._masked_data_vector = None
        self._masked_covariance_matrix = None

    def set_data_vector(self, vector):
        r"""Sets the data vector.

        Parameters
        ----------
        vector: numpy.ndarray
          The data vector is a 1-d array. Its size must match the size of the
          masking vector and of the theory vector
        """
        self._data_vector = vector
        self._masked_data_vector = None

    def set_theory_vector(self, vector):
        r"""Sets the theory vector.

        Parameters
        ----------
        vector: numpy.ndarray
          The theory vector is a 1-d array. Its size must match the size of
          the masking vector and of the data vector
        """
        self._theory_vector = vector
        self._masked_theory_vector = None

    def get_masked_data_vector(self):
        r"""Gets the masked data vector.

        Returns
        -------
        Masked data vector: numpy.ndarray
          The masked data vector, obtained by removing from the input data
          vector the elements corresponding to zero-entries in the masking
          vector.
          The size of the masked data vector is given by the number of nonzero
          entries in the masking vector

        Raises
        ------
        TypeError
          If the masking vector is not set or the data vector is not set
        """
        if self._masking_vector is None:
            raise TypeError(f'The masking vector is not set')
        if self._data_vector is None:
            raise TypeError(f'The data vector is not set')

        if self._masked_data_vector is None:
            self._masked_data_vector = self._data_vector[self._masking_vector]

        return self._masked_data_vector

    def get_masked_theory_vector(self):
        r"""Gets the masked theory vector.

        Returns
        -------
        Masked theory vector: numpy.ndarray
          The masked theory vector, obtained by removing from the input theory
          vector the elements corresponding to zero-entries in the masking
          vector.
          The size of the masked theory vector is given by the number of
          nonzero entries in the masking vector

        Raises
        ------
        TypeError
          If the masking vector is not set or the theory vector is not set
        """
        if self._masking_vector is None:
            raise TypeError(f'The masking vector is not set')
        if self._theory_vector is None:
            raise TypeError(f'The theory vector is not set')

        if self._masked_theory_vector is None:
            self._masked_theory_vector = (
                self._theory_vector[self._masking_vector])

        return self._masked_theory_vector

    def get_masked_covariance_matrix(self):
        r"""Gets the masked covariance matrix.

        Returns
        -------
        Masked covariance matrix: numpy.ndarray
          A 2-d numpy.ndarray representing the masked covariance matrix.
          It is obtained by removing from the input covariance matrix
          the rows and the columns that correspond to zero-valued entries in
          the masking vector.
          The number of rows (and columns) of the masked covariance matrix
          is given by the number of nonzero entries in the masking vector

        Raises
        ------
        TypeError
          If the masking vector is not set or the covariance matrix is not set
        """
        if self._masking_vector is None:
            raise TypeError(f'The masking vector is not set')
        if self._covariance_matrix is None:
            raise TypeError(f'The covariance matrix is not set')

        if self._masked_covariance_matrix is None:
            self._masked_covariance_matrix = (
                self._covariance_matrix
                [self._masking_vector][:, self._masking_vector]
            )

        return self._masked_covariance_matrix
