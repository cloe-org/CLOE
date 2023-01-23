# -*- coding: utf-8 -*-
"""MATRIX Transforms

Set of transforms for data and theory matrices in CLOE
"""

import numpy as np


class VectorizeMatrix(object):
    """
    Class Vectorize Matrix
    """

    def __init__(self, N):
        """Initialize

        Constructor of the class VectorizeMatrix

        Parameters
        ----------
        N: int
            dimension of the square matrix whose vectorisation
            or half-vectorisation will be computed
        """

        self.N = N

    def _u_vec(self, i, j):
        return np.array([1 if a == j * self.N + i - 1 / 2 * j * (j + 1) else 0
                        for a in range(self.N * (self.N + 1) // 2)])[:, None]

    def _T_vec(self, i, j):
        return np.array([1 if ((a == i and b == j) or (a == j and b == i))
                        else 0 for a in range(self.N)
                        for b in range(self.N)])[:, None]

    def _E_vec(self, i, j):
        return np.dot(np.identity(self.N)[:, i:i + 1],
                      np.identity(self.N)[j:j + 1, :]).reshape((self.N *
                                                                self.N, 1))

    def D_mat(self):
        """
        D_mat

        Calculates duplication matrix

        Returns
        -------
        D_mat: array
             duplication matrix

        """

        return np.sum([np.dot(self._u_vec(i, j),
                       self._T_vec(i, j).T)
                       for j in range(self.N)
                       for i in range(j, self.N)], axis=0).T

    def E_mat(self):
        """
        E_mat

        Calculates elimination matrix

        Returns
        -------
        E_mat: array
             elimination matrix
        """

        return np.sum([np.dot(self._u_vec(i, j), self._E_vec(i, j).T)
                      for j in range(self.N)
                      for i in range(j, self.N)], axis=0)


class BNT_transform():
    """
    Class BNT transform
    """

    def __init__(self, z_array, comoving_dist, n_i_z_array):
        """
        Initialize

        Constructor for the BNT matrix transform

        Parameters
        ----------
        z_array: array
            Array of z values at which the comoving distance and n(z)
            are computed
        comoving_dist: array
            Array containing the comoving distance at the values of z_array
        n_i_z_array: array
            Array containing the galaxy number density n(z)
            at the values of z_array
        """

        self.z = z_array
        self.chi = comoving_dist
        self.n_i_list = n_i_z_array
        self.N_bins = len(self.n_i_list)

    def get_matrix(self):
        """
        Get Matrix

        Compute Matrix for the BNT transform

        Returns
        -------
        BNT_matrix: array
             2D-array containing the BNT matrix transform
        """

        A_list = []
        B_list = []
        for i in range(self.N_bins):
            nz = self.n_i_list[i]
            A_list += [np.trapz(nz, self.z)]
            B_list += [np.trapz(nz / self.chi, self.z)]

        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1, 0] = -1.

        for i in range(2, self.nbins):
            mat = np.array([[A_list[i - 1], A_list[i - 2]],
                           [B_list[i - 1], B_list[i - 2]]])
            A = -1. * np.array([A_list[i], B_list[i]])
            soln = np.dot(np.linalg.inv(mat), A)
            BNT_matrix[i, i - 1] = soln[0]
            BNT_matrix[i, i - 2] = soln[1]
        return BNT_matrix
