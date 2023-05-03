# -*- coding: utf-8 -*-
"""MATRIX Transforms

Set of transforms for data and theory matrices in CLOE

The class VectorizeMatrix computes linear operators that allow to
vectorize 2D-matrices into 1D-vectors.
See Appendix B of the CLOE v2 release documentation.

The class BNT_transform computes the BNT matrix, given a fiducial cosmology.
This matrix can be used to modify the photometric C_ells to localize
their Kernels and allow for optimal scale-redshift cuts.
See Section 7 of the CLOE v2 release documentation.

"""

import numpy as np


class VectorizeMatrix(object):
    """
    Class VectorizeMatrix

    The class VectorizeMatrix computes linear operators that allow to
    vectorize 2D-matrices into 1D-vectors.
    See Appendix B of the CLOE v2 release documentation.
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

    def _u_vector_operator(self, i, j):
        return np.array([1 if a == j * self.N + i - 1 / 2 * j * (j + 1) else 0
                        for a in range(self.N * (self.N + 1) // 2)])[:, None]

    def _t_vector_operator(self, i, j):
        return np.array([1 if ((a == i and b == j) or (a == j and b == i))
                        else 0 for a in range(self.N)
                        for b in range(self.N)])[:, None]

    def _e_vector_operator(self, i, j):
        return np.dot(np.identity(self.N)[:, i:i + 1],
                      np.identity(self.N)[j:j + 1, :]).reshape((self.N *
                                                                self.N, 1))

    def duplication_matrix(self):
        """
        duplication_matrix

        Calculates duplication matrix

        Returns
        -------
        D_mat: array
             duplication matrix

        """

        return np.sum([np.dot(self._u_vector_operator(i, j),
                       self._t_vector_operator(i, j).T)
                       for j in range(self.N)
                       for i in range(j, self.N)], axis=0).T

    def elimination_matrix(self):
        """
        elimination_matrix

        Calculates elimination matrix

        Returns
        -------
        E_mat: array
             elimination matrix
        """

        return np.sum([np.dot(self._u_vector_operator(i, j),
                      self._e_vector_operator(i, j).T)
                      for j in range(self.N)
                      for i in range(j, self.N)], axis=0)


class BNT_transform():
    """
    Class BNT_transform

    The class BNT_transform computes the BNT matrix,
    given a fiducial cosmology.
    This matrix can be used to modify the photometric C_ells to localize
    their Kernels and allow for optimal scale-redshift cuts.
    See Section 7 of the CLOE v2 release documentation for a description
    of the BNT matrix.
    For the vectorized formalism and application see appendix B of the CLOE v2
    release documentation.
    """

    def __init__(self, z_array, comoving_dist, n_i_z_array,
                 test_unity=False):
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
        self.Nz_bins = len(self.n_i_list)
        self.test_unity = test_unity
        self.BNT_matrix = self.compute_BNT_matrix()

    def compute_BNT_matrix(self):
        """
        Compute BNT Matrix

        Compute Matrix for the BNT transform

        Returns
        -------
        BNT_matrix: array
             2D-array containing the BNT matrix transform
        """

        A_list = np.zeros((self.Nz_bins))
        B_list = np.zeros((self.Nz_bins))
        for i in range(self.Nz_bins):
            nz = self.n_i_list[i]
            A_list[i] = np.trapz(nz, self.z)
            B_list[i] = np.trapz(nz / self.chi, self.z)

        BNT_matrix = np.eye(self.Nz_bins)
        if self.test_unity is True:
            return BNT_matrix
        else:
            BNT_matrix[1, 0] = -1.
            for i in range(2, self.Nz_bins):
                mat = np.array([[A_list[i - 1], A_list[i - 2]],
                               [B_list[i - 1], B_list[i - 2]]])
                A = -1. * np.array([A_list[i], B_list[i]])
                soln = np.dot(np.linalg.inv(mat), A)
                BNT_matrix[i, i - 1] = soln[0]
                BNT_matrix[i, i - 2] = soln[1]
            return BNT_matrix

    def apply_vectorized_symmetric_BNT(self, N_ell_bins, observed_array):
        """
        apply_vectorized_symmetric_BNT

        Apply vectorized BNT transformed for symmetric array in z-indices

        Parameters
        ----------
        N_ell_bins: int
            Number of photometric multipole bins
        observed_array: array
            Stacked array containing the
            angular spectra C_ells to be transformed

        Returns
        -------
        transformed_array: array
            Stacked array containing the
            BNT-transformed angular spectra C_ells
        """

        self.N_mat_ell = np.identity(N_ell_bins)
        Vec = VectorizeMatrix(self.Nz_bins)
        D_mat = Vec.duplication_matrix()
        E_mat = Vec.elimination_matrix()
        C_slash_mat = (np.kron(self.N_mat_ell, D_mat) @ observed_array)
        B_kron = np.kron(self.BNT_matrix, self.BNT_matrix)
        B_kron_Ell = np.kron(self.N_mat_ell, B_kron)
        Ell_mat = np.kron(self.N_mat_ell, E_mat)
        transformed_array = (Ell_mat @ B_kron_Ell @ C_slash_mat)
        return transformed_array

    def apply_vectorized_nonsymmetric_BNT(self, N_ell_bins, observed_array):
        """
        apply_vectorized_symmetric_BNT

        Apply vectorized BNT transformed for symmetric array in z-indices

        Parameters
        ----------
        N_ell_bins: int
            Number of photometric multipole bins
        observed_array: array
            Stacked array containing the
            angular spectra C_ells to be transformed

        Returns
        -------
        transformed_array: array
            Stacked array containing the
            BNT-transformed angular spectra C_ells
        """

        self.N_mat_ell = np.identity(N_ell_bins)
        self.N_mat_z = np.identity(self.Nz_bins)
        A_slash_mat = \
            np.kron(self.N_mat_ell,
                    np.kron(self.N_mat_z, self.BNT_matrix)
                    )
        transformed_array = A_slash_mat @ observed_array
        return transformed_array
