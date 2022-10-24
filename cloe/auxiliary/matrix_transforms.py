# -*- coding: utf-8 -*-
"""MATRIX Transforms

Set of transforms for data and theory matrices in CLOE
"""

import numpy as np

class VectorizeMatrix(object):
    """
    class with methods for vectorization of matrices
    """
    def __init__(self, 
                 N          # dimension of the square matrix whose vectorisation
                            #/half-vectorisation
                            # you want to half-vectorise/vectorise
                ):
        """
        Initialize 
        Constructor of the class VectorizeMatrix

        
        """
        self.N = N

    def _u_vec(self, i, j):
        return np.array([ 1 if a==j*self.N+i-1/2*j*(j+1) else 0 for a in range(self.N*(self.N+1)//2) ])[:, None]

    def _T_vec(self, i, j):
        return np.array([ 1 if ((a==i and b==j) or (a==j and b==i)) else 0 
                         for a in range(self.N) for b in range(self.N) ])[:, None]

    def _E_vec(self, i, j):
        return np.dot(np.identity(self.N)[:, i:i+1], np.identity(self.N)[j:j+1, :]).reshape((self.N*self.N, 1))

    def D_mat(self):
        return np.sum([ np.dot(self._u_vec(i, j), self._T_vec(i, j).T) for j in range(self.N) for i in range(j, self.N) ], axis=0).T

    def E_mat(self):
        return np.sum([ np.dot(self._u_vec(i, j), self._E_vec(i, j).T) for j in range(self.N) for i in range(j, self.N) ], axis=0)


class BNT_transform():
    def __init__(self, z, comoving_dist, n_i_z_array):

        self.z = z
        self.chi = comoving_dist
        self.n_i_list = n_i_z_array
        self.N_bins = len(self.n_i_list)


    def get_matrix(self):

        A_list = []
        B_list = []
        for i in range(self.N_bins):
            nz = self.n_i_list[i]
            A_list += [np.trapz(nz, self.z)]
            B_list += [np.trapz(nz / self.chi, self.z)]


        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1,0] = -1.

        for i in range(2, self.nbins):
            mat = np.array([ [A_list[i-1], A_list[i-2]], [B_list[i-1], B_list[i-2]] ])
            A = -1. * np.array( [A_list[i], B_list[i]] )
            soln = np.dot(np.linalg.inv(mat), A)
            BNT_matrix[i,i-1] = soln[0]
            BNT_matrix[i,i-2] = soln[1]
        
        return BNT_matrix


def merge_matrices(matrix1, matrix2):
    r"""Merge two matrices

    Create a single matrix out of matrix1 and matrix2. See the
    documentation of the return value for a detailed description of the
    format of the output matrix

    Parameters
    ----------
    matrix1: ndarray
        A matrix of arbitrary size

    matrix2: ndarray
        A matrix of arbitrary size

    Returns
    -------
    outMatrix: ndarray
        A matrix in which the number of rows is the sum of the number of rows
        of matrix1 and matrix2, and the number of columns is the sum of the
        number of columns of matrix1 and matrix2.
        matrix1 and matrix2 are stored as diagonal blocks in outMatrix.
        matrix1[row, col] is stored in outMatrix[row, col], while
        matrix2[row, col] is stored in outMatrix[nRows1 + row, nCols1 + col],
        where nRows1 and nCols1 are are the number of rows and columns of
        matrix1, respectively.
        All the elements outside the two diagonal blocks are set to 0.
    """
    nRows1, nCols1 = matrix1.shape
    nRows2, nCols2 = matrix2.shape
    outMatrix = numpy.block([
        [matrix1, numpy.zeros((nRows1, nCols2))],
        [numpy.zeros((nRows2, nCols1)), matrix2]
    ])
    return outMatrix
