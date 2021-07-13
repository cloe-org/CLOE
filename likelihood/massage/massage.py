# -*- coding: utf-8 -*-
"""MATRIX MANIPULATION

Functions for manipulating matrices
"""

import numpy


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
    nRows1 = matrix1.shape[0]
    nCols1 = matrix1.shape[1]
    nRows2 = matrix2.shape[0]
    nCols2 = matrix2.shape[1]
    outMatrix = numpy.zeros((nRows1 + nRows2, nCols1 + nCols2))

    for row in range(0, nRows1):
        for col in range(0, nCols1):
            outMatrix[row, col] = matrix1[row, col]

    for row in range(0, nRows2):
        for col in range(0, nCols2):
            outMatrix[nRows1 + row, nCols1 + col] = matrix2[row, col]

    return outMatrix
