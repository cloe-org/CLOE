# -*- coding: utf-8 -*-
r"""MASKING PACKAGE

This module contains methods for implementing the masking of the
data vector and of the covariance matrix such that
the input matrix is a 2-d array representing a square matrix,
whose number of rows and columns must match the size of the data
vector, of the theory vector and of the masking vector.
It is assumed that the covariance matrix is already unrolled
and stacked with the same arrangement used for the data and masking
vectors.
"""

__all__ = ['data_handler',
           'masking']  # list submodules

from . import *
