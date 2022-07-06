# -*- coding: utf-8 -*-
"""hankel

Class to perform hankel transforms with the FFTLog algorithm.
"""

from cloe.fftlog.fftlog import fftlog
import numpy as np


class hankel(object):
    r"""Class for the Hankel transform

    Class for the Hankel transform, defined as

    .. math::
        F_n(y) = \int_0^\infty dx f(x) J_n(xy)

    where :math:`J_n` is the Bessel function of first kind and order :math:`n`

    It is based on the algorithm described in
    `Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>`__
    It decomposes the integrand function using the Fast Fourier Transform;
    after this decomposition, each term can be integrated analitically.
    Please refer to the fftlog class for more details.
    """

    def __init__(self, x, fx, nu=1.1, N_extrap_begin=0, N_extrap_end=0,
                 c_window_width=0.25, N_pad=0):
        """List of parameters

        Parameters
        ----------
        x: array
            original logarithmically sampled domain of the input function. If
            not even, the algorithmic implementation modifies it accordingly.
        fx: array
            original sampled input function
        nu: Float
            bias index, used to have a more stable fft
        N_extrap_begin: Int
            number of extrapolated points at the beginning of input arrays
        N_extrap_end: Int
            number of extrapolated points at the end of input arrays
        c_window_width: Float
            fraction of the c_m coefficients smoothed by the window
        N_pad: Int
            number of zero-padded points at the beginning and end of f_x array
        """
        self.fftlog = fftlog(x, fx * (x**(5 / 2)), nu, N_extrap_begin,
                             N_extrap_end, c_window_width, N_pad)

    def hankel(self, n):
        r"""
        Evaluates the Hankel transform, defined as

        .. math::
            F_n(y) = \int_0^\infty dx f(x) J_n(xy)

        where :math:`J_n` is the Bessel function of first kind and order
        :math:`n`

        Parameters
        ----------
        n: Int
            order of the Hankel transform

        Returns
        -------
        y: array
            logarithmically spaced array, set as y[:] = (ell+1)/x[::-1]
        Fy: array
            Hankel transform, evaluated over the y array
        """
        y, Fy = self.fftlog.fftlog(n - 0.5)
        Fy *= np.sqrt(2 * y / np.pi)
        return y, Fy
