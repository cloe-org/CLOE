# -*- coding: utf-8 -*-
"""HANKEL MODULE

Class to implement the FFTLog algorithm to perform the Hankel
transform of a periodic sequence of logarithmically spaced points.
This is based on the work published in
`Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>` and
`Fang et al. (2020) <https://arxiv.org/abs/1911.11947>`
"""

from cloe.fftlog.fftlog import fftlog
import numpy as np


class hankel(object):
    r"""Class for the Hankel transform.

    It implements Hankel transform, defined as

    .. math::
        F_{\rm n}(y) = \int_0^\infty dx f(x) J_{\rm n}(xy)

    where :math:`J_{\rm n}` is the Bessel function of first kind and
    order :math:`{\rm n}`.

    It is based on the algorithm described in
    `Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>`__.
    It decomposes the integrand function using the Fast Fourier Transform;
    after this decomposition, each term can be integrated analitically.
    Please refer to the :obj:`fftlog` class for more details.
    """

    def __init__(self, x, fx, nu=1.1, N_extrap_begin=0, N_extrap_end=0,
                 c_window_width=0.25, N_pad=0):
        """List of parameters.

        Parameters
        ----------
        x: numpy.ndarray
            Original logarithmically sampled domain of the input function. If
            not even, the algorithmic implementation modifies it accordingly.
        fx: numpy.ndarray
            Original sampled input function
        nu: float
            Bias index, used to have a more stable Fast Fourier Transform
        N_extrap_begin: int
            Number of extrapolated points at the beginning of input arrays
        N_extrap_end: int
            Number of extrapolated points at the end of input arrays
        c_window_width: float
            Fraction of the ``c_m`` coefficients smoothed by the window
        N_pad: int
            Number of zero-padded points at the beginning and
            end of ``f_x array``
        """
        self.fftlog = fftlog(x, fx * (x**(5 / 2)), nu, N_extrap_begin,
                             N_extrap_end, c_window_width, N_pad)

    def hankel(self, n):
        r"""Hankel.

        Evaluates the Hankel transform, defined as

        .. math::
            F_{\rm n}(y) = \int_0^\infty dx f(x) J_{\rm n}(xy)

        where :math:`J_n` is the Bessel function of first kind and order
        :math:`n`.

        Parameters
        ----------
        n: int
            Order of the Hankel transform

        Returns
        -------
        y, Fy: numpy.ndarray, numpy.ndarray
            Logarithmically spaced array set as
            ``y[:] = (ell+1)/x[::-1]``, Hankel transform
            evaluated over the y array
        """
        y, Fy = self.fftlog.fftlog(n - 0.5)
        Fy *= np.sqrt(2 * y / np.pi)
        return y, Fy
