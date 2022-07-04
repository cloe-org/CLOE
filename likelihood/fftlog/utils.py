# -*- coding: utf-8 -*-
"""Utils

Utility functions for the FFTLog module.
"""

import numpy as np
from scipy.special import gamma


def _log_extrap(x, N_extrap_begin, N_extrap_end):
    r"""Log extrapolation

    This function takes as input an array :math:`x`, evaluates its
    behaviour at extrema and use them to logarithmically extrapolate
    the array. The :math:`i`-th point is generated at the beginning/end of
    the array according to
    :math:`x_{i}=x_{0} \exp \left(i \Delta_{\ln x}\right)`, where :math:`x_0`
    is the first/last value and :math:`\Delta_{\ln x}` is the logarithm of the
    ratio of the first/last two values

    Parameters
    ----------
    x: array
        array to be extrapolated

    N_extrap_begin: Int
        number of points to add at the beginning of the input array

    N_extrap_end: Int
        number of points to add at the end of the input array

    Returns
    -------
    x_extrap: array
        returns the extrapolated array
    """

    low_x = high_x = []
    if(N_extrap_begin):
        dlnx_low = np.log(x[1] / x[0])
        low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_end, 0))
    if(N_extrap_end):
        dlnx_high = np.log(x[-1] / x[-2])
        high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_end + 1))

    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap


def _c_window(n, n_cut):
    r"""_c_window

    One-side window function of c_m,
    Adapted from Eq.(C1) in
    `McEwen et al. (2016) <https://arxiv.org/abs/1603.04826>`_

    .. math::
        W(x)= \begin{cases}
        \frac{x_{\max }-x}{x_{\max }-x_{\text {cut }}}-\frac{1}{2 \pi}
        \sin \left(2 \pi \frac{x_{\max }-x}{x_{\max }-
        x_{\text {cut }}}\right) & x>x_{\text {cut }} \\
        1 & \text{else}
        \end{cases}

    Parameters
    ----------
    n: array
        this array is assumed to be given by np.arange(0, N // 2 + 1),
        where N is the number of elements in the array to be transformed

    n_cut: Int
        specifies where the smoothing of c_m must start

    Returns
    -------
    W: array
        array containing the smoothing coefficients
    """

    n_right = n[-1] - n_cut
    n_r = n[n[:] > n_right]
    theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1)
    W = np.ones(n.size)
    W[n[:] > n_right] = theta_right - 1 / (2 * np.pi) \
        * np.sin(2 * np.pi * theta_right)
    return W


def _g_m_vals(mu, q):
    r"""Stable _g_m_vals

    Method adapted from FAST-PT, which computes

    .. math::
        \frac{\Gamma((\mu+1+q)/2)}{\Gamma((\mu+1-q)/2)}

    Switching to asymptotic form when |Im(q)| + |mu| > cut = 200, as done
    in FAST-PT

    Parameters
    ----------
    mu: Float
        the first coefficient is related to the value of the multipole
        involved in the FFTLog integral/Hankel transform

    q: array
        this array specifies the grid over which the _g_m_vals function is
        evaluated

    Returns
    -------
    g_m: array
        array containing the evaluated function
    """
    if(mu + 1 + q.real[0] == 0):
        print("gamma(0) encountered. Please change another nu value!")
        exit()
    imag_q = np.imag(q)
    g_m = np.zeros(q.size, dtype=complex)
    cut = 200
    cut_criterion_array = np.absolute(np.imag(q)) - np.absolute(mu)
    asym_q = q[cut_criterion_array > cut]
    asym_plus = (mu + 1 + asym_q) / 2.
    asym_minus = (mu + 1 - asym_q) / 2.

    q_good_bool_array = q[(cut_criterion_array <= cut) &
                          (q != mu + 1 + 0.0j)]
    q_good = q_good_bool_array

    alpha_plus = (mu + 1 + q_good) / 2.
    alpha_minus = (mu + 1 - q_good) / 2.

    g_m[(cut_criterion_array <= cut) &
        (q != mu + 1 + 0.0j)] = gamma(alpha_plus) / gamma(alpha_minus)

    # asymptotic form, taken from
    # https://github.com/JoeMcEwen/FAST-PT/blob/master/fastpt/gamma_funcs.py
    abs_im_q = np.absolute(imag_q)
    abs_mu = np.absolute(mu)

    # to improve readibility, the argument of the exponential has been divided
    # in more terms and then summed
    term1 = (asym_plus - 0.5) * np.log(asym_plus)
    term2 = - (asym_minus - 0.5) * np.log(asym_minus)
    term3 = - asym_q
    term4 = 1. / 12 * (1. / asym_plus - 1. / asym_minus)
    term5 = 1. / 360. * (1. / asym_minus**3 - 1. / asym_plus**3)
    term6 = 1 / 1260 * (1. / asym_plus**5 - 1. / asym_minus**5)
    exp_arg = term1 + term2 + term3 + term4 + term5 + term6

    g_m[cut_criterion_array > cut] = np.exp(exp_arg)

    g_m[np.where(q == mu + 1 + 0.0j)[0]] = 0. + 0.0j
    return g_m


def _g_l(ell, z_array):
    r"""Computes _g_l

    Computes the _g_l function, defined as

    .. math::
        g_\ell(z) = 2^z * \Gamma((\ell+z)/2) / \Gamma((3+\ell-z)/2)

    Parameters
    ----------
    ell: Float
        order of the Bessel function of the FFTLog transform
    z: array
        input array used to define the domain of _g_l

    Returns
    -------
    gl: array
        computed values of the _g_l function
    """
    gl = 2.**z_array * _g_m_vals(ell + 0.5, z_array - 1.5)
    return gl
