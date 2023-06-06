# -*- coding: utf-8 -*-

"""FFTLOG MODULE

This module computes integrals with the fast Fourier or
Hankel (= Fourier-Bessel) transforms of a periodic sequence
of logarithmically spaced points with using the
FFTLog algorithm described in `Hamilton (2000)
<https://arxiv.org/abs/astro-ph/9905191v4>`__.

"""

__all__ = ['fftlog',
           'hankel',
           'utils']  # list submodules

from . import *
