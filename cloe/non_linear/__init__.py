# -*- coding: utf-8 -*-

"""NONLINEAR MODULE

This module contains the nonlinear recipes implemented by the Euclid
Inter-Science Taskforce for Nonlinearities (IST:NL).
The module :obj:`miscellaneous` is a
temporary module that mimics the current linear implementation made by
IST:L and will be substituted by the nonlinear recipes from IST:NL.

"""

__all__ = ['miscellanous',
           'power_spectrum',
           'pgg_spectro',
           'pgg_phot',
           'pgL_phot',
           'pLL_phot',
           'nonlinear']  # list submodules

from . import *
