# -*- coding: utf-8 -*-

"""CLOE PACKAGE

The Cosmology Likelihood for Observables in Euclid (CLOE)
is a software package developed by the Euclid Consortium.
This documentation provides a description of the modules
in CLOE.

"""

# Explicitly import all subpackages so Sphinx autodoc can discover them
from . import (
    auxiliary,
    clusters_of_galaxies,
    cmbx_p,
    cobaya_interface,
    cosmo,
    data_reader,
    fftlog,
    info,
    like_calc,
    masking,
    non_linear,
    photometric_survey,
    spectroscopic_survey,
    tests,
    user_interface,
)

# Version comes from the info submodule
from .info import __version__

# Define public API
__all__ = [
    "auxiliary",
    "clusters_of_galaxies",
    "cmbx_p",
    "cobaya_interface",
    "cosmo",
    "data_reader",
    "fftlog",
    "info",
    "like_calc",
    "masking",
    "non_linear",
    "photometric_survey",
    "spectroscopic_survey",
    "tests",
    "user_interface",
    "__version__",
]
