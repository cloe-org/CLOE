# -*- coding: utf-8 -*-
"""PLOTTER DEFAULT

Contains default settings for plotting routines.
"""

default_settings = {
    'binX': 1,  # Photometric bin X
    'binY': 1,  # Photometric bin Y
    'ltype': 'log',  # Binning type for photometric probes
    'lmin': 10,  # Minimum ell for photometric probes
    'lmax': 5000,  # Maximum ell for photometric probes
    'nl': 1000,  # Number of bins for photometric probes
    'photo_colours': 'r',  # Line colour for photometric probes
    'photo_linestyle': 'dashed',  # Line style for photometric probes
    'redshift': 1.2,  # Redshift for spectroscopic probe
    'ktype': 'log',  # Binning type for spectroscopic probes
    'kmin': 0.001,  # Minimum scale for spectroscopic probes
    'kmax': 0.5,  # Maximum scale for spectroscopic probes
    'nk': 1000,  # Number of bins for spectroscopic probes
    'spec_colours': ['r', 'r', 'r'],  # Line colour for spectroscopic probes
    'spec_linestyle': 'dashed',  # Line style for spectroscopic probes
    'path': 'output/',  # Output path for storing product files
    'file_WL': 'CLOE_WL_predictions',  # Root for WL probe
    'file_GCphot': 'CLOE_GCphot_predictions',  # Root for GCphot probe
    'file_XC': 'CLOE_XC_predictions',  # Root for WL X GCphot probe
    'file_GCspec': 'CLOE_GCspec_predictions'  # Root for GCspec probe
                    }
