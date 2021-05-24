"""
module: power_spectrum

Contains mother class for a generic power spectrum.
"""

# Global
import numpy as np


class PowerSpectrum:
    r"""
    Class for generic power spectrum

    For the moment it only contains
    routines for the initialization and the update of the attributes
    which are shared by the four child classes
    """

    def __init__(self, cosmo_dic, misc):
        """Initialize

        Constructor of the class PowerSpectrum

        Parameters
        ----------
        cosmo_dic: dict
            cosmological dictionary from nonlinear module
        misc: Misc
            class containining informations needed by linear recipe
        """
        self.theory = cosmo_dic
        self.misc = misc

    def update_dic(self, cosmo_dic, misc):
        """
        Update theory with an external cosmo dictionary
        """
        self.theory = cosmo_dic
        self.misc = misc
