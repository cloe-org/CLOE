"""
POWER SPECTRUM

This module contains the mother class to compute a generic power spectrum.
"""


class PowerSpectrum:
    r"""
    Class for generic power spectrum.

    For the moment it only contains
    routines for the initialization and the update of the attributes
    which are shared by the four child classes.
    """

    def __init__(self, cosmo_dic, nonlinear_dic, misc):
        """Initialise.

        Constructor of the class PowerSpectrum.

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary from nonlinear module
        nonlinear_dic: dict
            Nonlinear dictionary from nonlinear module
        misc: Misc
            Class containing information needed by linear recipe
        """
        self.theory = cosmo_dic
        self.nonlinear_dic = nonlinear_dic
        self.misc = misc

    def update_dic(self, cosmo_dic, nonlinear_dic, misc):
        """
        Updates theory with an external cosmo dictionary.
        """
        self.theory = cosmo_dic
        self.nonlinear_dic = nonlinear_dic
        self.misc = misc
