"""Nonlinear

Class to compute non-linear recipes.
"""

import numpy as np
from scipy import interpolate


class NonlinearError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Nonlinear:
    """
    Class to compute non-linear recipes
    """

    def __init__(self, cosmo_dic):
        """
        Initialise class and non-linear code

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from Cosmology class
        """
        self.theory = cosmo_dic

    def update_dic(self, cosmo_dic):
        """
        Call all routines updating the cosmo dictionary

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from Cosmology class

        Returns
        -------
            Updated dictionary
        """
        self.theory = cosmo_dic
        self.calculate_boost()

        return self.theory

    def calculate_boost(self):
        """
        Check non-linear flag and computes the corresponding
        boost-factor, adding it to the dictionary
        """
        switcher = {1: self.linear_boost}
        boost = switcher.get(self.theory['nuisance_parameters']['NL_flag'],
                             "Invalid modeling option")
        self.theory['NL_boost'] = boost

    def linear_boost(self, redshift, scale):
        """
        Returns the boost factor for the linear case (i.e. 1)

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate the boost
        scale: float
            Wave mode at which to calculate the boost

        Returns
        -------
        Value of linear boost at input redhsift and scale
        """
        boost = 1.0
        return boost
