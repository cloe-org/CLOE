"""Nonlinear

Class to compute non-linear recipes.
"""

from likelihood.non_linear.miscellanous import Misc
from likelihood.non_linear.pgg_spectro import Pgg_spectro_model
from likelihood.non_linear.pgg_phot import Pgg_phot_model
from likelihood.non_linear.pgL_phot import PgL_phot_model
from likelihood.non_linear.pLL_phot import PLL_phot_model


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
        """Initialise

        Initialise class and nonlinear code

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from Cosmology class
        """
        self.theory = cosmo_dic

        self.misc = Misc(cosmo_dic)

        self.Pgg_spectro_model = Pgg_spectro_model(cosmo_dic, self.misc)
        self.Pgg_phot_model = Pgg_phot_model(cosmo_dic, self.misc)
        self.PgL_phot_model = PgL_phot_model(cosmo_dic, self.misc)
        self.PLL_phot_model = PLL_phot_model(cosmo_dic, self.misc)

    def update_dic(self, cosmo_dic):
        """Update Dic

        Call all routines updating the cosmo dictionary

        Parameters
        ----------
        cosmo_dic: dictionary
            External dictionary from Cosmology class

        Returns
        -------
        cosmo_dic: dict
            Updated dictionary

        """
        self.theory = cosmo_dic
        self.calculate_boost()

        self.misc.update_dic(cosmo_dic)

        self.Pgg_spectro_model.update_dic(cosmo_dic, self.misc)
        self.Pgg_phot_model.update_dic(cosmo_dic, self.misc)
        self.PgL_phot_model.update_dic(cosmo_dic, self.misc)
        self.PLL_phot_model.update_dic(cosmo_dic, self.misc)

        return self.theory

    def calculate_boost(self):
        """Calculate Boost

        Check non-linear flag and computes the corresponding
        boost-factor, adding it to the dictionary
        """
        switcher = {1: self.linear_boost}
        boost = switcher.get(self.theory['nuisance_parameters']['NL_flag'],
                             "Invalid modeling option")
        self.theory['NL_boost'] = boost

    def linear_boost(self, redshift, scale):
        """Linear Boost

        Returns the boost factor for the linear case (i.e. 1)

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate the boost
        scale: float
            Wave mode at which to calculate the boost

        Returns
        -------
        boost: float
           Value of linear boost at input redshift and scale

        """
        boost = 1.0
        return boost

    def Pmm_phot_def(self, redshift, k_scale, grid):
        r"""Interface for Pmm_phot_def

        Returns the photometric matter-matter power spectrum,
        defined in the pLL_phot module
        """
        return self.PLL_phot_model.Pmm_phot_def(redshift, k_scale, grid)

    def Pgg_spectro_def(self, redshift, k_scale, mu_rsd):
        r"""Interface for Pgg_spectro_def

        Returns the spectroscopic galaxy-galaxy power spectrum,
        defined in the pgg_spectro module
        """
        return self.Pgg_spectro_model.Pgg_spectro_def(redshift,
                                                      k_scale, mu_rsd)

    def Pgdelta_spectro_def(self, redshift, k_scale, mu_rsd):
        r"""Interface for Pgdelta_spectro_def

        Returns the spectroscopic galaxy-density power spectrum,
        defined in the pgg_spectro module
        """
        return self.Pgg_spectro_model.Pgdelta_spectro_def(redshift,
                                                          k_scale, mu_rsd)

    def Pgg_phot_def(self, redshift, k_scale):
        r"""Interface for Pgg_phot_def

        Returns the photometric galaxy-galaxy power spectrum,
        defined in the pgg_phot module
        """
        return self.Pgg_phot_model.Pgg_phot_def(redshift, k_scale)

    def Pii_def(self, redshift, k_scale):
        r"""Interface for Pii_def

        Returns the intrinsic-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        return self.PLL_phot_model.Pii_def(redshift, k_scale)

    def Pdeltai_def(self, redshift, k_scale):
        r"""Interface for Pdeltai_def

        Returns the density-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        return self.PLL_phot_model.Pdeltai_def(redshift, k_scale)

    def Pgi_phot_def(self, redshift, k_scale):
        r"""Interface for Pgi_phot_def

        Returns the photometric galaxy-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        return self.PLL_phot_model.Pgi_phot_def(redshift, k_scale)

    def Pgi_spectro_def(self, redshift, k_scale):
        r"""Interface for Pgi_spectro_def

        Returns the spectroscopic galaxy-intrinsic power spectrum,
        defined in the pLL_phot module
        """
        return self.PLL_phot_model.Pgi_spectro_def(redshift, k_scale)

    def Pgdelta_phot_def(self, redshift, k_scale):
        r"""Interface for Pgdelta_phot_def

        Returns the photometric galaxy-density power spectrum,
        defined in the pgL_phot module
        """
        return self.PgL_phot_model.Pgdelta_phot_def(redshift, k_scale)
