"""Photometric Redshift Distribution module
"""

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import where


class RedshiftDistribution:
    r"""
    Class representing the Redshift Distribution.
    """

    def __init__(self, probe, nz_dict, nuisance_dict):
        """Initialize

        Constructor of the class RedshiftDistribution.

        Parameters
        ----------
        probe: str
           It can be 'GCphot' or 'WL'.
        nz_dict: dict
            Dictionary of the form { 'n%d' : InterpolatedUnivariateSpline}.
        nuisance_dict: dict
            Dictionary containing the nuisance parameters.
            The relevant keys are:
            Shifts in z -> 'dz_%d_{probe}', where probe is 'GCphot' or 'WL'.

        Notes
        -----
        The tomographic bin number starts from one.
        """
        self.nz_dict = nz_dict
        for interp_n in self.nz_dict.values():
            # set extrapolation mode ext=1 to return 0 outside boundaries
            interp_n.ext = 1

        self.bins = [i + 1 for i in range(len(nz_dict))]
        self.dz_dict = {i: nuisance_dict[f'dz_{i}_{probe}'] for i in self.bins}
        # The uncertainties due to blending can be included
        # in future devel, by for example following MacCrann et al. (2021)
        # self.F_dict = {i: nuisance_dict[f'F_{i}_{probe}'] for i in bins}

        self.z_min = 0.001

    def evaluates_n_i_z(self, bin_i, zs):
        r"""Evaluates the redshift distribution at bin i.

        The function allows for shifts in the mean
        of the tomographic redshift distributions.

        .. math::
            n_i^{\rm (shift)}(z) = n_i (z - \Delta z_i)\\

        Note:
        We might want to extend this function to account for
        uncertainties in the redshift distributions due to blending
        using the parametrization described in
        [1] MacCrann, N., Becker, M. R., McCullough, J., et al., 2021,
        DOI: https://doi.org/10.1093/mnras/stab2870
        arXiv e-prints, arXiv:2012.08567

        Parameters
        ----------
        bin_i: int
            The tomographic redshift bin; bins starts from 1.
        zs: numpy.ndarray of float, or float
            The redshift(s) at which to evaluate the distribution.

        Returns
        -------
        n_i_z: numpy.ndarray of float, or float
            The evaluated redshift distribution.
        """
        ni = self.nz_dict[f'n{bin_i}']
        dz_i = self.dz_dict[bin_i]
        # Task 734: check the difference in this implementation
        # vs explicitly returning zero if (zs - dz_i) is out of bounds
        shifted_zs = zs - dz_i
        n_i_z = where(shifted_zs < self.z_min, 0, ni(shifted_zs))
        return n_i_z

    def interpolates_n_i(self, bin_i, zs):
        """Returns an interpolator for the redshift distribution at bin i.

        Parameters
        ----------
        bin_i: int
            The tomographic redshift bin; bins starts from 1.
        zs: numpy.ndarray of float
            The redshifts used for interpolating the distribution.

        Returns
        -------
        n_i: InterpolatedUnivariateSpline
            The interpolated redshift distribution for bin i.
        """
        values = self.evaluates_n_i_z(bin_i, zs)
        return InterpolatedUnivariateSpline(zs, values, ext='zeros')

    def get_tomographic_bins(self):
        """Gets the list of tomographic bins.

        Returns
        -------
        tomographic_bins: list of int
            The list of tomographic bins.
        """
        return self.bins

    def get_num_tomographic_bins(self):
        """Gets the number of tomographic bins.

        Returns
        -------
        num_tomographic_bins: int
            The number of tomographic bins.
        """
        return len(self.bins)
