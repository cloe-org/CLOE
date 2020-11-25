# -*- coding: utf-8 -*-
"""Plotter

Contains class to plot cosmological observables.
"""

import numpy as np

from likelihood.photometric_survey.photo import Photo
from likelihood.spectroscopic_survey.spec import Spec
from likelihood.data_reader.reader import Reader


class PlotError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Plotter:
    """
    Class to plot observables (power spectra and multipoles).
    """

    def __init__(self, cosmo_dic):
        """
        Constructor of class Plotter.

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary in structure specified within
            cosmo.cosmology.
        """
        self.cosmo_dic = cosmo_dic
        self.read_data = Reader()
        self.read_data.compute_nz()
        self.read_data.read_GC_spec()
        self.read_data.read_phot()
        # ACD: As we currently don't have a way to load fiducial r(z), d(z) and
        # H(z)s, these are currently set to match the ones from the cosmo_dic.
        # This will need to be fixed once this is decided.
        fid_dict = self.read_data.data_spec_fiducial_cosmo
        fid_dict['d_z_func'] = self.cosmo_dic['d_z_func']
        fid_dict['r_z_func'] = self.cosmo_dic['r_z_func']
        fid_dict['H_z_func'] = self.cosmo_dic['H_z_func']
        self.phot_ins = Photo(cosmo_dic, self.read_data.nz_dict_WL,
                              self.read_data.nz_dict_GC_Phot)
        self.spec_ins = Spec(cosmo_dic, fid_dict)

    def plot_Cl_WL(self, ells, bin_i, bin_j, pl_ax, pl_label=None,
                   pl_colour='b', pl_linestyle='-'):
        """
        Function to plot WL Cls computed with the cosmobox code
        for a given set of ells, and bin combination.

        Parameters
        ----------
        ells: array
            Ell-modes at which to evaluate power spectra. Note: minimum allowed
            ell is 10.
        bin_i: int
            Index of first redshift bin. Note: bin indices start
            from 1.
        bin_j: int
            Index of second redshift bin. Note: bin indices start
            from 1.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "Bin {:d} - Bin {:d}".format(bin_i, bin_j).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        """
        if ells[0] < 10.0:
            raise Exception('Minimum allowed ell-mode to plot is 10.0.')
        cl_arr = np.array([self.phot_ins.Cl_WL(cur_ell, bin_i, bin_j) for
                           cur_ell in ells])
        if pl_label is None:
            pl_label = "Bin {:d} - Bin {:d}".format(bin_i, bin_j)
        pl_ax.plot(ells, cl_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_Cl_Wl(self, bin_1, bin_2, pl_ax, pl_label=None,
                            pl_colour='b', pl_linestyle='-', no_bins=10):
        """
        Plots external OU-LE3 WL power spectra, for given redshift bin
        combination, and errors on those.

        Parameters
        ----------
        bin_1: int
            Index of first redshift bin. Note: bin indices start
            from 1.
        bin_2: int
            Index of second redshift bin. Note: bin indices start
            from 1.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "OU-LE3 Bin {:d} - Bin {:d}".format(bin_i, bin_j).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        no_bins: int
            Number of redshift bins for WL probe.
        """
        if bin_1 > no_bins or bin_2 > no_bins:
            raise Exception('Reqeusted bin index greater than number of bins.')
        if bin_1 <= bin_2:
            bin_i = bin_1
            bin_j = bin_2
        else:
            bin_i = bin_2
            bin_j = bin_1
        ells = self.read_data.data_dict['WL']['ells']
        ext_cs = self.read_data.data_dict['WL']['E{:d}-E{:d}'.format(bin_i,
                                                                     bin_j)]
        if pl_label is None:
            pl_label = "OU-LE3 Bin {:d} - Bin {:d}".format(bin_i, bin_j)
        pl_ax.plot(ells, ext_cs, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        cov_diags = np.sqrt(np.diagonal(self.read_data.data_dict['WL']['cov']))
        counter = 0
        for i in range(1, no_bins + 1):
            for j in range(i, no_bins + 1):
                if i == bin_i and j == bin_j:
                    cur_index = counter
                counter += 1
        err_arr = []
        for mult in range(len(ells)):
            cur_err = cov_diags[(counter * mult) + cur_index]
            err_arr.append(cur_err)
        err_arr = np.array(err_arr)
        pl_ax.plot(ells, ext_cs + err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(ells, ext_cs - err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.fill_between(ells, ext_cs - err_arr, ext_cs + err_arr,
                           color=pl_colour, alpha=0.2)
        return pl_ax

    def plot_Cl_GC_phot(self, ells, bin_i, bin_j, pl_ax, pl_label=None,
                        pl_colour='b', pl_linestyle='-'):
        """
        Function to plot GC-phot Cls computed with the cosmobox code
        for a given set of ells, and bin combination.

        Parameters
        ----------
        ells: array
            Ell-modes at which to evaluate power spectra. Note: minimum allowed
            ell is 10.
        bin_i: int
            Index of first redshift bin. Note: bin indices start
            from 1.
        bin_j: int
            Index of second redshift bin. Note: bin indices start
            from 1.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "Bin {:d} - Bin {:d}".format(bin_i, bin_j).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        """
        if ells[0] < 10.0:
            raise Exception('Minimum allowed ell-mode to plot is 10.0.')
        cl_arr = np.array([self.phot_ins.Cl_GC_phot(cur_ell, bin_i, bin_j) for
                           cur_ell in ells])
        if pl_label is None:
            pl_label = "Bin {:d} - Bin {:d}".format(bin_i, bin_j)
        pl_ax.plot(ells, cl_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_Cl_GC_phot(self, bin_1, bin_2, pl_ax, pl_label=None,
                                 pl_colour='b', pl_linestyle='-', no_bins=10):
        """
        Plots external OU-LE3 GC-phot power spectra, for given redshift bin
        combination, and errors on those.

        Parameters
        ----------
        bin_1: int
            Index of first redshift bin. Note: bin indices start
            from 1.
        bin_2: int
            Index of second redshift bin. Note: bin indices start
            from 1.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "OU-LE3 Bin {:d} - Bin {:d}".format(bin_i, bin_j).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        no_bins: int
            Number of redshift bins for GC-phot probe.
        """
        if bin_1 > no_bins or bin_2 > no_bins:
            raise Exception('Reqeusted bin index greater than number of bins.')
        if bin_1 <= bin_2:
            bin_i = bin_1
            bin_j = bin_2
        else:
            bin_i = bin_2
            bin_j = bin_1
        ells = self.read_data.data_dict['GC-Phot']['ells']
        ext_cs = self.read_data.data_dict['GC-Phot'][
            'P{:d}-P{:d}'.format(bin_i, bin_j)]
        if pl_label is None:
            pl_label = "OU-LE3 Bin {:d} - Bin {:d}".format(bin_i, bin_j)
        pl_ax.plot(ells, ext_cs, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        cov_diags = np.sqrt(np.diagonal(self.read_data.data_dict[
                                        'GC-Phot']['cov']))
        counter = 0
        for i in range(1, no_bins + 1):
            for j in range(i, no_bins + 1):
                if i == bin_i and j == bin_j:
                    cur_index = counter
                counter += 1
        err_arr = []
        for mult in range(len(ells)):
            cur_err = cov_diags[(counter * mult) + cur_index]
            err_arr.append(cur_err)
        err_arr = np.array(err_arr)
        pl_ax.plot(ells, ext_cs + err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(ells, ext_cs - err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.fill_between(ells, ext_cs - err_arr, ext_cs + err_arr,
                           color=pl_colour, alpha=0.2)
        return pl_ax

    def plot_Cl_XC(self, ells, bin_WL, bin_GC, pl_ax, pl_label=None,
                   pl_colour='b', pl_linestyle='-'):
        """
        Function to plot WL Cls computed with the cosmobox code
        for a given set of ells, and bin combination.

        Parameters
        ----------
        ells: array
            Ell-modes at which to evaluate power spectra. Note: minimum allowed
            ell is 10.
        bin_WL: int
            Index of WL redshift bin. Note: bin indices start
            from 1.
        bin_GC: int
            Index of GC-phot redshift bin. Note: bin indices start
            from 1.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "WL Bin {:d} - GC-Phot Bin {:d}".format(bin_WL, bin_GC).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        """
        if ells[0] < 10.0:
            raise Exception('Minimum allowed ell-mode to plot is 10.0.')
        cl_arr = np.array([self.phot_ins.Cl_cross(cur_ell, bin_WL, bin_GC) for
                           cur_ell in ells])
        if pl_label is None:
            pl_label = "WL Bin {:d} - GC-Phot Bin {:d}".format(bin_WL, bin_GC)
        pl_ax.plot(ells, cl_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_Cl_XC(self, bin_WL, bin_GC, pl_ax, pl_label=None,
                            pl_colour='b', pl_linestyle='-', no_bins_WL=10,
                            no_bins_GC=10):
        """
        Plots external OU-LE3 XC-Phot power spectra, for given redshift bin
        combination, and errors on those.

        Parameters
        ----------
        bin_WL: int
            Index of WL redshift bin. Note: bin indices start
            from 1.
        bin_GC: int
            Index of GC redshift bin. Note: bin indices start
            from 1.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "OU-LE3 WL Bin {:d} - GC-Phot Bin {:d}".format(bin_WL,
            bin_GC).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        no_bins_WL: int
            Number of redshift bins for WL probe.
        no_bins_GC: int
            Number of redshift bins for GC-phot probe.
        """
        if bin_WL > no_bins_WL or bin_GC > no_bins_GC:
            raise Exception('Reqeusted bin index greater than number of bins.')
        ells = self.read_data.data_dict['XC-Phot']['ells']
        ext_cs = self.read_data.data_dict['XC-Phot'][
            'P{:d}-E{:d}'.format(bin_WL, bin_GC)]
        if pl_label is None:
            pl_label = "OU-LE3 WL Bin {:d} - GC-Phot Bin {:d}".format(bin_WL,
                                                                      bin_GC)
        pl_ax.plot(ells, ext_cs, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        cov_diags = np.sqrt(np.diagonal(self.read_data.data_dict['XC-Phot'][
                                        'cov_XC_only']))
        counter = no_bins_WL * no_bins_GC
        cur_index = 0
        for i in range(1, no_bins_WL + 1):
            for j in range(1, no_bins_GC + 1):
                if i == bin_WL and j == bin_GC:
                    break
                cur_index += 1
            if i == bin_WL and j == bin_GC:
                break
        err_arr = []
        for mult in range(len(ells)):
            cur_err = cov_diags[(counter * mult) + cur_index]
            err_arr.append(cur_err)
        err_arr = np.array(err_arr)
        pl_ax.plot(ells, ext_cs + err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(ells, ext_cs - err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.fill_between(ells, ext_cs - err_arr, ext_cs + err_arr,
                           color=pl_colour, alpha=0.2)
        return pl_ax

    def plot_GC_spec_multipole(self, redshift, ks, multipole_order, pl_ax,
                               pl_label=None, pl_colour='b', pl_linestyle='-'):
        """
        Plots GC-spec multipole spectra as calculated by the cosmobox, for a
        given redshift, set of ks, and multipole order.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate spectrum.
        ks: array
            Wavenumber values at which to evaluate spectrum.
        multipole_order: int
            Multipole order of spectrum to be evaluated. Note: Must be 0, 2, or
            4.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "l={:d}".format(multipole_order).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        """
        if multipole_order not in [0, 2, 4]:
            raise Exception('Multipole order must be 0, 2, or 4.')
        pk_arr = np.array([self.spec_ins.multipole_spectra(redshift, k_val,
                                                           multipole_order) for
                           k_val in ks])
        if pl_label is None:
            pl_label = "l={:d}".format(multipole_order)
        pl_ax.plot(ks, pk_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_GC_spec(self, redshift, multipole_order, pl_ax,
                              pl_label=None, pl_colour='b', pl_linestyle='-'):
        """
        Plots GC-spec multipole spectra from OU-LE3 files, for a
        given redshift, and multipole order.

        Parameters
        ----------
        redshift: str
            Redshift at which to evaluate spectrum.
        multipole_order: int
            Multipole order of spectrum to be evaluated. Note: Must be 0, 2, or
            4.
        pl_ax: matplotlib axis object
            Axis object within which to carry out plotting.
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to "OU-LE3 l={:d}".format(multipole_order).
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue.
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Deafault is '-'.
        """
        if multipole_order not in [0, 2, 4]:
            raise Exception('Multipole order must be 0, 2, or 4.')
        cov = self.read_data.data_dict['GC-Spec'][redshift]['cov']
        samp_ks = self.read_data.data_dict['GC-Spec'][redshift]['k_pk']
        errs = np.sqrt(np.diagonal(cov))
        nk = len(samp_ks)
        if multipole_order == 0:
            pk = self.read_data.data_dict['GC-Spec'][redshift]['pk0']
            epk = errs[0:nk]
        elif multipole_order == 2:
            pk = self.read_data.data_dict['GC-Spec'][redshift]['pk2']
            epk = errs[nk:2 * nk]
        elif multipole_order == 4:
            pk = self.read_data.data_dict['GC-Spec'][redshift]['pk4']
            epk = errs[2 * nk:]
        if pl_label is None:
            pl_label = "OU-LE3 l={:d}".format(multipole_order)
        pl_ax.plot(samp_ks, pk, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(samp_ks, pk + epk, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(samp_ks, pk - epk, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.fill_between(samp_ks, pk - epk, pk + epk, color=pl_colour,
                           alpha=0.2)
        return pl_ax
