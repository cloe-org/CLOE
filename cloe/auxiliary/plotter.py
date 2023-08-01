"""PLOTTER

Contains class to plot cosmological observables.
"""

import numpy as np
import matplotlib.pyplot as plt
from cloe.auxiliary.run_method import run_is_interactive
from cloe.auxiliary.plotter_default import default_settings
from cloe.photometric_survey.photo import Photo
from cloe.spectroscopic_survey.spectro import Spectro
from cloe.data_reader.reader import Reader
from matplotlib import rc
rc('font', **{'family': 'serif',
              'serif': ['Times']})
rc('text', usetex=True)


class PlotError(Exception):
    """
    Class to define Exception Error.
    """

    pass


class Plotter:
    """
    Class to plot observables (angular power spectra and multipoles).
    """

    def __init__(self, cosmo_dic, data, settings=default_settings):
        """
        Constructor of class :obj:`Plotter`.

        Parameters
        ----------
        cosmo_dic: dict
            Cosmological dictionary in structure specified within
            :obj:`cosmo.cosmology`
        data: dict
            Dictionary containing specifications for data loading and handling
        settings: dict
            Dictionary containing settings for plotting observables
        """
        self.cosmo_dic = cosmo_dic
        self.read_data = Reader(data)
        self.read_data.compute_nz()
        self.read_data.read_GC_spectro()
        self.zkeys = self.read_data.data_dict['GC-Spectro'].keys()
        self.read_data.read_phot()

        self.phot_ins = Photo(cosmo_dic, self.read_data.nz_dict_WL,
                              self.read_data.nz_dict_GC_Phot)
        self.spec_ins = Spectro(cosmo_dic, list(self.zkeys))

        self.binX = settings['binX']
        self.binY = settings['binY']
        self.lmin = settings['lmin']
        self.lmax = settings['lmax']
        self.nl = settings['nl']
        self.ells = self._set_binning(self.lmin, self.lmax, self.nl,
                                      settings['ltype'])
        self.pcols = settings['photo_colours']
        self.pls = settings['photo_linestyle']

        self.redshift = settings['redshift']
        self.kmin = settings['kmin']
        self.kmax = settings['kmax']
        self.nk = settings['nk']
        self.ks = self._set_binning(self.kmin, self.kmax, self.nk,
                                    settings['ktype'])
        self.scols = settings['spectro_colours']
        self.sls = settings['spectro_linestyle']

        self.path = settings['path']
        self.file_WL = settings['file_WL']
        self.file_GCphot = settings['file_GCphot']
        self.file_XC = settings['file_XC']
        self.file_GCspec = settings['file_GCspec']

    def _set_binning(self, xmin, xmax, nx, btype):
        r"""Sets bins according to edges, number of samples and bin type.

        Parameters
        ----------
        xmin: float
            The left edge of the bins
        xmax: float
            The right edge of the bins
        nx: int
            Number of bins
        btype: str
            Binning type (can be either 'lin' or 'log')

        Returns
        -------
        binning: numpy.ndarray
            The linear/logarithmic sampling in the range [xmin,xmax]
            with nx number of samples

        Raises
        ------
        ValueError
            If btype is neither 'lin' or 'log'
        """
        if btype == 'lin':
            return np.linspace(xmin, xmax, nx)
        elif btype == 'log':
            return np.logspace(np.log10(xmin), np.log10(xmax), nx)
        else:
            raise ValueError('Bin type has to be chosen from '
                             '[\'lin\', \'log\']')

    def plot_Cl_phot(self, ells, bin_i, bin_j, pl_ax, probe='WL',
                     pl_label=None, pl_colour='b', pl_linestyle='-',
                     no_bins=10):
        """Plots photometric angular power spectra.

        Function to plot given photometric Cls computed with CLOE
        for a given set of ells, and bin combination.

        Parameters
        ----------
        ells: numpy.ndarray
            Ell-modes at which to evaluate power spectra. Note: minimum allowed
            ell is 10, and maximum is 5000
        bin_i: int
            Index of first redshift bin. Note: bin indices start
            from 1
        bin_j: int
            Index of second redshift bin. Note: bin indices start
            from 1
        pl_ax: :obj:`matplotlib.axes.Axes`
            Matplotlib axes object within which to carry out plotting
        probe: str
            Which photometric probe to plot power spectra for. Either
            'WL' for weak lensing or 'GC-Phot' for photometric galaxy
            clustering. Set to 'WL' by default
            Note: :obj:`Plotter.plot_Cl_XC` must be used for the
            cross-correlation
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to `'Bin {:d} - Bin {:d}'.format(bin_i, bin_j)`
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Default is '-'
        no_bins: int
            Number of redshift bins for chosen probe
        """
        if probe not in ['WL', 'GC-Phot']:
            raise Exception('Must choose valid type of probe: WL, or GC-Phot.')
        if bin_i > no_bins or bin_j > no_bins:
            raise Exception('Requested bin index greater than number of bins.')
        # Note: As the range of ell-modes from OU-LE3 is not set in
        # stone, if this changes, the following exception handling will need to
        # be reviewed.
        if np.min(ells) < 10.0 or np.max(ells) > 5000.0:
            raise Exception('ell-modes must be between 10 and 5000.')
        if probe == 'WL':
            c_func = self.phot_ins.Cl_WL
        elif probe == 'GC-Phot':
            c_func = self.phot_ins.Cl_GC_phot
        cl_arr = c_func(ells, bin_i, bin_j)
        if pl_label is None:
            pl_label = "Bin {:d} - Bin {:d}".format(bin_i, bin_j)
        pl_ax.plot(ells, cl_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_Cl_phot(self, bin_m, bin_n, pl_ax, probe='WL',
                              pl_label=None, pl_colour='b', pl_linestyle='-',
                              no_bins=10):
        """Plots external photometric angular power spectra.

        Plots external OU-LE3 angular power spectra for a stated individual
        probe, for given redshift bin combination, and errors on those.

        Parameters
        ----------
        bin_m: int
            Index of first redshift bin. Note: bin indices start
            from 1
        bin_n: int
            Index of second redshift bin. Note: bin indices start
            from 1
        pl_ax: :obj:`matplotlib.axes.Axes`
            Axis object within which to carry out plotting
        probe: str
            Which photometric probe to plot power spectra for. Must be either
        `WL` for weak lensing or `GC-Phot` for photometric galaxy
            clustering. Set to `WL` by default
            Note: Plotter.plot_external_Cl_XC must be used for the
            cross-correlation
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to `'OU-LE3 Bin {:d} - Bin {:d}'`.format(bin_i, bin_j)`
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Default is '-'
        no_bins: int
            Number of redshift bins for WL probe
        """
        if probe not in ['WL', 'GC-Phot']:
            raise Exception('Must choose valid type of probe: WL, or GC-Phot.')
        if bin_m > no_bins or bin_n > no_bins:
            raise Exception('Requested bin index greater than number of bins.')
        if bin_m <= bin_n:
            bin_i = bin_m
            bin_j = bin_n
        else:
            bin_i = bin_n
            bin_j = bin_m
        if probe == 'WL':
            p_let = 'E'
        else:
            p_let = 'P'
        power_label = '{:s}{:d}-{:s}{:d}'.format(p_let, bin_i, p_let, bin_j)
        ells = self.read_data.data_dict[probe]['ells']
        ext_cs = self.read_data.data_dict[probe][power_label]
        if pl_label is None:
            pl_label = "OU-LE3 Bin {:d} - Bin {:d}".format(bin_i, bin_j)
        pl_ax.plot(ells, ext_cs, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        # Note:As covariance format is not set in stone, if the format
        # changes, the following code will need to be reviewed to correct the
        # error bar calculation.
        nbins_wl = self.read_data.num_bins_wl
        nbins_xc = self.read_data.num_bins_xcphot
        nells_wl = self.read_data.num_scales_wl
        nells_xc = self.read_data.num_scales_xcphot
        nells_gc = self.read_data.num_scales_gcphot

        cov_diags = np.sqrt(np.diagonal(self.read_data.data_dict['3x2pt_cov']))
        counter = 0
        for i in range(1, no_bins + 1):
            for j in range(i, no_bins + 1):
                if i == bin_i and j == bin_j:
                    cur_index = counter
                counter += 1
        if probe == 'WL':
            i_low = cur_index * nells_wl
            i_high = (cur_index + 1) * nells_wl
        else:
            i_low = nbins_wl * nells_wl + nbins_xc * nells_xc + \
                cur_index * nells_gc
            i_high = nbins_wl * nells_wl + nbins_xc * nells_xc + \
                (cur_index + 1) * nells_gc
        err_arr = cov_diags[i_low:i_high]

        pl_ax.plot(ells, ext_cs + err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(ells, ext_cs - err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.fill_between(ells, ext_cs - err_arr, ext_cs + err_arr,
                           color=pl_colour, alpha=0.2)
        return pl_ax

    def plot_Cl_XC(self, ells, bin_WL, bin_GC, pl_ax, pl_label=None,
                   pl_colour='b', pl_linestyle='-', no_bins_WL=10,
                   no_bins_GC=10):
        """Plots cross-correlation photometric angular power spectra.

        Function to plot XC Cls computed with CLOE
        for a given set of ells, and bin combination.

        Parameters
        ----------
        ells: numpy.ndarray
            Ell-modes at which to evaluate power spectra. Note: minimum allowed
            ell is 10, and maximum is 5000
        bin_WL: int
            Index of WL redshift bin. Note: bin indices start
            from 1
        bin_GC: int
            Index of GCphot redshift bin. Note: bin indices start
            from 1
        pl_ax: :obj:`matplotlib.axes.Axes`
            Axis object within which to carry out plotting
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to `'WL Bin {:d} - GC-Phot Bin {:d}'.format(bin_WL, bin_GC)`
        pl_colour: str
            Matplotlib colour choice for current plot. Default is `b` for blue
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Default is '-'
        no_bins_WL: int
            Number of redshift bins for WL probe
        no_bins_GC: int
            Number of redshift bins for GCphot probe
        """
        if bin_WL > no_bins_WL or bin_GC > no_bins_GC:
            raise Exception('Requested bin index greater than number of bins.')
        # Note: As the range of ell-modes from OU-LE3 is not set in
        # stone, if this changes, the following exception handling will need to
        # be reviewed.
        if np.min(ells) < 10.0 or np.max(ells) > 5000.0:
            raise Exception('ell-modes must be between 10 and 5000.')
        cl_arr = self.phot_ins.Cl_cross(ells, bin_WL, bin_GC)
        if pl_label is None:
            pl_label = "WL Bin {:d} - GC-Phot Bin {:d}".format(bin_WL, bin_GC)
        pl_ax.plot(ells, cl_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_Cl_XC(self, bin_WL, bin_GC, pl_ax, pl_label=None,
                            pl_colour='b', pl_linestyle='-', no_bins_WL=10,
                            no_bins_GC=10):
        """Plots external cross-correlation photometric angular power spectra.

        Plots external OU-LE3 XC-Phot power spectra, for given redshift bin
        combination, and errors on those.

        Parameters
        ----------
        bin_WL: int
            Index of WL redshift bin. Note: bin indices start
            from 1
        bin_GC: int
            Index of GC redshift bin. Note: bin indices start
            from 1
        pl_ax: :obj:`matplotlib.axes.Axes`
            Axis object within which to carry out plotting
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to `OU-LE3 WL Bin {:d} - GC-Phot Bin {:d}'.format(bin_WL,
            bin_GC)`
        pl_colour: str
            Matplotlib colour choice for current plot. Default is 'b' for blue
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Default is '-'
        no_bins_WL: int
            Number of redshift bins for WL probe
        no_bins_GC: int
            Number of redshift bins for GCphot probe
        """
        if bin_WL > no_bins_WL or bin_GC > no_bins_GC:
            raise Exception('Requested bin index greater than number of bins.')
        ells = self.read_data.data_dict['XC-Phot']['ells']
        ext_cs = self.read_data.data_dict['XC-Phot'][
            'P{:d}-E{:d}'.format(bin_GC, bin_WL)]
        if pl_label is None:
            pl_label = "OU-LE3 WL Bin {:d} - GC-Phot Bin {:d}".format(bin_WL,
                                                                      bin_GC)
        pl_ax.plot(ells, ext_cs, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        # Note: As the covariance format is not set in stone, if the format
        # changes, the following code will need to be reviewed to correct the
        # error bar calculation.
        nbins_wl = self.read_data.num_bins_wl
        nells_wl = self.read_data.num_scales_wl
        nells_xc = self.read_data.num_scales_xcphot

        cov_diags = np.sqrt(np.diagonal(self.read_data.data_dict['3x2pt_cov']))
        counter = 0
        for i in range(1, no_bins_WL + 1):
            for j in range(1, no_bins_GC + 1):
                if i == bin_WL and j == bin_GC:
                    cur_index = counter
                counter += 1
        i_low = nbins_wl * nells_wl + cur_index * nells_xc
        i_high = nbins_wl * nells_wl + (cur_index + 1) * nells_xc
        err_arr = cov_diags[i_low:i_high]

        pl_ax.plot(ells, ext_cs + err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.plot(ells, ext_cs - err_arr, color=pl_colour,
                   linestyle=pl_linestyle)
        pl_ax.fill_between(ells, ext_cs - err_arr, ext_cs + err_arr,
                           color=pl_colour, alpha=0.2)
        return pl_ax

    def plot_GC_spectro_multipole(self, redshift, ks, multipole_order, pl_ax,
                                  pl_label=None, pl_colour='b',
                                  pl_linestyle='-'):
        """Plots spectrocopic power spectra.

        Plots spectroscopic multipole spectra as calculated by CLOE,
        for a given redshift, set of wavenumbers, and multipole order.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate spectrum
        ks: numpy.ndarray
            Wavenumber values at which to evaluate spectrum
        multipole_order: int
            Multipole order of spectrum to be evaluated. Note: Must be 0, 2, or
            4
        pl_ax: :obj:`matplotlib.axes.Axes`
            Axis object within which to carry out plotting
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to `'l={:d}'.format(multipole_order)`
        pl_colour: str
            Matplotlib colour choice for current plot. Default is `b` for blue
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Default is '-'
        """
        # Note: These limits for k are set based on what is currently the
        # expected range for OU-LE3 data. Should this change, this range should
        # also be adjusted accordingly.
        if np.min(ks) < 0.001 or np.max(ks) > 0.5:
            raise Exception('ks must be between 0.001 and 0.5')
        if redshift > 1.8:
            raise Exception('Euclid maximum redshift for GC-spectro is 1.8.')
        if multipole_order not in [0, 2, 4]:
            raise Exception('Multipole order must be 0, 2, or 4.')
        pk_arr = np.array([self.spec_ins.multipole_spectra(
            redshift, k_val, ms=[multipole_order]) for k_val in ks])
        if pl_label is None:
            pl_label = "l={:d}".format(multipole_order)
        pl_ax.plot(ks, pk_arr, label=pl_label, color=pl_colour,
                   linestyle=pl_linestyle)
        return pl_ax

    def plot_external_GC_spectro(self, redshift, multipole_order, pl_ax,
                                 pl_label=None, pl_colour='b',
                                 pl_linestyle='-'):
        """Plots external spectroscopic power spectra.

        Plots spectroscopic multipole spectra from OU-LE3 files,
        for a given redshift, and multipole order, with errors.

        Parameters
        ----------
        redshift: str
            Redshift at which to evaluate spectrum. Note: Here, this must be
            specified as string
        multipole_order: int
            Multipole order of spectrum to be evaluated. Note: Must be 0, 2, or
            4
        pl_ax: :obj:`matplotlib.axes.Axes`
            Axis object within which to carry out plotting
        pl_label: str
            Label for plot to appear in legend. If none is given, this will
            be set to `'OU-LE3 l={:d}'.format(multipole_order)`
        pl_colour: str
            Matplotlib colour choice for current plot. Default is `b` for blue
        pl_linestyle: str
            Matplotlib linestyle choice for current plot. Default is '-'
        """
        if float(redshift) > 1.8:
            raise Exception('Euclid maximum redshift for GC-spectro is 1.8.')
        if multipole_order not in [0, 2, 4]:
            raise Exception('Multipole order must be 0, 2, or 4.')
        # Note: The format for the GC-Spectro Covariance matrix is also not
        # completely fixed yet. If the format changes, this section of code
        # should be reviewed to ensure it correctly extracts the error bars.
        cov = self.read_data.data_dict['GC-Spectro'][redshift]['cov']
        samp_ks = self.read_data.data_dict['GC-Spectro'][redshift]['k_pk']
        errs = np.sqrt(np.diagonal(cov))
        nk = len(samp_ks)
        if multipole_order == 0:
            pk = self.read_data.data_dict['GC-Spectro'][redshift]['pk0']
            epk = errs[0:nk]
        elif multipole_order == 2:
            pk = self.read_data.data_dict['GC-Spectro'][redshift]['pk2']
            epk = errs[nk:2 * nk]
        elif multipole_order == 4:
            pk = self.read_data.data_dict['GC-Spectro'][redshift]['pk4']
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

    def output_Cl_WL(self):
        r"""Plots weak lensing observable.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax = self.plot_external_Cl_phot(self.binX, self.binY, ax,
                                        probe='WL', pl_label='Benchmark')
        ax = self.plot_Cl_phot(self.ells, self.binX, self.binY, ax,
                               probe='WL', pl_colour=self.pcols,
                               pl_linestyle=self.pls)
        self._set_ax(ax, title=f'Weak-Lensing bins {self.binX}-{self.binY}',
                     xlabel=r'$\ell$', ylabel=r'$C_\ell$ $[\mathrm{sr}^{-1}]$',
                     fontsize=20, xscale='log', yscale='log')
        ax.legend()
        plt.tight_layout()

        filename = self.path + self.file_WL
        if run_is_interactive():
            plt.show()
        else:
            plt.savefig(f'{filename}.png', dpi=300)

        c_func = self.phot_ins.Cl_WL
        cl_arr = np.array([c_func(cur_ell, self.binX, self.binY)
                           for cur_ell in self.ells])
        np.savetxt(f'{filename}.dat', list(zip(self.ells, cl_arr)),
                   fmt='%.12e', delimiter='\t', newline='\n',
                   header=f'{"ell" : >16}{"Cell[sr^(-1)]" : >24}')

    def output_Cl_phot(self):
        r"""Plots photometric clustering observable.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax = self.plot_external_Cl_phot(self.binX, self.binY, ax,
                                        probe='GC-Phot', pl_label='Benchmark')
        ax = self.plot_Cl_phot(self.ells, self.binX, self.binY, ax,
                               probe='GC-Phot', pl_colour=self.pcols,
                               pl_linestyle=self.pls)
        self._set_ax(ax, title=f'GC-photo bins {self.binX}-{self.binY}',
                     xlabel=r'$\ell$', ylabel=r'$C_\ell$ $[\mathrm{sr}^{-1}]$',
                     fontsize=20, xscale='log', yscale='log')
        ax.legend()
        plt.tight_layout()

        filename = self.path + self.file_GCphot
        if run_is_interactive():
            plt.show()
        else:
            plt.savefig(f'{filename}.png', dpi=300)

        c_func = self.phot_ins.Cl_GC_phot
        cl_arr = np.array([c_func(cur_ell, self.binX, self.binY)
                           for cur_ell in self.ells])
        np.savetxt('%s.dat' % (filename), list(zip(self.ells, cl_arr)),
                   fmt='%.12e', delimiter='\t', newline='\n',
                   header=f'{"ell" : >16}{"Cell[sr^(-1)]" : >24}')

    def output_Cl_XC(self):
        r"""Plots photometric cross-correlation observable.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax = self.plot_external_Cl_XC(self.binX, self.binY, ax,
                                      pl_label='Benchmark')
        ax = self.plot_Cl_XC(self.ells, self.binX, self.binY, ax,
                             pl_colour=self.pcols,
                             pl_linestyle=self.pls)
        self._set_ax(ax, title=f'WL x GC-photo bins {self.binX}-{self.binY}',
                     xlabel=r'$\ell$', ylabel=r'$C_\ell$ $[\mathrm{sr}^{-1}]$',
                     fontsize=20, xscale='log')
        ax.legend()
        plt.tight_layout()

        filename = self.path + self.file_XC
        if run_is_interactive():
            plt.show()
        else:
            plt.savefig(f'{filename}.png', dpi=300)

        cl_arr = np.array([self.phot_ins.Cl_cross(cur_ell, self.binX,
                                                  self.binY)
                           for cur_ell in self.ells])
        np.savetxt(f'{filename}.dat', list(zip(self.ells, cl_arr)),
                   fmt='%.12e', delimiter='\t', newline='\n',
                   header=f'{"ell" : >16}{"Cell[sr^(-1)]" : >24}')

    def output_GC_spectro(self):
        r"""Plots spectroscopic galaxy clustering observable.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if self.redshift in [1.0, 1.2, 1.4, 1.65]:
            ax = self.plot_external_GC_spectro(str(self.redshift), 0, ax,
                                               pl_colour='blue')
            ax = self.plot_external_GC_spectro(str(self.redshift), 2, ax,
                                               pl_colour='royalblue')
            ax = self.plot_external_GC_spectro(str(self.redshift), 4, ax,
                                               pl_colour='dodgerblue')
        ax = self.plot_GC_spectro_multipole(self.redshift, self.ks, 0, ax,
                                            pl_colour=self.scols[0],
                                            pl_linestyle=self.sls)
        ax = self.plot_GC_spectro_multipole(self.redshift, self.ks, 2, ax,
                                            pl_colour=self.scols[1],
                                            pl_linestyle=self.sls)
        ax = self.plot_GC_spectro_multipole(self.redshift, self.ks, 4, ax,
                                            pl_colour=self.scols[2],
                                            pl_linestyle=self.sls)
        self._set_ax(ax, title=f'GC-spectro $z={self.redshift}$',
                     xlabel=r'$k$ $[\mathrm{Mpc}^{-1}]$',
                     ylabel=r'$P_\ell$ $[\mathrm{Mpc}^3]$',
                     fontsize=20, xscale='log', yscale='log')
        l1, = ax.plot([], [], color=self.scols[0])
        l2, = ax.plot([], [], color=self.scols[1])
        l3, = ax.plot([], [], color=self.scols[2])
        ax.legend([l1, l2, l3],
                  ['$l=0$', '$l=2$', '$l=4$'])
        plt.tight_layout()

        filename = self.path + self.file_GCspec
        if run_is_interactive():
            plt.show()
        else:
            plt.savefig(f'{filename}.png', dpi=300)

        pk0 = np.array([self.spec_ins.multipole_spectra(self.redshift, k_val,
                                                        [0])
                        for k_val in self.ks])
        pk2 = np.array([self.spec_ins.multipole_spectra(self.redshift, k_val,
                                                        [2])
                        for k_val in self.ks])
        pk4 = np.array([self.spec_ins.multipole_spectra(self.redshift, k_val,
                                                        [4])
                        for k_val in self.ks])

        np.savetxt(f'{filename}.dat', list(zip(self.ks, pk0, pk2, pk4)),
                   fmt='%.12e', delimiter='\t', newline='\n',
                   header=(
                    f'{"k[Mpc^(-1)]" : >16}'
                    f'{"P0[Mpc^3]" : >24}'
                    f'{"P2[Mpc^3]" : >24}'
                    f'{"P4[Mpc^3]" : >24}'
                          ))

    def _set_ax(self, axes, title, xlabel, ylabel,
                fontsize=20, xscale='linear', yscale='linear'):
        r"""Sets title, axis labels, fontsize, scales of a given axes object.

        Parameters
        ----------
        axes: :obj:`matplotlib.axes.Axes`
           Instance on which the settings are applied
        title: str
            Title of the axes object
        xlabel: str
            Label of the x-axis
        ylabel: str
            Label of the y-axis
        fontsize: str, optional
            Fontsize for title and axis labels
        xscale: str, optional
            Scale of the x-axis
        yscale: str, optional
            Scale of the y-axis
        """
        axes.set_xlabel(xlabel, fontsize=fontsize)
        axes.set_ylabel(ylabel, fontsize=fontsize)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_title(title, fontsize=fontsize)
