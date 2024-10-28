"""
EFT MODULE

This module computes several nonlinear spectra using the `PBJ code`
for the Joint analysis of Power Spectrum and Bispectrum,
adapted from arXiv:2207.14784 / 2306.09275, ported to CLOE by IST:NL.
"""

import numpy as np
from scipy.fftpack import dst, idst
from scipy.integrate import simps
from scipy.interpolate import splrep, splev, RectBivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.special import spherical_jn
from scipy.special import legendre
from scipy.signal import fftconvolve
from collections import OrderedDict
import fastpt.FASTPT_simple as fpts


class EFTofLSS:
    r"""Effective field theory of large scale structures.

    Class to construct the `EFTofLSS` model for the nonlinear anisotropic
    galaxy power spectrum. For this recipe we consider the standard
    assumption that galaxy clustering can be modelled starting from
    fluctuations of the cold dark matter and baryon density field,
    regardless of the presence of massive neutrinos.
    """

    def __init__(self, cosmo_dic, log10k_min=-4, log10k_max=1.7, nk_tot=1000):

        self.ks = np.logspace(log10k_min, log10k_max, nk_tot, endpoint=True)

        self.z = cosmo_dic['z_win']
        self.ns = cosmo_dic['ns']
        self.As = cosmo_dic['As']
        self.h = cosmo_dic['H0'] / 100.0
        self.Obh2 = cosmo_dic['ombh2']
        self.Omh2 = cosmo_dic['omch2'] + self.Obh2
        self.tau = cosmo_dic['tau']

        self.linear_power_func = cosmo_dic['Pk_cb']
        if cosmo_dic['use_gamma_MG']:
            self.growth_factor_func = cosmo_dic['D_z_k_func_MG']
        else:
            self.growth_factor_func = cosmo_dic['D_z_k_func']
        self.growth_rate_func = cosmo_dic['f_z']

        self.fastpt = FASTPTPlus(self.ks, -2, low_extrap=-6, high_extrap=5,
                                 n_pad=1000)
        self.use_gamma_MG = cosmo_dic['use_gamma_MG']
        self.IR_type_in_grid = None

    def CallEH_NW(self, redshift):
        r"""Calls ``Eisenstein-Hu_NW``.

        Computes the smooth matter power spectrum at a given redshift following
        the prescription of Eisenstein & Hu (1998).

        Parameters
        ----------
        redshift: float
            Redshift

        Returns
        -------
        Eisenstein-Hu fit power spectrum: numpy.ndarray
            Eisenstein-Hu fit for the matter power spectrum
            computed on a log-spaced k-grid, in
            :math:`(1/Mpc)^3` units
        """

        Tcmb = 2.726
        kL = self.ks
        s = 44.5 * np.log(9.83 / self.Omh2) / np.sqrt(1.0 + 10.0 *
                                                      (self.Obh2)**0.75)
        Gamma = self.Omh2 / self.h
        AG = (1.0 - 0.328 * np.log(431.0 * self.Omh2) * self.Obh2 / self.Omh2 +
              0.38 * np.log(22.3 * self.Omh2) * (self.Obh2 / self.Omh2)**2)
        Gamma = Gamma * (AG + (1.0 - AG) / (1.0 + (0.43 * kL * s)**4))
        Theta = Tcmb / 2.7
        q = kL * Theta**2 / Gamma / self.h
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        T0 = L0 / (L0 + C0 * q * q)
        T0 /= T0[0]
        P_EH = kL**self.ns * T0**2
        P_EH *= self.linear_power_func.P(redshift, self.ks)[0] / P_EH[0]

        return P_EH

    def _gaussianFiltering(self, lamb, redshift):
        def extrapolateX(x):
            logx = np.log10(x)
            xSize = x.size
            dlogx = np.log10(x[1] / x[0])
            newlogx = np.linspace(logx[0] - xSize * dlogx / 2,
                                  logx[-1] + xSize * dlogx / 2, 2 * xSize)
            newx = 10**newlogx
            return newx

        def extrapolateFx(x, Fx):
            newx = extrapolateX(x)
            backB = np.log(Fx[1] / Fx[0]) / np.log(x[1] / x[0])
            backA = Fx[0] / x[0]**backB
            forwB = np.log(Fx[-1] / Fx[-2]) / np.log(x[-1] / x[-2])
            forwA = Fx[-1] / x[-1]**forwB
            backFx = backA * newx[:x.size // 2]**backB
            forwFx = forwA * newx[3 * x.size // 2:]**forwB
            return np.hstack((backFx, Fx, forwFx))

        kNumber = len(self.ks)
        qlog = np.log10(self.ks)
        dqlog = qlog[1] - qlog[0]
        qlog = np.log10(extrapolateX(self.ks))
        pLin = extrapolateFx(self.ks, self.linear_power_func.P(redshift,
                                                               self.ks))
        pEH = self.CallEH_NW(redshift)
        pEH = extrapolateFx(self.ks, pEH)
        smoothPowerSpectrum = gaussian_filter1d(pLin / pEH, lamb / dqlog) * pEH

        return smoothPowerSpectrum[kNumber // 2:kNumber // 2 + kNumber]

    def pksmooth_dst(self, redshift):
        r"""pksmooth_dst.

        Computes the no-wiggle power spectrum by identifying and removing the
        BAO bump in configuration space by means of a type-II dst transform,
        then transforming back to Fourier space. Based on arXiv:1712.08067,
        arXiv:1906.02742, adapted from the baccoemu implementation.

        Parameters
        ----------
        redshift: float
            Redshift

        Returns
        -------
        smooth_itp: interpolator object
            No-wiggle power spectrum at the input redshift
        """

        # Sample k, P(k) in 2^15 points
        nk = int(2**15)
        kmin = 1e-4
        kmax = 10
        klin = np.linspace(kmin, kmax, nk)
        pklin = self.linear_power_func.P(redshift, klin)

        # DST-II log(k *P(k))
        logkpk = np.log10(klin * pklin)
        dstpk = dst(logkpk, type=2)

        # Split in even and odd indices,
        even = dstpk[0::2]  # array with even indexes
        odd = dstpk[1::2]  # array with odd indexes
        i_even = np.arange(len(even)).astype(int)
        i_odd = np.arange(len(odd)).astype(int)
        even_cs = splrep(i_even, even, s=0)
        odd_cs = splrep(i_odd, odd, s=0)

        # Compute second derivatives and interpolate with cubic splines
        even_2nd_der = splev(i_even, even_cs, der=2, ext=0)
        odd_2nd_der = splev(i_odd, odd_cs, der=2, ext=0)

        # Rescale the fixed ranges for the BAO bump based on omega_m
        range_imin = np.array([150, 300])
        range_imax = np.array([300, 600])
        range_imin = (range_imin * (0.1376591 / self.Omh2)**0.25).astype(int)
        range_imax = (range_imax * (0.1376591 / self.Omh2)**0.25).astype(int)
        imin_start = np.argmax(even_2nd_der[range_imin[0]:range_imin[1]]) + \
            range_imin[0] - 30
        imax_start = np.where(
            even_2nd_der[range_imax[0]:range_imax[1]] < 0.04)[0][0] + \
            range_imax[0]

        # Cut out the BAOs
        # mask indices
        i_even_holed = np.concatenate((i_even[:imin_start],
                                       i_even[imax_start:]))
        i_odd_holed = np.concatenate((i_odd[:imin_start], i_odd[imax_start:]))
        # mask log(k*P(k))
        even_holed = np.concatenate((even[:imin_start], even[imax_start:]))
        odd_holed = np.concatenate((odd[:imin_start], odd[imax_start:]))

        # Interpolate the arrays rescaled by (i+1)^2 with cubic splines
        even_holed_cs = \
            splrep(i_even_holed, even_holed * (i_even_holed + 1)**2, s=0)
        odd_holed_cs = \
            splrep(i_odd_holed, odd_holed * (i_odd_holed + 1)**2, s=0)

        # Merge the two arrays containing the respective elements without BAO,
        # and without the rescaling factor of (i + 1)^2, and inversely FST
        even_smooth = splev(i_even, even_holed_cs, der=0, ext=0) / \
            (i_even + 1)**2
        odd_smooth = splev(i_odd, odd_holed_cs, der=0, ext=0) / (i_odd + 1)**2

        dstkpk_smooth = np.zeros(nk)
        dstkpk_smooth[0::2] = even_smooth
        dstkpk_smooth[1::2] = odd_smooth
        pksmooth = idst(dstkpk_smooth, type=2) / (2 * len(dstkpk_smooth))
        pksmooth = 10**(pksmooth) / klin

        k_low = klin[np.where(klin <= 5)]
        k_high = self.ks[np.where(self.ks > 5)]
        pksmooth_low = pksmooth[np.where(klin <= 5)]
        pksmooth_high = self.linear_power_func.P(redshift, k_high)

        ktot = np.concatenate((k_low, k_high))
        pksmooth_tot = np.concatenate((pksmooth_low, pksmooth_high))

        pksmooth_spl = splrep(np.log(ktot), np.log(pksmooth_tot), s=0)
        smooth_itp = np.exp(splev(np.log(self.ks), pksmooth_spl, der=0, ext=0))

        return smooth_itp

    def IRresum(self, IRres='DST', redshift=0.0, lamb=0.25, kS=0.2,
                lOsc=102.707):
        r"""Function for IR-resummation.

        Splits the linear power spectrum into a smooth and a wiggly part,
        and computes the damping factors.

        Parameters
        ----------
        IRres: string
            Algorithm for infra-red resummation. Options include 'EH' for
            Eisenstein-Hu, 'DST' for discrete sine transform (default 'DST')
        redshift: string
            Redshift (default 0.0)
        lamb: float
            Width of Gaussian filter (default 0.25)
        kS: float
            Scale of separation between large and
            small scales (default 0.2)
        lOsc: float
            Scale of the BAO (default to 102.707)

        Returns
        -------
        Pnw, Pw, Sigma2, dSigma2: numpy.ndarray, numpy.ndarray, float, float
           No-wiggle power spectrum, Wiggle power spectrum,
           Damping factor for IR-resummation,
           :math:`d\Sigma` for IR-resummation
        """
        lOsc /= self.h
        kS *= self.h

        # E&H + gaussian filtering
        if IRres == 'EH':
            Pnw = self._gaussianFiltering(lamb=lamb, redshift=redshift)
        # Discrete Sine Transform
        elif IRres == 'DST':
            Pnw = self.pksmooth_dst(redshift=redshift)

        # Wiggle-smooth split
        Pw = self.linear_power_func.P(redshift, self.ks) - Pnw

        # Sigma2 as integral up to 0.2;
        # Uses Simpson integration (no extreme accuracy needed)
        icut = (self.ks <= kS)
        kLcut = self.ks[icut]
        Pnwcut = Pnw[icut]
        kosc = 1.0 / lOsc
        norm = 1.0 / (6.0 * np.pi**2)
        Sigma2 = norm * simps(Pnwcut * (1.0 - spherical_jn(0, kLcut / kosc) +
                                        2.0 * spherical_jn(2, kLcut / kosc)),
                              x=kLcut)
        dSigma2 = norm * simps(3.0 * Pnwcut * spherical_jn(2, kLcut / kosc),
                               x=kLcut)

        return Pnw, Pw, Sigma2, dSigma2

    def _muxdamp(self, mu, Sigma2, dSigma2, f):
        r"""Muxdamp

        Computes the full damping factor for IR-resummation in redshift space
        and the (mu^n * exponential damping)
        """
        Sig2mu = Sigma2 + f * mu**2 * (2.0 * Sigma2 + f *
                                       (Sigma2 + dSigma2 * (mu**2 - 1.0)))
        RSDdamp = np.exp(-self.ks[:, np.newaxis]**2 * Sig2mu)
        return Sig2mu, RSDdamp

    def _Pgg_kmu_terms(self, redshift, IRres='DST'):
        r"""Pgg_kmu_terms

        Computes the terms for the loop corrections at redshift z=0, splits
        into wiggle and no-wiggle and stores them as attributes of the class

        Parameters
        ----------
        redshift: float
            Redshift
        IRres: string
            Algorithm for infra-red resummation. Options include 'EH' for
            Eisenstein-Hu, 'DST' for discrete sine transform (default 'DST')
        """
        self.PL = self.linear_power_func.P(redshift, self.ks)
        self.Pnw, self.Pw, self.Sigma2, self.dSigma2 = \
            self.IRresum(IRres=IRres, redshift=redshift)

        # Loops on P_L and Pnw
        loop22_L = self.fastpt.Pkmu_22_one_loop_terms(self.ks, self.PL,
                                                      C_window=0.75)
        loop22_nw = self.fastpt.Pkmu_22_one_loop_terms(self.ks, self.Pnw,
                                                       C_window=0.75)
        loop13_L = self.fastpt.Pkmu_13_one_loop_terms(self.ks, self.PL)
        loop13_nw = self.fastpt.Pkmu_13_one_loop_terms(self.ks, self.Pnw)
        setattr(self, 'loop22_nw', loop22_nw)
        setattr(self, 'loop13_nw', loop13_nw)

        # Compute wiggle
        loop22_w = np.array([i - j for i, j in zip(loop22_L, loop22_nw)])
        loop13_w = np.array([i - j for i, j in zip(loop13_L, loop13_nw)])
        setattr(self, 'loop22_w', loop22_w)
        setattr(self, 'loop13_w', loop13_w)

    def _Pgg_k_terms_L(self):
        r"""Pgg_k_terms_L

        Computes the terms for the real space loop corrections at redshift z=0,
        with the linear power spectrum and stores them as attributes of the
        class.
        """

        PL = self.linear_power_func.P(0.0, self.ks)

        # Loops on P_L
        loop22_L = self.fastpt.Pk_22_one_loop_terms_realspace(PL,
                                                              C_window=0.75)
        loop13_L = self.fastpt.Pkmu_13_one_loop_terms(self.ks, PL)
        setattr(self, 'loop22_L', loop22_L)
        setattr(self, 'loop13_L', loop13_L)

    def P_kmu_z(self, redshift, use_growth_rescaling=False, IRres='DST',
                **kwargs):
        r"""P_kmu_z.

        Computes the nonlinear galaxy power spectrum in redshift space at the
        proper redshift specified by the growth parameters.

        Parameters
        ----------
        redshift: float
            Redshift
        use_growth_rescaling: bool
            Keyword to select rescaling of power spectrum with linear growth
            rate or computation at the requested redshift (default False)
        IRres: string
            Algorithm for infra-red resummation. Options include 'EH' for
            Eisenstein-Hu, 'DST' for discrete sine transform (default 'DST')

        **kwargs: Dictionary containing the model parameters
        b1: float
            Linear bias parameter
        b2: float
            Quadratic local bias parameter
        bG2: float
            Quadratic nonlical bias parameter
        bG3: float
            Cubic nonlocal bias parameter
        c0: float
            EFT counterterm
        c2: float
            EFT counterterm
        c4: float
            EFT counterterm
        ck4: float
            EFT counterterm
        aP: float
            Shot-noise parameter
        e0k2: float
            Shot-noise parameter, scale-dependent
        e2k2:
            Shot-noise parameter, scale- and mu-dependent
        Psn: float
            Poisson shot-noise

        Returns
        -------
        Nonlinear galaxy power spectrum: object
            Scipy Interpolator
        """

        if use_growth_rescaling:
            if self.IR_type_in_grid is None or self.IR_type_in_grid != IRres:
                self._Pgg_kmu_terms(redshift=0.0, IRres=IRres)
                self.IR_type_in_grid = IRres
            if self.use_gamma_MG:
                D = self.growth_factor_func(redshift)
            else:
                D = self.growth_factor_func(redshift, 0.05)
        else:
            self._Pgg_kmu_terms(redshift=redshift, IRres=IRres)
            self.IR_type_in_grid = None
            D = 1.0

        D2 = D**2
        D4 = D**4

        f = self.growth_rate_func(redshift)

        mu = np.linspace(-1, 1, 51).reshape(1, 51)

        # Rescaling of Sigma and dSigma2
        Sigma2 = self.Sigma2 * D2
        dSigma2 = self.dSigma2 * D2

        Sig2mu, RSDdamp = self._muxdamp(mu, Sigma2, dSigma2, f)

        # Rescale and reshape the wiggle and no-wiggle P(k)
        Pnw = self.Pnw
        Pw = self.Pw
        Pnw_sub = D2 * Pnw[:, np.newaxis]
        Pw_sub = D2 * Pw[:, np.newaxis]
        ks_sub = self.ks[:, np.newaxis]

        # Compute loop nw and rescale to D**4
        loop22_nw_sub = np.array([self.loop22_nw[i] for i in
                                  range(len(self.loop22_nw))]) * D4
        loop13_nw_sub = np.array([self.loop13_nw[i] for i in
                                  range(len(self.loop13_nw))]) * D4
        loop22_w = self.loop22_w * D4
        loop13_w = self.loop13_w * D4

        # Bias parameters
        b1 = kwargs['b1'] if 'b1' in kwargs.keys() else 1.0
        b2 = kwargs['b2'] if 'b2' in kwargs.keys() else 0.0
        bG2 = kwargs['bG2'] if 'bG2' in kwargs.keys() else 0.0
        bG3 = kwargs['bG3'] if 'bG3' in kwargs.keys() else 0.0
        c0 = kwargs['c0'] if 'c0' in kwargs.keys() else 0.0
        c2 = kwargs['c2'] if 'c2' in kwargs.keys() else 0.0
        c4 = kwargs['c4'] if 'c4' in kwargs.keys() else 0.0
        ck4 = kwargs['ck4'] if 'ck4' in kwargs.keys() else 0.0

        def Kaiser(b1, f, mu):
            return b1 + f * mu**2

        # Next-to-leading order and counterterm. Noise corrections are computed
        # in the relative power spectrum model module
        PNLO = Kaiser(b1, f, mu)**2 * (Pnw_sub + RSDdamp * Pw_sub *
                                       (1.0 + ks_sub**2 * Sig2mu))
        Pkmu_ctr = (-2.0 * (c0 + c2 * f * mu**2 + c4 * f * f * mu**4) *
                    ks_sub**2 + ck4 * f**4 * mu**4 * Kaiser(b1, f, mu)**2 *
                    ks_sub**4) * (Pnw_sub + RSDdamp * Pw_sub)

        # Biases
        bias22 = np.array([b1**2 * mu**0, b1 * b2 * mu**0, b1 * bG2 * mu**0,
                           b2**2 * mu**0, b2 * bG2 * mu**0, bG2**2 * mu**0,
                           mu**2 * f * b1, mu**2 * f * b2, mu**2 * f * bG2,
                           (mu * f * b1)**2, (mu * b1)**2 * f,
                           mu**2 * f * b1 * b2, mu**2 * f * b1 * bG2,
                           (mu * f)**2 * b1, (mu * f)**2 * b2,
                           (mu * f)**2 * bG2, (mu * f)**4, mu**4 * f**3,
                           mu**4 * f**3 * b1, mu**4 * f**2 * b2,
                           mu**4 * f**2 * bG2, mu**4 * f**2 * b1,
                           mu**4 * f**2 * b1**2, mu**4 * f**2,
                           mu**6 * f**4, mu**6 * f**3, mu**6 * f**3 * b1,
                           mu**8 * f**4])
        bias13 = np.array([b1 * Kaiser(b1, f, mu), bG3 * Kaiser(b1, f, mu),
                           bG2 * Kaiser(b1, f, mu),
                           mu**2 * f * Kaiser(b1, f, mu),
                           mu**2 * f * b1 * Kaiser(b1, f, mu),
                           (mu * f)**2 * Kaiser(b1, f, mu),
                           mu**4 * f**2 * Kaiser(b1, f, mu)])

        Pkmu_22_nw = np.einsum('ijl,ik->kl', bias22, loop22_nw_sub)
        Pkmu_22_w = np.einsum('ijl,ik->kl', bias22, loop22_w)
        Pkmu_13_nw = np.einsum('ijl,ik->kl', bias13, loop13_nw_sub)
        Pkmu_13_w = np.einsum('ijl,ik->kl', bias13, loop13_w)

        Pkmu_22 = Pkmu_22_nw + RSDdamp * Pkmu_22_w
        Pkmu_13 = Pkmu_13_nw + RSDdamp * Pkmu_13_w
        Pkmu = PNLO + Pkmu_22 + Pkmu_13 + Pkmu_ctr

        return RectBivariateSpline(self.ks, mu, Pkmu)

    def P_realspace_terms_kz(self, D=None):
        r"""P_realspace_terms_kz

        Computes interpolators for all the Perturbation Theory (PT) terms
        of the non-linear galaxy bias expansion for Pgg and Pgm in
        real space at the proper redshift specified by the growth parameters.

        Parameters
        ----------
        D(z):             nunmpy.ndarray, growth factor at all redshifts

        Returns
        -------
        Tuple of (P_b1b2(k,z), P_b1bG2(k,z),
        P_b2b2(k,z), P_b2bG2(k,z), P_bG2bG2(k,z),
        P_Z1bG3(k,z), P_Z1bG2(k,z)) as 2D interpolator
        """
        if D is None:
            if self.use_gamma_MG:
                D = self.growth_factor_func(self.z)
            else:
                D = self.growth_factor_func(self.z, 0.05)
        # We use D^4 because one-loop terms go as ~ P_lin^2
        D4 = D**4

        # Compute loop 22 linear and rescale to D**4
        Pb1b2 = np.array([self.loop22_L[1] * d for d in D4])
        Pb1bG2 = np.array([self.loop22_L[2] * d for d in D4])
        Pb2b2 = np.array([self.loop22_L[3] * d for d in D4])
        Pb2bG2 = np.array([self.loop22_L[4] * d for d in D4])
        PbG2bG2 = np.array([self.loop22_L[5] * d for d in D4])

        # Compute loop 13 linear and rescale to D**4
        PZ1bG3 = np.array([self.loop13_L[1] * d for d in D4])
        PZ1bG2 = np.array([self.loop13_L[2] * d for d in D4])

        return (RectBivariateSpline(self.z, self.ks, Pb1b2),
                RectBivariateSpline(self.z, self.ks, Pb1bG2),
                RectBivariateSpline(self.z, self.ks, Pb2b2),
                RectBivariateSpline(self.z, self.ks, Pb2bG2),
                RectBivariateSpline(self.z, self.ks, PbG2bG2),
                RectBivariateSpline(self.z, self.ks, PZ1bG3),
                RectBivariateSpline(self.z, self.ks, PZ1bG2))


class FASTPTPlus(fpts.FASTPT):
    r"""FASTPTPlus.

    Class to compute all the building blocks for the Ivanov EFT based model,
    inherits from `fastpt.FASTPT_simple`.
    """

    def Pkmu_22_one_loop_terms(self, k, P, P_window=None, C_window=None):
        r"""Pkmu_22_one_loop_terms.

        Computes the mode-coupling loop corrections for the redshift space
        galaxy power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        Pb1b1, Pb1b2,.. , Pmu8f4: numpy.ndarray
            Mode-coupling loop corrections
        """
        Power, MAT = self.J_k(P, P_window=P_window, C_window=C_window)
        J000, J002, J004, J2m22, J1m11, J1m13, J2m20r = MAT

        Pb1b1 = (1219 / 735.0 * J000 + J2m20r / 3.0 + 124 / 35.0 * J1m11 +
                 2 / 3.0 * J2m22 + 1342 / 1029.0 * J002 + 16 / 35.0 * J1m13 +
                 64 / 1715.0 * J004)
        Pb1b2 = 34 / 21.0 * J000 + 2.0 * J1m11 + 8 / 21.0 * J002
        Pb1bG2 = (-72 / 35.0 * J000 + 8 / 5.0 * (J1m13 - J1m11) +
                  88 / 49.0 * J002 + 64 / 245.0 * J004)
        Pb2b2 = 0.5 * (J000 - J000[0])
        Pb2bG2 = 4 / 3. * (J002 - J000)
        PbG2bG2 = 16 / 15. * J000 - 32 / 21. * J002 + 16 / 35. * J004

        Pmu2fb1 = 2. * (1003 / 735.0 * J000 + J2m20r / 3.0 +
                        116 / 35.0 * J1m11 + 1606 / 1029.0 * J002 +
                        2 / 3.0 * J2m22 + 24 / 35.0 * J1m13 +
                        128 / 1715.0 * J004)
        Pmu2fb2 = 26 / 21.0 * J000 + 2.0 * J1m11 + 16 / 21.0 * J002
        Pmu2fbG2 = (-152 / 105. * J000 + 8 / 5. * (J1m13 - J1m11) +
                    136 / 147. * J002 + 128 / 245. * J004)
        Pmu2f2b12 = (-J000 + J2m20r + J002 - J2m22) / 3.0
        Pmu2fb12 = (82 / 21.0 * J000 + 2 / 3.0 * J2m20r + 264 / 35.0 * J1m11 +
                    44 / 21.0 * J002 + 4 / 3.0 * J2m22 + 16 / 35.0 * J1m13)
        Pmu2fb1b2 = 2.0 * (J000 + J1m11)
        Pmu2fb1bG2 = 8.0 * ((J002 - J000) / 3.0 + (J1m13 - J1m11) / 5.0)
        Pmu2f2b1 = 0.25 * Pb1bG2
        Pmu2f2b2 = 0.25 * Pb2bG2
        Pmu2f2bG2 = 0.50 * PbG2bG2

        Pmu4f4 = 3 / 32.0 * PbG2bG2
        Pmu4f3 = Pmu2fbG2 / 4.0
        Pmu4f3b1 = (4 / 3.0 * (J002 - J000) + 2 / 3.0 * (J2m20r - J2m22) +
                    2 / 5.0 * (J1m13 - J1m11))
        Pmu4f2b2 = 5 / 3.0 * J000 + 2.0 * J1m11 + J002 / 3.0
        Pmu4f2bG2 = (8 / 5.0 * (J1m13 - J1m11) - 32 / 15.0 * J000 +
                     40 / 21.0 * J002 + 8 / 35.0 * J004)
        Pmu4f2b1 = (98 / 15.0 * J000 + 4 / 3.0 * J2m20r + 498 / 35.0 * J1m11 +
                    794 / 147. * J002 + 8 / 3.0 * J2m22 + 62 / 35.0 * J1m13 +
                    16 / 245.0 * J004)
        Pmu4f2b12 = 8 / 3.0 * J000 + 4.0 * J1m11 + J002 / 3.0 + J2m22
        Pmu4f2 = (851 / 735.0 * J000 + J2m20r / 3.0 + 108 / 35.0 * J1m11 +
                  1742 / 1029.0 * J002 + 2 / 3.0 * J2m22 + 32 / 35.0 * J1m13 +
                  256 / 1715.0 * J004)

        Pmu6f4 = (-14 / 15.0 * J000 + (J2m20r - J2m22) / 3.0 +
                  2 / 5.0 * (J1m13 - J1m11) + 19 / 21.0 * J002 + J004 / 35.0)
        Pmu6f3 = (292 / 105.0 * J000 + 2 / 3.0 * J2m20r + 234 / 35.0 * J1m11 +
                  454 / 147.0 * J002 + 4 / 3.0 * J2m22 + 46 / 35.0 * J1m13 +
                  32 / 245.0 * J004)
        Pmu6f3b1 = (14 / 3.0 * J000 + 38 / 5.0 * J1m11 + 4 / 3.0 * J002 +
                    2.0 * J2m22 + 2 / 5.0 * J1m13)

        Pmu8f4 = (21 / 10.0 * J000 + 18 / 5.0 * J1m11 + 6 / 7.0 * J002 +
                  J2m22 + 2 / 5.0 * J1m13 + 3 / 70.0 * J004)

        if (self.extrap):
            _, Pb1b1 = self.EK.PK_original(Pb1b1)
            _, Pb1b2 = self.EK.PK_original(Pb1b2)
            _, Pb1bG2 = self.EK.PK_original(Pb1bG2)
            _, Pb2b2 = self.EK.PK_original(Pb2b2)
            _, Pb2bG2 = self.EK.PK_original(Pb2bG2)
            _, PbG2bG2 = self.EK.PK_original(PbG2bG2)

            _, Pmu2fb1 = self.EK.PK_original(Pmu2fb1)
            _, Pmu2fb2 = self.EK.PK_original(Pmu2fb2)
            _, Pmu2fbG2 = self.EK.PK_original(Pmu2fbG2)
            _, Pmu2f2b12 = self.EK.PK_original(Pmu2f2b12)
            _, Pmu2fb12 = self.EK.PK_original(Pmu2fb12)
            _, Pmu2fb1b2 = self.EK.PK_original(Pmu2fb1b2)
            _, Pmu2fb1bG2 = self.EK.PK_original(Pmu2fb1bG2)
            _, Pmu2f2b1 = self.EK.PK_original(Pmu2f2b1)
            _, Pmu2f2b2 = self.EK.PK_original(Pmu2f2b2)
            _, Pmu2f2bG2 = self.EK.PK_original(Pmu2f2bG2)

            _, Pmu4f4 = self.EK.PK_original(Pmu4f4)
            _, Pmu4f3 = self.EK.PK_original(Pmu4f3)
            _, Pmu4f3b1 = self.EK.PK_original(Pmu4f3b1)
            _, Pmu4f2b2 = self.EK.PK_original(Pmu4f2b2)
            _, Pmu4f2bG2 = self.EK.PK_original(Pmu4f2bG2)
            _, Pmu4f2b1 = self.EK.PK_original(Pmu4f2b1)
            _, Pmu4f2b12 = self.EK.PK_original(Pmu4f2b12)
            _, Pmu4f2 = self.EK.PK_original(Pmu4f2)

            _, Pmu6f4 = self.EK.PK_original(Pmu6f4)
            _, Pmu6f3 = self.EK.PK_original(Pmu6f3)
            _, Pmu6f3b1 = self.EK.PK_original(Pmu6f3b1)

            _, Pmu8f4 = self.EK.PK_original(Pmu8f4)

        return (Pb1b1, Pb1b2, Pb1bG2, Pb2b2, Pb2bG2, PbG2bG2, Pmu2fb1, Pmu2fb2,
                Pmu2fbG2, Pmu2f2b12, Pmu2fb12, Pmu2fb1b2, Pmu2fb1bG2, Pmu2f2b1,
                Pmu2f2b2, Pmu2f2bG2, Pmu4f4, Pmu4f3, Pmu4f3b1, Pmu4f2b2,
                Pmu4f2bG2, Pmu4f2b1, Pmu4f2b12, Pmu4f2, Pmu6f4, Pmu6f3,
                Pmu6f3b1, Pmu8f4)

    def Pk_22_one_loop_terms_realspace(self, P, P_window=None, C_window=None):
        r"""Pk_22_one_loop_terms_realspace.

        Computes only the real-space 1-loop contributions of :math:`P(k)_{22}`.
        This is used for photometric clustering.
        In that function :math:`P_{b_2 b_2}` is not renormalized by subtracting
        its low-k limit (while it is for the equivalent function
        that is used to compute the spectro redshift-space galaxy
        power spectrum).

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        Pb1b1, Pb1b2,.. , PbG2bG2: numpy.ndarray
            Mode-coupling loop corrections
        """
        Power, MAT = self.J_k(P, P_window=P_window, C_window=C_window)
        J000, J002, J004, J2m22, J1m11, J1m13, J2m20r = MAT

        Pb1b1 = (1219 / 735.0 * J000 + J2m20r / 3.0 + 124 / 35.0 * J1m11 +
                 2 / 3.0 * J2m22 + 1342 / 1029.0 * J002 + 16 / 35.0 * J1m13 +
                 64 / 1715.0 * J004)
        Pb1b2 = 34 / 21.0 * J000 + 2.0 * J1m11 + 8 / 21.0 * J002
        Pb1bG2 = (-72 / 35.0 * J000 + 8 / 5.0 * (J1m13 - J1m11) +
                  88 / 49.0 * J002 + 64 / 245.0 * J004)
        Pb2b2 = 0.5 * J000
        Pb2bG2 = 4 / 3. * (J002 - J000)
        PbG2bG2 = 16 / 15. * J000 - 32 / 21. * J002 + 16 / 35. * J004

        if (self.extrap):
            _, Pb1b1 = self.EK.PK_original(Pb1b1)
            _, Pb1b2 = self.EK.PK_original(Pb1b2)
            _, Pb1bG2 = self.EK.PK_original(Pb1bG2)
            _, Pb2b2 = self.EK.PK_original(Pb2b2)
            _, Pb2bG2 = self.EK.PK_original(Pb2bG2)
            _, PbG2bG2 = self.EK.PK_original(PbG2bG2)

        return (Pb1b1, Pb1b2, Pb1bG2, Pb2b2, Pb2bG2, PbG2bG2)

    def Pkmu_13_one_loop_terms(self, k, P):
        r"""Pkmu_13_one_loop_terms.

        Computes the propagator loop corrections for the redshift space galaxy
        power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        PZ1b1, PZ1bG3, ..., PZ1mu4f2: numpy.ndarray
            Propagator loop corrections
        """

        PZ1b1 = self.P_dd_13_reg(k, P)
        PZ1bG3 = self.P_b1bG3(k, P)
        PZ1bG2 = 5.0 / 2.0 * PZ1bG3
        PZ1mu2f = self.P_tt_13_reg(k, P)
        PZ1mu2fb1 = self.P_mu2fb1_13(k, P)
        PZ1mu2f2 = 3.0 / 8.0 * self.P_mu2f2_13(k, P)
        PZ1mu4f2 = self.P_mu4f2_13(k, P)

        return PZ1b1, PZ1bG3, PZ1bG2, PZ1mu2f, PZ1mu2fb1, PZ1mu2f2, PZ1mu4f2

    def P_dd_13_reg(self, k, P):
        r"""P_dd_13_reg.

        Computes the regularized version of :math:`P_{13}` for the matter
        power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        P_bar: numpy.ndarray
            Regularised version of :math:`P_{13}` for the matter power spectrum
        """

        N = k.size
        n = np.arange(-N + 1, N)
        dL = np.log(k[1]) - np.log(k[0])
        s = n * dL
        cut = 7
        high_s = s[s > cut]
        low_s = s[s < -cut]
        mid_high_s = s[(s <= cut) & (s > 0)]
        mid_low_s = s[(s >= -cut) & (s < 0)]

        def _Z(r):
            return (12.0 / r**2 + 10.0 + 100.0 * r**2 - 42.0 * r**4 +
                    3.0 / r**3 * (r**2 - 1.0)**3 * (7 * r**2 + 2.0) *
                    np.log((r + 1.0) / np.absolute(r - 1.0))) * r

        def _Z_low(r):
            return (352 / 5.0 + 96 / 0.5 / r**2 - 160 / 21.0 / r**4 -
                    526 / 105.0 / r**6 + 236 / 35.0 / r**8) * r

        def _Z_high(r):
            return (928 / 5.0 * r**2 - 4512 / 35.0 * r**4 +
                    416 / 21.0 * r**6 + 356 / 105.0 * r**8) * r

        f_mid_low = _Z(np.exp(-mid_low_s))
        f_mid_high = _Z(np.exp(-mid_high_s))
        f_high = _Z_high(np.exp(-high_s))
        f_low = _Z_low(np.exp(-low_s))

        f = np.hstack((f_low, f_mid_low, 80, f_mid_high, f_high))
        g = fftconvolve(P, f) * dL
        g_k = g[N - 1:2 * N - 1]
        P_bar = 1 / 252.0 * k**3 / (2 * np.pi)**2 * P * g_k

        return P_bar

    def P_b1bG3(self, k, P):
        r"""P_b1bG3

        Computes the contribution :math:`P_{b_1 b_{\mathcal{G}_3}}`
        for the galaxy power spectrum

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        P_b1bG3: numpy.ndarray
            :math:`P_{b_1 b_{\mathcal{G}_3}}` contribution to the
            galaxy power spectrum
        """

        N = k.size
        n = np.arange(-N + 1, N)
        dL = np.log(k[1]) - np.log(k[0])
        s = n * dL
        cut = 7
        high_s = s[s > cut]
        low_s = s[s < -cut]
        mid_high_s = s[(s <= cut) & (s > 0)]
        mid_low_s = s[(s >= -cut) & (s < 0)]

        def _Zb1g3(r):
            return r * (-12.0 / r**2 + 44.0 + 44.0 * r**2 - 12 * r**4 +
                        6.0 / r**3 * (r**2 - 1.0)**4 *
                        np.log((r + 1.0) / np.absolute(r - 1.0)))

        def _Zb1g3_low(r):
            return r * (512 / 5.0 - 1536 / 35.0 / r**2 +
                        512 / 105.0 / r**4 + 512 / 1155.0 / r**6 +
                        512 / 5005.0 / r**8)

        def _Zb1g3_high(r):
            return r * (512 / 5.0 * r**2 - 1536 / 35.0 * r**4 +
                        512 / 105.0 * r**6 + 512 / 1155.0 * r**8)

        fb1g3_mid_low = _Zb1g3(np.exp(-mid_low_s))
        fb1g3_mid_high = _Zb1g3(np.exp(-mid_high_s))
        fb1g3_high = _Zb1g3_high(np.exp(-high_s))
        fb1g3_low = _Zb1g3_low(np.exp(-low_s))

        fb1g3 = np.hstack((fb1g3_low, fb1g3_mid_low, 64, fb1g3_mid_high,
                           fb1g3_high))
        gb1g3 = fftconvolve(P, fb1g3) * dL
        gb1g3_k = gb1g3[N - 1:2 * N - 1]
        Pb1g3_bar = -1.0 / 42.0 * k**3 / (2 * np.pi)**2 * P * gb1g3_k

        return Pb1g3_bar

    def P_mu2f2_13(self, k, P):
        r"""P_mu2f2_13.

        Computes the contribution :math:`P_{\mu^2 f^2, 13}`
        for the galaxy power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        Pb1g3_bar: numpy.ndarray
            Contribution :math:`P_{\mu^2 f^2, 13}` for
            the galaxy power spectrum
        """

        N = k.size
        n = np.arange(-N + 1, N)
        dL = np.log(k[1]) - np.log(k[0])
        s = n * dL
        cut = 7
        high_s = s[s > cut]
        low_s = s[s < -cut]
        mid_high_s = s[(s <= cut) & (s > 0)]
        mid_low_s = s[(s >= -cut) & (s < 0)]

        def _Zb1g3(r):
            return r * (-12.0 / r**2 + 44.0 + 44.0 * r**2 - 12 * r**4 +
                        6.0 / r**3 * (r**2 - 1.0)**4 *
                        np.log((r + 1.0) / np.absolute(r - 1.0)))

        def _Zb1g3_low(r):
            return r * (512 / 5.0 - 1536 / 35.0 / r**2 +
                        512 / 105.0 / r**4 + 512 / 1155.0 / r**6 +
                        512 / 5005.0 / r**8)

        def _Zb1g3_high(r):
            return r * (512 / 5.0 * r**2 - 1536 / 35.0 * r**4 +
                        512 / 105.0 * r**6 + 512 / 1155.0 * r**8)

        fb1g3_mid_low = _Zb1g3(np.exp(-mid_low_s))
        fb1g3_mid_high = _Zb1g3(np.exp(-mid_high_s))
        fb1g3_high = _Zb1g3_high(np.exp(-high_s))
        fb1g3_low = _Zb1g3_low(np.exp(-low_s))

        fb1g3 = np.hstack((fb1g3_low, fb1g3_mid_low, 64, fb1g3_mid_high,
                           fb1g3_high))
        gb1g3 = fftconvolve(P, fb1g3) * dL
        gb1g3_k = gb1g3[N - 1:2 * N - 1]
        Pb1g3_bar = -1.0 / 42.0 * k**3 / (2 * np.pi)**2 * P * gb1g3_k

        return Pb1g3_bar

    def P_tt_13_reg(self, k, P):
        r"""P_tt_13_reg.

        Computes the contribution :math:`P_{Z_1 \mu^2 f}` for the galaxy
        power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        P13tt_bar: numpy.ndarray
            Contribution :math:`P_{Z_1 \mu^2 f}` for the galaxy power spectrum
        """

        N = k.size
        n = np.arange(-N + 1, N)
        dL = np.log(k[1]) - np.log(k[0])
        s = n * dL
        cut = 7
        high_s = s[s > cut]
        low_s = s[s < -cut]
        mid_high_s = s[(s <= cut) & (s > 0)]
        mid_low_s = s[(s >= -cut) & (s < 0)]

        def _Z13tt(r):
            return -1.5 * (-24.0 / r**2 + 52.0 - 8.0 * r**2 + 12.0 * r**4 -
                           6.0 * (r**2 - 1.0)**3 * (r**2 + 2.0) / r**3 *
                           np.log((r + 1.0) / np.absolute(r - 1.0))) * r

        def _Z13tt_low(r):
            return r * (-672.0 / 5.0 + 3744.0 / 35.0 / r**2 -
                        608.0 / 35.0 / r**4 - 160.0 / 77.0 / r**6 -
                        2976.0 / 5005.0 / r**8)

        def _Z13tt_high(r):
            return r * (-96.0 / 5.0 * r**2 - 288.0 / 7.0 * r**4 +
                        352.0 / 35.0 * r**6 + 544.0 / 385.0 * r**8)

        f13tt_mid_low = _Z13tt(np.exp(-mid_low_s))
        f13tt_mid_high = _Z13tt(np.exp(-mid_high_s))
        f13tt_high = _Z13tt_high(np.exp(-high_s))
        f13tt_low = _Z13tt_low(np.exp(-low_s))

        f13tt = np.hstack((f13tt_low, f13tt_mid_low, -48, f13tt_mid_high,
                           f13tt_high))
        g13tt = fftconvolve(P, f13tt) * dL
        g13tt_k = g13tt[N - 1:2 * N - 1]
        P13tt_bar = 1.0 / 252.0 * k**3 / (2 * np.pi)**2 * P * g13tt_k

        return P13tt_bar

    def P_mu2fb1_13(self, k, P):
        r"""P_mu2fb1_13.

        Computes the contribution :math:`P_{Z_1 \mu^2 f b_1}` for the galaxy
        power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        P13_bar: numpy.ndarray
            Contribution :math:`P_{Z_1 \mu^2 f b_1}` for the galaxy
            power spectrum
        """
        N = k.size
        n = np.arange(-N + 1, N)
        dL = np.log(k[1]) - np.log(k[0])
        s = n * dL
        cut = 7
        high_s = s[s > cut]
        low_s = s[s < -cut]
        mid_high_s = s[(s <= cut) & (s > 0)]
        mid_low_s = s[(s >= -cut) & (s < 0)]

        def _Z13(r):
            return (36.0 + 96.0 * r**2 - 36.0 * r**4 +
                    18.0 * (r**2 - 1.0)**3 / r *
                    np.log((r + 1.0) / np.absolute(r - 1.0))) * r

        def _Z13_low(r):
            return r * (576.0 / 5.0 - 576.0 / 35.0 / r**2 -
                        64.0 / 35.0 / r**4 - 192.0 / 385.0 / r**6 -
                        192.0 / 1001.0 / r**8)

        def _Z13_high(r):
            return r * (192.0 * r**2 - 576.0 / 5.0 * r**4 +
                        576.0 / 35.0 * r**6 + 64.0 / 35.0 * r**8)

        f13_mid_low = _Z13(np.exp(-mid_low_s))
        f13_mid_high = _Z13(np.exp(-mid_high_s))
        f13_high = _Z13_high(np.exp(-high_s))
        f13_low = _Z13_low(np.exp(-low_s))

        f13 = np.hstack((f13_low, f13_mid_low, 96, f13_mid_high, f13_high))
        g13 = fftconvolve(P, f13) * dL
        g13_k = g13[N - 1:2 * N - 1]
        P13_bar = 1.0 / 84.0 * k**3 / (2 * np.pi)**2 * P * g13_k

        return P13_bar

    def P_mu4f2_13(self, k, P):
        r"""P_mu4f2_13.

        Computes the contribution :math:`P_{Z_1 \mu^4 f^2 }` for the galaxy
        power spectrum.

        Parameters
        ----------
        k: numpy.ndarray
            Fourier wavenumbers, log-spaced
        P: numpy.ndarray
            Power spectrum

        Returns
        -------
        P13_bar: numpy.ndarray
            Contribution :math:`P_{Z_1 \mu^4 f^2 }` for the
            galaxy power spectrum
        """
        N = k.size
        n = np.arange(-N + 1, N)
        dL = np.log(k[1]) - np.log(k[0])
        s = n * dL
        cut = 7
        high_s = s[s > cut]
        low_s = s[s < -cut]
        mid_high_s = s[(s <= cut) & (s > 0)]
        mid_low_s = s[(s >= -cut) & (s < 0)]

        def _Z13(r):
            return (36.0 / r**2 + 12.0 + 252.0 * r**2 - 108.0 * r**4 +
                    18.0 * (r**2 - 1.0)**3 * (1.0 + 3.0 * r**2) / r**3 *
                    np.log((r + 1.0) / np.absolute(r - 1.0))) * r

        def _Z13_low(r):
            return r * (768.0 / 5.0 + 2304.0 / 35.0 / r**2 -
                        768.0 / 35.0 / r**4 - 256.0 / 77.0 / r**6 -
                        768.0 / 715.0 / r**8)

        def _Z13_high(r):
            return r * (2304.0 / 5.0 * r**2 - 2304.0 / 7.0 * r**4 +
                        256.0 / 5.0 * r**6 + 2304.0 / 385.0 * r**8)

        f13_mid_low = _Z13(np.exp(-mid_low_s))
        f13_mid_high = _Z13(np.exp(-mid_high_s))
        f13_high = _Z13_high(np.exp(-high_s))
        f13_low = _Z13_low(np.exp(-low_s))

        f13 = np.hstack((f13_low, f13_mid_low, 192, f13_mid_high, f13_high))
        g13 = fftconvolve(P, f13) * dL
        g13_k = g13[N - 1:2 * N - 1]
        P13_bar = 1.0 / 336.0 * k**3 / (2 * np.pi)**2 * P * g13_k

        return P13_bar
