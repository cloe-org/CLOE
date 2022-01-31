"""UNIT TESTS FOR LIKE_CALC

This module contains unit tests for the likelihood calculation module.
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from likelihood.like_calc.euclike import Euclike
from astropy import constants as const
from pathlib import Path
from scipy import interpolate
from likelihood.tests.test_input.data import mock_data
from likelihood.tests.test_input.mock_observables import build_mock_observables


# temporary fix, see #767
class mock_CAMB_data:
    def __init__(self, rz_interp):
        self.rz_interp = rz_interp

    def angular_diameter_distance2(self, z1, z2):
        return self.rz_interp(z1) - self.rz_interp(z2)


def mock_MG_func(z, k):
    """
    Test MG function that simply returns 1.

    Parameters
    ----------
    z: float
        Redshift.
    k: float
        Angular scale.

    Returns
    -------
    float
        Returns 1 for test purposes.
    """
    return 1.0


class mock_P_obj:
    def __init__(self, p_interp):
        self.P = p_interp


class likecalcTestCase(TestCase):

    def setUp(self):
        cur_dir = Path(__file__).resolve().parents[0]
        cmov_file = np.loadtxt(str(cur_dir) +
                               '/test_input/ComDist-LCDM-Lin-zNLA.dat')
        zs_r = cmov_file[:, 0]
        rs = cmov_file[:, 1]
        ang_dists = rs / (1.0 + zs_r)

        rz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r, y=rs,
                                                             ext=0)
        dz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_r,
                                                             y=ang_dists,
                                                             ext=0)

        Hz_file = np.loadtxt(str(cur_dir) + '/test_input/Hz.dat')
        zs_H = Hz_file[:, 0]
        Hs = Hz_file[:, 1]
        Hs_mpc = Hz_file[:, 1] / const.c.to('km/s').value

        Hz_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H, y=Hs,
                                                             ext=0)

        Hmpc_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                               y=Hs_mpc,
                                                               ext=0)

        f_sig_8_arr = np.load(str(cur_dir) +
                              '/test_input/f_sig_8_arr.npy',
                              allow_pickle=True)
        sig_8_arr = np.load(str(cur_dir) +
                            '/test_input/sig_8_arr.npy',
                            allow_pickle=True)
        f_z_arr = np.load(str(cur_dir) +
                          '/test_input/f_z_arr.npy',
                          allow_pickle=True)

        sig_8_interp = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(0.0, 5.0, 50),
            y=sig_8_arr[::-1], ext=0)
        f_sig_8_interp = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(0.0, 5.0, 50),
            y=f_sig_8_arr[::-1], ext=0)
        f_z_interp = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(0.0, 5.0, 50),
            y=f_z_arr[::-1], ext=0)

        MG_interp = mock_MG_func

        pdd = np.load(str(cur_dir) + '/test_input/pdd.npy')
        pdi = np.load(str(cur_dir) + '/test_input/pdi.npy')
        pgd = np.load(str(cur_dir) + '/test_input/pgd.npy')
        pgg = np.load(str(cur_dir) + '/test_input/pgg.npy')
        pgi_phot = np.load(str(cur_dir) + '/test_input/pgi_phot.npy')
        pgi_spectro = np.load(str(cur_dir) + '/test_input/pgi_spectro.npy')
        pii = np.load(str(cur_dir) + '/test_input/pii.npy')

        zs_base = np.linspace(0.0, 4.0, 100)
        ks_base = np.logspace(-3.0, 1.0, 100)

        mock_cosmo_dic = {'ombh2': 0.022445, 'omch2': 0.121203, 'H0': 67.0,
                          'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046,
                          'omkh2': 0.0, 'omnuh2': 0.0, 'ns': 0.96,
                          'w': -1.0, 'sigma8_0': 0.816,
                          'As': 2.115e-9, 'sigma8_z_func': sig_8_interp,
                          'fsigma8_z_func': f_sig_8_interp,
                          'f_z': f_z_interp,
                          'r_z_func': rz_interp, 'd_z_func': dz_interp,
                          # pretend that f_K_z_func behaves as r_z_func.
                          # This is not realistic but it is fine for the
                          # purposes of the unit tests
                          'f_K_z_func': rz_interp,
                          'H_z_func_Mpc': Hmpc_interp,
                          'H_z_func': Hz_interp,
                          'z_win': zs_base,
                          'k_win': ks_base,
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value,
                          'nuisance_parameters': {
                              'b1_photo': 1.0997727037892875,
                              'b2_photo': 1.220245876862528,
                              'b3_photo': 1.2723993083933989,
                              'b4_photo': 1.316624471897739,
                              'b5_photo': 1.35812370570578,
                              'b6_photo': 1.3998214171814918,
                              'b7_photo': 1.4446452851824907,
                              'b8_photo': 1.4964959071110084,
                              'b9_photo': 1.5652475842498528,
                              'b10_photo': 1.7429859437184225,
                              'b1_spectro': 1.4614804,
                              'b2_spectro': 1.6060949,
                              'b3_spectro': 1.7464790,
                              'b4_spectro': 1.8988660,
                              'aia': 1.72,
                              'nia': -0.41,
                              'bia': 0.0,
                              'dz_1_GCphot': 0., 'dz_1_WL': 0.,
                              'dz_2_GCphot': 0., 'dz_2_WL': 0.,
                              'dz_3_GCphot': 0., 'dz_3_WL': 0.,
                              'dz_4_GCphot': 0., 'dz_4_WL': 0.,
                              'dz_5_GCphot': 0., 'dz_5_WL': 0.,
                              'dz_6_GCphot': 0., 'dz_6_WL': 0.,
                              'dz_7_GCphot': 0., 'dz_7_WL': 0.,
                              'dz_8_GCphot': 0., 'dz_8_WL': 0.,
                              'dz_9_GCphot': 0., 'dz_9_WL': 0.,
                              'dz_10_GCphot': 0., 'dz_10_WL': 0.}}

        # precomputed parameters
        mock_cosmo_dic['H0_Mpc'] = \
            mock_cosmo_dic['H0'] / const.c.to('km/s').value
        mock_cosmo_dic['Omb'] = \
            mock_cosmo_dic['ombh2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
        mock_cosmo_dic['Omc'] = \
            mock_cosmo_dic['omch2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
        mock_cosmo_dic['Omnu'] = \
            mock_cosmo_dic['omnuh2'] / (mock_cosmo_dic['H0'] / 100.) ** 2.
        mock_cosmo_dic['Omm'] = (mock_cosmo_dic['Omnu'] +
                                 mock_cosmo_dic['Omc'] +
                                 mock_cosmo_dic['Omb'])

        p_matter = mock_P_obj(interpolate.interp2d(zs_base, ks_base, pdd.T,
                                                   fill_value=0))
        mock_cosmo_dic['Pk_delta'] = p_matter

        mock_cosmo_dic['Pmm_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pdd,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pgg_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgg,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pgdelta_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgd,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pii'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pii,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pdeltai'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pdi,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pgi_phot'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgi_phot,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pgi_spectro'] = \
            interpolate.RectBivariateSpline(zs_base,
                                            ks_base,
                                            pgi_spectro,
                                            kx=1, ky=1)

        mock_cosmo_dic['Pgg_spectro'] = np.vectorize(self.Pgg_spectro_def)

        # temporary fix, see #767
        mock_cosmo_dic['CAMBdata'] = mock_CAMB_data(rz_interp)

        fid_H_arr = np.load(str(cur_dir) + '/test_input/spectro_fid_HZ.npy')
        fid_d_A_arr = np.load(str(cur_dir) +
                              '/test_input/spectro_fid_d_A.npy')

        fid_H_interp = \
            interpolate.InterpolatedUnivariateSpline(x=zs_H, y=fid_H_arr,
                                                     ext=0)
        fid_dA_interp = \
            interpolate.InterpolatedUnivariateSpline(x=zs_H, y=fid_d_A_arr,
                                                     ext=0)

        # Note: the 'fiducial' cosmology declared here is purely for the
        # purposes of testing the spectro module. It is not representative
        # of our fiducial model nor does it correspond to the fiducial model
        # used by OU-LE3 to compute distances.
        fid_mock_dic = {'H0': 67.5,
                        'omch2': 0.122,
                        'ombh2': 0.022,
                        'omnuh2': 0.00028,
                        'omkh2': 0.0,
                        'w': -1.0,
                        'mnu': 0.06,
                        'tau': 0.07,
                        'nnu': 3.046,
                        'ns': 0.9674,
                        'As': 2.1e-9,
                        'c': const.c.to('km/s').value,
                        'd_z_func': fid_dA_interp,
                        'H_z_func': fid_H_interp,
                        'z_win': zs_base,
                        'k_win': ks_base,
                        'MG_sigma': MG_interp,
                        'NL_flag': 1,
                        'nuisance_parameters': {
                            'b1_photo': 1.0997727037892875,
                            'b2_photo': 1.220245876862528,
                            'b3_photo': 1.2723993083933989,
                            'b4_photo': 1.316624471897739,
                            'b5_photo': 1.35812370570578,
                            'b6_photo': 1.3998214171814918,
                            'b7_photo': 1.4446452851824907,
                            'b8_photo': 1.4964959071110084,
                            'b9_photo': 1.5652475842498528,
                            'b10_photo': 1.7429859437184225,
                            'b1_spectro': 1.4614804,
                            'b2_spectro': 1.6060949,
                            'b3_spectro': 1.7464790,
                            'b4_spectro': 1.8988660,
                            'aia': 1.72,
                            'nia': -0.41,
                            'bia': 0.0,
                            'dz_1_GC': 0., 'dz_1_WL': 0.,
                            'dz_2_GC': 0., 'dz_2_WL': 0.,
                            'dz_3_GC': 0., 'dz_3_WL': 0.,
                            'dz_4_GC': 0., 'dz_4_WL': 0.,
                            'dz_5_GC': 0., 'dz_5_WL': 0.,
                            'dz_6_GC': 0., 'dz_6_WL': 0.,
                            'dz_7_GC': 0., 'dz_7_WL': 0.,
                            'dz_8_GC': 0., 'dz_8_WL': 0.,
                            'dz_9_GC': 0., 'dz_9_WL': 0.,
                            'dz_10_GC': 0., 'dz_10_WL': 0.}}

        self.fiducial_dict = fid_mock_dic
        self.test_dict = mock_cosmo_dic
        # init Euclike
        mock_obs = build_mock_observables()
        self.like_tt = Euclike(mock_data, mock_obs)

        # The correct check value, using the h scaling for the h from
        # supplied external file is:
        self.check_loglike = -88.074928

    def tearDown(self):
        self.check_loglike = None

    def istf_spectro_galbias(self, redshift, bin_edge_list=None):
        """
        Updates galaxy bias for the spectroscopic galaxy clustering
        probe, at given redshift, according to default recipe.

        Note: for redshifts above the final bin (z > 1.80), we use the bias
        from the final bin. Similarly, for redshifts below the first bin
        (z < 0.90), we use the bias of the first bin.

        Attention: this will change in the future

        Parameters
        ----------
        redshift: float
            Redshift at which to calculate bias.
        bin_edge_list: list
            List of redshift bin edges for spectroscopic GC probe.
            Default is Euclid IST: Forecasting choices.

        Returns
        -------
        bi_val: float
            Value of spectroscopic galaxy bias at input redshift
        """
        if bin_edge_list is None:
            bin_edge_list = [0.90, 1.10, 1.30, 1.50, 1.80]

        istf_bias_list = [self.test_dict['nuisance_parameters']['b1_spectro'],
                          self.test_dict['nuisance_parameters']['b2_spectro'],
                          self.test_dict['nuisance_parameters']['b3_spectro'],
                          self.test_dict['nuisance_parameters']['b4_spectro']]

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
        elif redshift >= bin_edge_list[-1]:
            bi_val = istf_bias_list[-1]
        elif redshift < bin_edge_list[0]:
            bi_val = istf_bias_list[0]
        return bi_val

    def Pgg_spectro_def(self, redshift, k_scale, mu_rsd):
        r"""
        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float
            cosine of the angle between the pair separation and
            the line of sight

        Returns
        -------
        pval: float
            Value of galaxy-galaxy power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy clustering spectroscopic
        """
        bias = self.istf_spectro_galbias(redshift)
        growth = self.test_dict['f_z'](redshift)
        power = self.test_dict['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def test_loglike(self):
        npt.assert_allclose(self.like_tt.loglike(
            self.test_dict,
            self.fiducial_dict),
                self.check_loglike, rtol=1e-06, err_msg='Loglike failed')
