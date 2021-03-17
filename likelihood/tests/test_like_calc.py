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
        pgi_spec = np.load(str(cur_dir) + '/test_input/pgi_spec.npy')
        pii = np.load(str(cur_dir) + '/test_input/pii.npy')

        zs_base = np.linspace(0.0, 4.0, 100)
        ks_base = np.logspace(-3.0, 1.0, 100)

        mock_cosmo_dic = {'ombh2': 0.022445, 'omch2': 0.121203, 'H0': 67.0,
                          'tau': 0.07, 'mnu': 0.06, 'nnu': 3.046,
                          'omkh2': 0.0, 'omnuh2': 0.0, 'ns': 0.96,
                          'w': -1.0, 'sigma_8_0': 0.816,
                          'As': 2.115e-9, 'sigma8_z_func': sig_8_interp,
                          'fsigma8_z_func': f_sig_8_interp,
                          'f_z': f_z_interp,
                          'r_z_func': rz_interp, 'd_z_func': dz_interp,
                          'H_z_func_Mpc': Hmpc_interp,
                          'H_z_func': Hz_interp,
                          'z_win': zs_base,
                          'k_win': ks_base,
                          'MG_sigma': MG_interp, 'c': const.c.to('km/s').value,
                          'nuisance_parameters': {
                              'like_selection': 2,
                              'full_photo': True,
                              'NL_flag': 1,
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
                              'b1_spec': 1.4614804,
                              'b2_spec': 1.6060949,
                              'b3_spec': 1.7464790,
                              'b4_spec': 1.8988660,
                              'aia': 1.72,
                              'nia': -0.41,
                              'bia': 0.0}}

        # MM: precomputed parameters
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
        mock_cosmo_dic['Pk_interpolator'] = p_matter
        mock_cosmo_dic['Pk_delta'] = p_matter
        mock_cosmo_dic['Pgg_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgg.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgdelta_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                              pgd.T,
                                                              fill_value=0.0)
        mock_cosmo_dic['Pii'] = interpolate.interp2d(zs_base, ks_base,
                                                     pii.T,
                                                     fill_value=0.0)
        mock_cosmo_dic['Pdeltai'] = interpolate.interp2d(zs_base, ks_base,
                                                         pdi.T,
                                                         fill_value=0.0)
        mock_cosmo_dic['Pgi_phot'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgi_phot.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgi_spec'] = interpolate.interp2d(zs_base, ks_base,
                                                          pgi_spec.T,
                                                          fill_value=0.0)
        mock_cosmo_dic['Pgg_spec'] = np.vectorize(self.Pgg_spec_def)

        fid_H_arr = np.load(str(cur_dir) + '/test_input/spec_fid_HZ.npy')
        fid_d_A_arr = np.load(str(cur_dir) + '/test_input/spec_fid_d_A.npy')

        fid_H_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                                y=fid_H_arr,
                                                                ext=0)
        fid_dA_interp = interpolate.InterpolatedUnivariateSpline(x=zs_H,
                                                                 y=fid_d_A_arr,
                                                                 ext=0)

        # ACD: Note - the 'fiducial' cosmology declared here is purely for the
        # purposes of testing the spec module. It is not representative of our
        # fiducial model nor does it correspond to the fiducial model used by
        # OU-LE3 to compute distances.

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
                        'nuisance_parameters': {
                            'like_selection': 2,
                            'full_photo': True,
                            'NL_flag': 1,
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
                            'b1_spec': 1.4614804,
                            'b2_spec': 1.6060949,
                            'b3_spec': 1.7464790,
                            'b4_spec': 1.8988660,
                            'aia': 1.72,
                            'nia': -0.41,
                            'bia': 0.0}}

        self.fiducial_dict = fid_mock_dic
        self.test_dict = mock_cosmo_dic
        # init Euclike
        self.like_tt = Euclike()

        # (SJ): For now use loglike below, to be updated
        # (SJ): First one without, second one with h and (2pi/h)^3 corrections
        # (SJ): third one with only 1/h^3 correction, pre-update to data
        # self.check_loglike = 4.607437e+11
        # self.check_loglike = 30568.400834
        # (AP): The following is with the original bias interpolator
        #       of the spec class
        # self.check_loglike = 1.963907e+11
        # (AP): The following is with the IST:F predictions
        # self.check_loglike = 2.459737e+11
        # (AP): The following is with the correct amplitude of GCSpec ((2pi)^3)
        # self.check_loglike = -2284.723389
        # (ACD): The following is the incorrect amplitude when assuming
        # fiducial cosmology from code rather than for file for unit
        # conversion:
        # self.check_loglike = 4569.446812
        # (ACD): The correct check value, using the h scaling for the h from
        # supplied external file is:
        self.check_loglike = -86.386838

    def tearDown(self):
        self.check_loglike = None

    def istf_spec_galbias(self, redshift,
                          bin_edge_list=[0.90, 1.10, 1.30, 1.50, 1.80]):
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

        istf_bias_list = [self.test_dict['nuisance_parameters']['b1_spec'],
                          self.test_dict['nuisance_parameters']['b2_spec'],
                          self.test_dict['nuisance_parameters']['b3_spec'],
                          self.test_dict['nuisance_parameters']['b4_spec']]

        if bin_edge_list[0] <= redshift < bin_edge_list[-1]:
            for i in range(len(bin_edge_list) - 1):
                if bin_edge_list[i] <= redshift < bin_edge_list[i + 1]:
                    bi_val = istf_bias_list[i]
        elif redshift >= bin_edge_list[-1]:
            bi_val = istf_bias_list[-1]
        elif redshift < bin_edge_list[0]:
            bi_val = istf_bias_list[0]
        return bi_val

    def Pgg_spec_def(self, redshift, k_scale, mu_rsd):
        r"""
        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spec}(z, k) &=\
            [b_{\rm g}^{\rm spec}(z) + f(z, k)\mu_{k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum.
        k_scale: float
            k-mode at which to evaluate the power spectrum.
        mu_rsd: float
            cosinus of the angle between the pair separation and
            the line of sight

        Returns
        -------
        pval: float
            Value of galaxy-galaxy power spectrum
            at a given redshift, k-mode and :math:`\mu_{k}`
            for galaxy cclustering spectroscopic
        """
        bias = self.istf_spec_galbias(redshift)
        growth = self.test_dict['f_z'](redshift)
        power = self.test_dict['Pk_delta'].P(redshift, k_scale)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def test_loglike(self):
        npt.assert_allclose(self.like_tt.loglike(
            self.test_dict,
            self.fiducial_dict),
                self.check_loglike, rtol=1e-06, err_msg='Loglike failed')
