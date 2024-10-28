"""DATA

This module contains default data configurations for unit tests.

.. testcode::

        mock_data = {
            'sample': 'ExternalBenchmark',
            'spectro': {
                'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
                'redshifts': ["1.", "1.2", "1.4", "1.65"],
                'cov_is_num': False,
                'scale_cuts_fourier': 'GCspectro-Fourier_test.yaml'},
            'photo': {
                'photo_data': 'standard',
                'luminosity_ratio': 'luminosity_ratio.dat',
                'ndens_GC': 'niTab-EP10-RB00.dat',
                'ndens_WL': 'niTab-EP10-RB00.dat',
                'root_GC': 'Cls_{:s}_PosPos.dat',
                'root_WL': 'Cls_{:s}_ShearShear.dat',
                'root_XC': 'Cls_{:s}_PosShear.dat',
                'root_fits': 'fs2_cls_{:d}zbins_32ellbins.fits',
                'root_mixing_matrix': 'fs2_mms_10zbins_32ellbins.fits',
                'IA_model': 'zNLA',
                'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npz',
                'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npz',
                'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npz',
                'cov_model': 'Gauss',
                'cov_is_num': False,
                'Fourier': True}
                     }
"""

mock_data = {
  'sample': 'ExternalBenchmark',
  'spectro': {
    'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
    'redshifts': ["1.", "1.2", "1.4", "1.65"],
    'cov_is_num': False,
    'Fourier': True,
    'scale_cuts_fourier': 'GCspectro-Fourier_test.yaml',
    'root_mixing_matrix': 'mm_FS230degCircle_m3_nosm_obsz_z0.9-1.1.fits'},
  'photo': {
    'photo_data': 'standard',
    'luminosity_ratio': 'luminosity_ratio.dat',
    'ndens_GC': 'niTab-EP10-RB00.dat',
    'ndens_WL': 'niTab-EP10-RB00.dat',
    'root_GC': 'Cls_{:s}_PosPos.dat',
    'root_WL': 'Cls_{:s}_ShearShear.dat',
    'root_XC': 'Cls_{:s}_PosShear.dat',
    'root_fits': 'fs2_cls_{:d}zbins_32ellbins.fits',
    'root_mixing_matrix': 'fs2_mms_10zbins_32ellbins.fits',
    'IA_model': 'zNLA',
    'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npz',
    'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npz',
    'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npz',
    'cov_model': 'Gauss',
    'cov_is_num': False,
    'Fourier': True}
}
