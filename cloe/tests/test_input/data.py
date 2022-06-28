"""Contains default data configurations for unit tests

"""

mock_data = {
  'sample': 'ExternalBenchmark',
  'spectro': {
    'root': 'cov_power_galaxies_dk0p004_z{:s}.fits',
    'redshifts': ["1.", "1.2", "1.4", "1.65"]},
  'photo': {
    'ndens_GC': 'niTab-EP10-RB00.dat',
    'ndens_WL': 'niTab-EP10-RB00.dat',
    'root_GC': 'Cls_{:s}_PosPos.dat',
    'root_WL': 'Cls_{:s}_ShearShear.dat',
    'root_XC': 'Cls_{:s}_PosShear.dat',
    'IA_model': 'zNLA',
    'cov_GC': 'CovMat-PosPos-{:s}-20Bins.npy',
    'cov_WL': 'CovMat-ShearShear-{:s}-20Bins.npy',
    'cov_3x2pt': 'CovMat-3x2pt-{:s}-20Bins.npy',
    'cov_model': 'Gauss'}
}
