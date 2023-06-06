"""MOCK OBJECTS

This module contains mock objects used in unit tests.

"""


class mock_CAMB_data:
    """Mocks CAMB data."""

    def __init__(self, rz_interp):

        self.rz_interp = rz_interp

    def angular_diameter_distance2(self, z1, z2):
        """Angular diameter distance."""

        add2 = (
            self.rz_interp(z2) / (1.0 + z2) - self.rz_interp(z1) / (1.0 + z2)
        )

        return add2


class mock_P_obj:
    """Mocks P Object."""

    def __init__(self, p_interp):

        self.P = p_interp


def mock_MG_func(z, k):
    """
    Tests MG function that simply returns 1.

    Parameters
    ----------
    z: float
        Redshift
    k: float
        Angular scale

    Returns
    -------
    float
        Returns 1 for test purposes
    """
    return 1.0


def update_dict_w_mock(cosmo_dict):
    """Updates Dictionary with mock data.

    This funciton takes in a cosmology dictionary object and assigns mock
    objects to certain keys. This allows the dictionary object to be saved as
    a :obj:`pickle`.

    Parameters
    ----------
    cosmo_dict: dict
        Cosmology dictionary

    """

    cosmo_dict['MG_sigma'] = mock_MG_func
    cosmo_dict['MG_mu'] = mock_MG_func

    if 'Pdd_phot' in cosmo_dict.keys():
        cosmo_dict['Pk_delta'] = mock_P_obj(cosmo_dict['Pdd_phot'])

    if 'r_z_func' in cosmo_dict.keys():
        cosmo_dict['CAMBdata'] = mock_CAMB_data(cosmo_dict['r_z_func'])

    return cosmo_dict
