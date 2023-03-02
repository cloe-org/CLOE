"""redshift_bins

Contains common function to operate on redshift bin edges
and select nuisance parameters.
"""

import numpy as np


def coerce(zs, redshift_edges, is_sorted=False):
    """
    Return the redshift(s) coerced within the edges.

    Parameters
    ----------
    zs: float or numpy.ndarray or list
        Input redshift(s) to be binned
    redshift_edges: numpy.ndarray
        Array of redshift bin edges.
        It has to be 1-dimensional.
    is_sorted: bool (optional)
        Specifies whether redshift_edges is sorted.

    Returns
    -------
    coerced_zs: float or numpy.ndarray
        Coerced redshift(s)
    """
    min_edge = redshift_edges[0] if is_sorted else min(redshift_edges)
    max_edge = redshift_edges[-1] if is_sorted else max(redshift_edges)
    return np.maximum(min_edge, np.minimum(zs, max_edge))


def find_bin(zs, redshift_edges, check_bounds=False):
    """
    Return the redshift bin(s) to which zs in input belongs.

    Parameters
    ----------
    zs: float or numpy.array or list
        Input redshift(s) to be binned.
    redshift_edges: numpy.ndarray
        Array of redshift bin edges.
        It has to be 1-dimensional and monotonic.
    check_bounds: bool
        If True, raises ValueError for
        input zs outside the boundaries of redshift_edges.

    Returns
    -------
    zs_bin: int or numpy.ndarray of int
        Bin(s) of the redshift.
        The bin indexing is 1-based:
        zs in the first bin returns 1.
        If check_bounds is False,
        zs below the first bin returns 0,
        and zs above the last bin returns
        the length of redshift_edges.

    Raises
    ------
    ValueError
        If zs < wrt the lowest edge
        and check_bounds is True
    ValueError
        If zs >= wrt the highest edge
        and check_bounds is True
    """
    zs_bin = np.digitize(zs, redshift_edges)
    if check_bounds is True:
        zs_bin_array = np.asarray(zs_bin)
        if 0 in zs_bin_array:
            raise ValueError('Some input redshift value'
                             ' below the lowest bin edge '
                             f'{redshift_edges[0]}')
        elif len(redshift_edges) in zs_bin_array:
            raise ValueError('Some input redshift value'
                             ' above the highest bin edge '
                             f'{redshift_edges[-1]}')
    return zs_bin


def compute_means_of_consecutive(redshift_edges):
    """
    Return the means of consecutive redshift edges.

    Parameters
    ----------
    redshift_edges: numpy.ndarray
        Array of redshift bins.
        It has to be 1-dimensional and monotonic.

    Returns
    -------
    means: numpy.ndarray
        Means of consecutive redshift edges.
    """
    return (redshift_edges[1:] + redshift_edges[:-1]) / 2


def reduce(z_bins, lower_bound, upper_bound):
    """
    Selects z_bins elements within boundaries.

    Parameters
    ----------
    z_bins: ndarray
        Array of redshift bins.
    lower_bound: float
        Lower bound for the selection (>=)
    upper_bound: float
        Upper bound for the selection (<)

    Returns
    -------
    reduced_z_bins: ndarray
        Selected z_bins elements greater (or equal) than lower bound
        and less than upper bound.
    """
    return z_bins[(z_bins >= lower_bound) & (z_bins < upper_bound)]


def select_spectro_parameters(redshift, nuis_dict, bin_edges=None):
    """Selector of parameters for GCspectro recipes

    Returns dictionary of GCspectro parameters based on input redshift,
    according to the specified bin edges for the spectroscopic bins.

    Parameters
    ----------
    redshift: float or numpy.ndarray
        Redshift(s) at which to provide GCspectro parameters.
    nuis_dict: dict
        Dictionary containing pairs of all the nuisance parameters and
        their current values.
    bin_edges: numpy.ndarray
        Array containing the edges of the redshift bins for GSspectro.
        Default is Euclid IST:F choices.

    Returns
    -------
    par_dict: dict
        Dictionary containing GCspectro parameters for the selected input
        redshift. Dictionary values are float or numpy.ndarray, depending
        if redshift is a float or a numpy.ndarray, respectively.

    Raises
    ------
    ValueError
        If redshift is outside of the bounds defined by the first
        and last element of the input bin edges.
    KeyError
        If nuisance parameter dictionary does not contain expected
        GCspectro parameters.
    """

    if bin_edges is None:
        bin_edges = np.array([0.90, 1.10, 1.30, 1.50, 1.80])

    try:
        z_bin = find_bin(redshift, bin_edges, False)
        b1 = np.array([nuis_dict[f'b1_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        b2 = np.array([nuis_dict[f'b2_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        c0 = np.array([nuis_dict[f'c0_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        c2 = np.array([nuis_dict[f'c2_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        c4 = np.array([nuis_dict[f'c4_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        aP = np.array([nuis_dict[f'aP_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        Psn = np.array([nuis_dict[f'Psn_spectro_bin{i}']
                       for i in np.nditer(z_bin)])
        if np.isscalar(redshift):
            par_dict = {
                'b1': b1[0], 'b2': b2[0],
                'c0': c0[0], 'c2': c2[0], 'c4': c4[0],
                'aP': aP[0], 'Psn': Psn[0]
            }
        else:
            par_dict = {
                'b1': b1, 'b2': b2,
                'c0': c0, 'c2': c2, 'c4': c4,
                'aP': aP, 'Psn': Psn
            }
        return par_dict
    except ValueError:
        raise ValueError('Spectroscopic galaxy bias cannot be obtained. '
                         'Check that redshift is inside the bin edges and '
                         'valid bi_spec\'s are provided.')
    except KeyError:
        raise KeyError('Input nuisance parameter dictionary does not '
                       'contain one or more of the expected GCspectro '
                       'parameters.')
