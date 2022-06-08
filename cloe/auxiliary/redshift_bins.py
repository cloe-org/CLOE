"""redshift_bins

Contains common function to operate
on redshift bin edges
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
