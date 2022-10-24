"""TEST DATA HANDLER

This module contains functions for handling the IO of unit test data products.

"""

import numpy as np
import pickle
from os.path import join
from pathlib import Path

# Set default path for unit test data
default_path = join(Path(__file__).resolve().parents[1], 'test_input')


def load_test_npy(file_name, path=None):
    """Load Test Numpy Binary

    Load Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of Numpy binary file
    path : str, optional
        Path to test data

    Returns
    -------
    numpy.ndarray
        Numpy array

    Raises
    ------
    ValueError
        For unknown file extension

    """
    if not path:
        path = default_path

    file_path = join(path, file_name)

    if file_name.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    elif file_name.endswith('.dat'):
        return np.loadtxt(file_path)
    else:
        raise ValueError('Unknown file extension for Numpy test data.')


def load_test_pickle(file_name, path=None):
    """Load Test Pickle

    Load Python pickle file from the ``test_input`` directory.

    Parameters
    ----------
    file_name : str
        Name of Python pickle file
    path : str, optional
        Path to test data

    Returns
    -------
    object
        Python object (e.g. a dictionary)

    """
    if not path:
        path = default_path

    with open(join(path, file_name), 'rb') as pickle_file:
        unpickled_object = pickle.load(pickle_file)

    return unpickled_object


def save_test_pickle(file_name, object, path=None):
    """Dump Test Pickle

    Save Python pickle file to specified path.

    Parameters
    ----------
    file_name : str
        Name of Python pickle file with full path
    object : object
        A Python object (e.g. a dictionary)
    path : str, optional
        Path to test data

    """
    if not path:
        path = default_path

    with open(join(path, file_name), 'wb') as pickle_file:
        pickle.dump(object, pickle_file)
