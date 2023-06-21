# -*- coding: utf-8 -*-
"""YAML HANDLER

Functions for reading and writing :obj:`yaml` files, these are completely
transparent to the CLOE-specific configuration.
"""

import yaml
from warnings import warn
from os.path import exists


def yaml_read(file_name):
    r"""Reads a stream from a yaml file.

    Parameters
    ----------
    file_name: str
        The name of the file where to read the configuration from

    Returns
    -------
    Dictionary: dict
       The configuration read from the input file

    Notes
    -----
    The reading is performed using the python built-in :obj:`io`
    functions and the :obj:`PyYAML` package.
    Any error checking is delegated to these packages.
    In this function we use the :obj:`yaml.load()` with
    :obj:`Loader=yaml.FullLoader` according to the :obj:`PyYAML`
    documentation could be dangerous if used on files from
    untrusted sources.
    """

    with open(file_name, 'r') as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)


def yaml_read_and_check_dict(file_name, needed_keys: list):
    r"""Reads a stream from a :obj:`yaml` file and check the dictionary.

    Gets a dictionary from a :obj:`yaml` file,
    checks that the dictionary contains the specified needed keys.

    Parameters
    ----------
    file_name: str
        The name of the file where to read the dictionary from
    needed_keys: list, str
        The keys that must be in the dictionary

    Returns
    -------
    Dictionary: dict
        The dictionary read from the input file

    Raises
    ------
    TypeError
        If a dictionary is not read from the :obj:`yaml` file
    KeyError
        If any of the needed keys are not present in the dictionary.
    """

    dictionary = yaml_read(file_name)

    if type(dictionary) is not dict:
        raise TypeError(f'File {file_name} not formatted as dictionary')

    for key in needed_keys:
        if key not in dictionary:
            raise KeyError(f'key \'{key}\' not found in {file_name}')

    return dictionary


def yaml_write(file_name, config, overwrite=False):
    r"""Writes a dictionary to a yaml file.

    Parameters
    ----------
    file_name: str
        The name of the file the stream will be written to
    config: dict
        The dictionary to be written to file
    overwrite: bool
        Overwrites the output file, if already exists

    Raises
    ------
    TypeError
       If the config input parameter is not a dictionary
    RuntimeError
        If the output file already exists and overwrite is set to
        :obj:`False`

    Notes
    -----
    The writing is performed using the python built-in :obj:`io` functions
    and the :obj:`PyYAML` package
    """

    if type(config) is not dict:
        raise TypeError('Input configuration is not a dict: {config}')

    file_exists = exists(file_name)
    if file_exists and not overwrite:
        raise RuntimeError(f'File {file_name} already exists.\n')
    elif file_exists and overwrite:
        warn(f'Overwriting file {file_name}.')

    with open(file_name, 'w', encoding='utf8') as file:
        file.write(
            yaml.dump(config, default_flow_style=False),
        )
