# -*- coding: utf-8 -*-
"""YAML HANDLER

Functions for reading and writing yaml files, these are completely transparent
to the CLOE-specific configuration.
"""

import yaml
from warnings import warn
from os.path import exists


def yaml_read(file_name):
    r"""Read a stream from a yaml file.

    Parameters
    ----------
    file_name: Path or str
        The name of the file where to read the configuration from.

    Returns
    -------
    dict
       The configuration read from the input file

    Notes
    -----
    The reading is performed using the python builtin i/o functions and the
    pyyaml package.
    Any error checking is delegated to these packages.
    In this function we use the yaml.load() with Loader=yaml.FullLoader:
    according to the pyyaml documentation could be dangerous if used on
    files from untrusted sources.
    """

    with open(file_name, 'r') as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)


def yaml_read_and_check_dict(file_name, needed_keys: list):
    r"""Read a stream from a yaml file and check the dictionary.

    Get a dictionary from a yaml file,
    checks that the dictionary contains the specified needed keys.

    Parameters
    ----------
    file_name: Path or str
        The name of the file where to read the dictionary from.
    needed_keys: list of str
        The keys that must be in the dictionary

    Returns
    -------
    dictionary: dict
        The dictionary read from the input file

    Raises
    ------
    TypeError
        if a dictionary is not read from the yaml file
    KeyError
        if any of the needed keys are not present in the dictionary.
    """

    dictionary = yaml_read(file_name)

    if type(dictionary) is not dict:
        raise TypeError(f'File {file_name} not formatted as dictionary')

    for key in needed_keys:
        if key not in dictionary:
            raise KeyError(f'key \'{key}\' not found in {file_name}')

    return dictionary


def yaml_write(file_name, config, overwrite=False):
    r"""Write a dictionary to a yaml file.

    Parameters
    ----------
    file_name: Path or str
        The name of the file the stream will be written to.
    config: dict
        The dictionary to be written to file
    overwrite: bool
        Overwrite the output file, if already exists?

    Raises
    ------
    TypeError
       if the config input parameter is not a dict
    RuntimeError
        if the output file already exists and overwrite is set to False

    Notes
    -----
    The writing is performed using the python builtin i/o functions and the
    pyyaml package.
    """

    if type(config) is not dict:
        raise TypeError('Input configuration is not a dict: {config}')

    file_exists = exists(file_name)
    if file_exists and not overwrite:
        raise RuntimeError(f'File {file_name} already exists.\n')
    elif file_exists and overwrite:
        warn(f'Overwriting file {file_name}.')

    with open(file_name, 'w') as file:
        file.write(yaml.dump(config, default_flow_style=False))
