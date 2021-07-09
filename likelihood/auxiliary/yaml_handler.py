# -*- coding: utf-8 -*-
"""YAML HANDLER

Functions for reading and writing yaml files, these are completely transparent
to the CLOE-specific configuration.
"""

import yaml


def yaml_read(file_name):
    r"""Read a stream from a yaml file.

    Parameters
    ----------
    file_name: str
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


def yaml_write(file_name, config):
    r"""Write a dictionary to a yaml file.

    Parameters
    ----------
    file_name: str
        The name of the file the stream will be written to.
    config: dict
        The dictionary to be written to file

    Raises
    ------
    TypeError
       if the config input parameter is not a dict

    Notes
    -----
    The writing is performed using the python builtin i/o functions and the
    pyyaml package.
    """
    if type(config) is not dict:
        raise TypeError('Input configuration is not a dict: {config}')
    with open(file_name, 'w') as file:
        file.write(yaml.dump(config))
