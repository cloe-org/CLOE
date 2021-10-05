"""likelihood_yaml_handler

Contains functions to handle the yaml files and related dictionaries
in CLOE.
"""

from pathlib import Path
from copy import deepcopy
from likelihood.auxiliary import yaml_handler


def generate_params_yaml(models=None):
    """Generates params.yaml from a model list.

    Cobaya requests parameters defined in the theory
    code (i.e: CAMB/CLASS and the LCDM parameters)
    and also parameters defined by the likelihood
    (i.e: CLOE and nuisance parameters).

    When invoking Cobaya with CLOE, CLOE will
    understand cosmology parameters but not any
    extra parameter (such as nuisance or flags)
    unless they are defined either in the `cobaya_interface.py`
    or in a 'params.yaml' file.

    This function creates the 'params.yaml' file so that
    Cobaya understands that CLOE requests some
    extra parameters.

    Parameters
    ----------
    models: list of strings
        Strings corresponding to a model.
    """

    likelihood_path = Path(__file__).resolve().parents[1]
    models_path = likelihood_path / 'config' / 'models'

    if models is None:
        models = ['nuisance_bias', 'nuisance_ia',
                  'spectro', 'likelihood_flags']
    likelihood_params = {}

    for model in models:
        model_path = str(models_path / model) + '.yaml'
        model_params = yaml_handler.yaml_read(model_path)
        likelihood_params.update(model_params)

    params_path = str(likelihood_path / 'params.yaml')
    yaml_handler.yaml_write(params_path, likelihood_params, True)
    print('{} written'.format(params_path))


def load_model_dict_from_yaml(file_name: str):
    """Loads user model dictionary from yaml file.

    Get a dictionary from a yaml file,
    checks that it matches a user model dictionary.

    Parameters
    ----------
    file_name: str
        The name of the yaml file where to read the dictionary from.

    Returns
    -------
    model_dict: dict
       The user model dictionary read from the input file
    """

    needed_keys = ['user_models', 'user_options']
    model_dict =\
        yaml_handler.yaml_read_and_check_dict(file_name, needed_keys)

    return model_dict


def write_params_yaml_from_cobaya_dict(cobaya_dict: dict, file_path=None):
    """Writes the params yaml file from the Cobaya dictionary

    The cosmological parameters are *excluded* in params.yaml.
    Note: the input Cobaya dictionary is NOT modified

    When invoking Cobaya with CLOE, CLOE can
    understand non-cosmology parameters
    only if they are defined in this params.yaml file.

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary (it is not modified)
    file_path: str
        The output file path (default: likelihood/params.yaml)
    """
    params_dict = cobaya_dict['params']
    params_without_cosmo = get_params_dict_without_cosmo_params(params_dict)

    if file_path is None:
        file_path = Path(__file__).resolve().parents[1] / 'params.yaml'
    yaml_handler.yaml_write(str(file_path), params_without_cosmo, True)


def get_params_dict_without_cosmo_params(params_dict: dict):
    """Returns params dict without the cosmological parameters.

    Note: the input params dictionary is NOT modified.

    Parameters
    ----------
    params_dict: dict
        The params dictionary (it is not modified).

    Returns
    -------
    new_params_dict: dict
       A deep copy of the input params_dict,
       stripped of cosmological parameters

    Raises
    ------
    ValueError
        if params_dict is None
    TypeError
        if params_dict is not a dict
    """

    if params_dict is None:
        raise ValueError('Empty params dictionary')
    elif type(params_dict) is not dict:
        raise TypeError('Params dictionary is not a dict')

    new_params_dict = deepcopy(params_dict)

    cosmo_params = ['As', 'logA', 'H0', 'N_eff', 'mnu', 'mnu', 'ns', 'sigma8',
                    'ombh2', 'omch2', 'omnuh2', 'omegam',
                    'omegab', 'omegac', 'omeganu', 'omk', 'tau', 'w', 'wa']

    for cosmo_param in cosmo_params:
        new_params_dict.pop(cosmo_param, None)

    return new_params_dict


def generate_params_dict_from_model_dict(model_dict: dict,
                                         include_cosmology=True):
    """Generates the params dictionary from a user model dictionary

    Cobaya requests parameters defined in the theory
    code (i.e: CAMB/CLASS and cosmological parameters)
    and also parameters defined by the likelihood
    (i.e: CLOE and nuisance parameters).

    The function can also generate a params dictionary without
    the cosmological parameters.

    Parameters
    ----------
    model_dict: dict
        The user model dictionary.

    include_cosmology: bool
        If true, the cosmological parameters (such as :math:`H_0, n_s` ...)
        are included in the params dictionary.

    Returns
    -------
    params_dict: dict
        The params dictionary
    """

    if model_dict is None:
        raise ValueError('Empty model dictionary')

    model_path = \
        Path(__file__).resolve().parents[2] / 'configs' / 'models'

    options = model_dict['user_options']
    for key, value in options.items():
        if key == 'model_path':
            model_path = Path(value)

    model = model_dict['user_models']
    if model is None:
        raise ValueError('Empty user model dictionary')
    elif type(model) is not dict:
        raise TypeError('User model dictionary is not a dict')

    params_dict = {}
    for key, filename in model.items():
        if key == 'cosmology' and include_cosmology is False:
            continue
        full_filepath = (model_path / Path(filename)).resolve()
        specific_dict = yaml_handler.yaml_read(str(full_filepath))
        params_dict.update(specific_dict)

    return params_dict


def generate_params_dict_from_model_yaml(file_name: str):
    """Generates the params dictionary from the model yaml file.

    The cosmological parameters are *included* in the dictionary.

    Parameters
    ----------
    file_name: str
        The name of the user model yaml file.

    Returns
    -------
    params_dict: dict
        The params dictionary (including cosmological parameters)
    """

    model_dict = load_model_dict_from_yaml(file_name)
    return generate_params_dict_from_model_dict(model_dict, True)


def write_params_yaml_from_model_dict(model_dict: dict):
    """Writes the params yaml file from the model dictionary.

    The cosmological parameters are *excluded* in params.yaml.

    When invoking Cobaya with CLOE, CLOE can
    understand non-cosmology parameters
    only if they are defined in this params.yaml file.

    Parameters
    ----------
    model_dict: dict
        The user model dictionary.
    """

    params_filepath = Path(__file__).resolve().parents[1] / 'params.yaml'
    overwrite = False

    options = model_dict['user_options']
    for key, value in options.items():
        if key == 'out_params_yaml_path':
            params_filepath = Path(value).resolve()
        elif key == 'overwrite':
            overwrite = value

    param_dict = generate_params_dict_from_model_dict(model_dict, False)
    yaml_handler.yaml_write(str(params_filepath), param_dict, overwrite)


def write_params_yaml_from_model_yaml(file_name: str):
    """Writes the params yaml file from the model yaml file.

    The cosmological parameters are *excluded* in params.yaml.

    Parameters
    ----------
    file_name: str
        The name of the user model yaml file.
    """

    model_dict = load_model_dict_from_yaml(file_name)
    write_params_yaml_from_model_dict(model_dict)


def update_cobaya_dict_from_model_yaml(cobaya_dict: dict, file_name: str):
    """Updates a Cobaya dictionary starting from the model yaml file

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    file_name: str
        The name of the user model yaml file.

    Returns
    -------
    cobaya_dict: dict
        The updated Cobaya dictionary.
    """

    params_dict = generate_params_dict_from_model_yaml(file_name)
    cobaya_dict['params'] = params_dict
    return cobaya_dict


def write_data_yaml_from_data_dict(data: dict):
    """Writes data.yaml from the data dictionary.

    The Cobaya interface requires a data dictionary storing paths
    to data files. This function saves the data dictionary to a yaml file,
    i.e. data.yaml, which is subsequently called inside EuclidLikelihood.yaml

    Parameters
    ----------
    data: dict
        Dictionary containing specifications for data loading and handling.
    """

    parent_path = Path(__file__).resolve().parents[1]
    data_file = str(parent_path / 'data.yaml')
    yaml_handler.yaml_write(data_file, data, overwrite=True)
    print('Written data file: {}'.format(data_file))
