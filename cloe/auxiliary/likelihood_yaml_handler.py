"""likelihood_yaml_handler

Contains functions to handle the yaml files and related dictionaries
in CLOE.
"""

from pathlib import Path
from copy import deepcopy
from cloe.auxiliary import yaml_handler
from cloe.auxiliary.logger import log_info


def get_default_configs_path():
    """
    Returns the default configs path.

    This function returns a path object with the
    default path of the configs directory.

    Returns
    -------
    models_path: Path
        the default path of the configs directory.
    """
    return Path(__file__).resolve().parents[2] / 'configs'


def get_default_models_path():
    """
    Returns the default models path.

    This function returns a path object with the
    default path of the models directory.

    Returns
    -------
    models_path: Path
        the default path of the models directory.
    """
    return get_default_configs_path() / 'models'


def get_default_params_yaml_path():
    """
    Returns the default path of params.yaml.

    This function returns a path object with the
    default path of params.yaml.

    Returns
    -------
    params_yaml_path: Path
        the default path of params.yaml.
    """
    return get_default_configs_path() / 'params.yaml'


def load_model_dict_from_yaml(file_name):
    """Loads user model dictionary from yaml file.

    Get a dictionary from a yaml file,
    checks that it matches a user model dictionary.

    Parameters
    ----------
    file_name: Path or str
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


def update_cobaya_params_from_model_yaml(cobaya_dict, file_name):
    """Updates Cobaya dictionary and params.yaml starting from a model yaml

    Notes: params.yaml will be updated only if the
    'overwrite' key in the model yaml is set to True.

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    file_name: Path or str
        The name of the user model yaml file.
    """

    model_dict = load_model_dict_from_yaml(file_name)
    params_dict = generate_params_dict_from_model_dict(model_dict, True)
    cobaya_dict['params'] = params_dict

    overwrite_params_yaml = model_dict['user_options']['overwrite']
    if overwrite_params_yaml is True:
        params_filepath = get_default_params_yaml_path()
        params_no_cosmo = get_params_dict_without_cosmo_params(params_dict)
        yaml_handler.yaml_write(params_filepath, params_no_cosmo, True)


def generate_params_dict_from_model_dict(model_dict,
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

    model_path = get_default_models_path()

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
        specific_dict = yaml_handler.yaml_read(full_filepath)
        params_dict.update(specific_dict)

    return params_dict


def write_params_yaml_from_model_dict(model_dict):
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

    params_filepath = get_default_params_yaml_path()
    overwrite = False

    options = model_dict['user_options']
    for key, value in options.items():
        if key == 'overwrite':
            overwrite = value

    param_dict = generate_params_dict_from_model_dict(model_dict, False)
    yaml_handler.yaml_write(params_filepath, param_dict, overwrite)


def write_params_yaml_from_info_dict(info_dict):
    """Writes the params yaml file from the info dictionary.

    The cosmological parameters are *excluded* in params.yaml.

    When invoking Cobaya with CLOE, CLOE can
    understand non-cosmology parameters
    only if they are defined in this params.yaml file.

    Parameters
    ----------
    info_dict: dict
        The user model dictionary.
    """
    if 'params' not in info_dict.keys():
        raise KeyError('No params subdictionary found in info dictionary.')

    params_filepath = get_default_params_yaml_path()
    params_dict = info_dict['params']
    params_no_cosmo = get_params_dict_without_cosmo_params(params_dict)
    yaml_handler.yaml_write(params_filepath, params_no_cosmo, True)


def update_cobaya_dict_with_halofit_version(cobaya_dict):
    """Updates the main cobaya dictionary with the halofit version to use

    The choice of the halofit_version key can be done only outside of the
    EuclidLikelihood class. This function reads the value
    of the nonlinear flag, and updates the cobaya dictionary
    accordingly.

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    """

    NL_flag = cobaya_dict['likelihood']['Euclid']['NL_flag_phot_matter']
    set_halofit_version(cobaya_dict, NL_flag)


def set_halofit_version(cobaya_dict: dict, NL_flag: int):
    """Sets the Halofit version of a cobaya dictionary according to a flag

    | The flag/Halofit relation is as follows:
    | NL_flag=0: not set (no request for Halofit to the Boltzman solver)
    | NL_flag=1: takahashi \
        (Ref: https://arxiv.org/abs/1208.2701)
    | NL_flag=2: mead2020 \
        (Ref: https://arxiv.org/abs/2009.01858)
    | NL_flag>2: mead2020 (current default version)

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    NL_flag: int
        The nonlinear flag
    """

    def switch_halofit_version(flag: int) -> str:
        switcher = {
            1: 'takahashi',
            2: 'mead2020'
        }
        return switcher.get(flag, 'mead2020')

    if NL_flag > 0:
        cobaya_dict['theory']['camb']['extra_args'][
            'halofit_version'] = switch_halofit_version(NL_flag)


def get_params_dict_without_cosmo_params(params_dict):
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

    cosmo_params = ['H0', 'tau', 'tau_reio', 'omk', 'Omega_k',
                    'ombh2', 'omega_b', 'omch2', 'omega_cdm',
                    'omnuh2', 'omega_ncdm', 'mnu', 'm_ncdm',
                    'nnu', 'N_eff',
                    'num_nu_massless', 'num_nu_massive', 'N_ur', 'N_ncdm',
                    'As', 'logA', 'A_s', 'ns', 'n_s',
                    'w', 'wa', 'w0_fld', 'wa_fld',
                    'sigma8', 'omegab', 'omegac', 'omeganu', 'omegam']

    for cosmo_param in cosmo_params:
        new_params_dict.pop(cosmo_param, None)

    return new_params_dict


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
        Possible strings: 'nuisance_bias', 'nuisance_ia', 'nuisance_nz',
        'nuisance_magnification_bias', 'nuisance_shear_calibration',
        'nonlinearities'

    Notes
    -----
    This function is deprecated.
    """

    models_path = get_default_models_path()

    if models is None:
        models = ['nuisance_bias', 'nuisance_ia', 'nuisance_nz',
                  'nuisance_magnification_bias', 'nuisance_shear_calibration',
                  'nonlinearities']
    likelihood_params = {}

    for model in models:
        model_path = str(models_path / model) + '.yaml'
        model_params = yaml_handler.yaml_read(model_path)
        likelihood_params.update(model_params)

    params_path = get_default_params_yaml_path()
    yaml_handler.yaml_write(params_path, likelihood_params, True)
    log_info('{} written'.format(params_path))
