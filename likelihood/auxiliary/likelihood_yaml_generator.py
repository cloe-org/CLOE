"""likelihood_yaml_generator

Contains functions that generate params.yaml and data.yaml
"""

from pathlib import Path
from likelihood.auxiliary import yaml_handler
import yaml
import os
import sys
import warnings


def generate_params_yaml(model=['nuisance_bias', 'nuisance_ia']):
    """
    Params Generator function.

    THIS IS A PROOF OF PRINCIPLE
    Sept 7th, 2021:
    THIS FUNCTION IS *TEMPORARLY* KEPT FOR COMPATIBILITY,
    UNTIL THE DEMO AND UI ARE UPDATED WITH THE NEW FUNCTIONS

    Cobaya requests parameters defined in the theory
    code (i.e: CAMB/CLASS and the LCDM parameters)
    and also parameters defined by the likelihood
    (i.e: CLOE and nuisance parameters).

    When invoking Cobaya with CLOE, CLOE will
    understand LCDM parameters but not the
    nuisance parameters unless they are defined
    either in the `cobaya_interface.py` or in a
    yaml file.

    This function creates that yaml file so that
    Cobaya understands that CLOE requests some
    nuisance parameters.

    Parameters
    ----------
    model: int
        Tentative. Select number corresponding to a model.
    """

    parent_path = str(Path(Path(__file__).resolve().parents[1]))
    # As June 2021, CLOE V1.0 the likelihood
    # expects always the following params
    likelihood_params = {
        'like_selection': 1,
        'full_photo': True,
        'NL_flag': False}

    if not model:
        print('ATTENTION: no model was selected')
        pass
    if model == ['nuisance_bias', 'nuisance_ia']:
        # If model = 1 is selected, bias and ia params
        # and spec multipoles are added
        nuisance_bias_path = parent_path + '/Models/nuisance_bias.yaml'
        nuisance_ia_path = parent_path + '/Models/nuisance_ia.yaml'
        spec_path = parent_path + '/Models/spec.yaml'
        params_path_list = [nuisance_bias_path, nuisance_ia_path,
                            spec_path]
        for params_path_element in params_path_list:
            try:
                with open(params_path_element) as file:
                    params_file = yaml.load(file, Loader=yaml.FullLoader)
                    likelihood_params.update(params_file)
            except OSError as err:
                print('File {0} not found. Error: {1}'.
                      format(params_path_element, err))
                sys.exit(1)
            except BaseException:
                print('an unexpected error occurred')
                sys.exit(1)
    else:
        print("ATTENTION: No other model is available." +
              "Please choose nuisance_bias or nuisance_ia.")

    params_path = parent_path + '/params.yaml'
    if os.path.exists(params_path):
        warnings.warn(
            'Be aware that {} has been overwritten'.format(params_path))
    with open(params_path, 'w') as outfile:
        yaml.dump(likelihood_params, outfile, default_flow_style=False)
        print('{} written'.format(params_path))


def load_model_dict_from_yaml(file_name: str):
    """Load user model dictionary from yaml file.

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


def load_cobaya_dict_from_yaml(file_name: str):
    """Load Cobaya dictionary from yaml file.

    Get a dictionary from a yaml file,
    checks that it matches a Cobaya dictionary

    Parameters
    ----------
    file_name: str
        The name of the yaml file where to read the dictionary from.

    Returns
    -------
    cobaya_dict: dict
       The cobaya dictionary read from the input file
    """

    needed_keys = \
        ['force', 'likelihood', 'output', 'params', 'sampler', 'theory']
    cobaya_dict =\
        yaml_handler.yaml_read_and_check_dict(file_name, needed_keys)

    return cobaya_dict


def generate_params_dict_from_model_dict(model_dict: dict,
                                         include_cosmology=True):
    r"""Generate the params dictionary from a user model dictionary

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

    model_path = Path(__file__).resolve().parents[1] / 'Models'

    options = model_dict['user_options']
    for key, value in options.items():
        if key == 'model_path':
            model_path = Path(value)

    params_dict = {}
    model = model_dict['user_models']
    for key, filename in model.items():
        if key == 'cosmology' and include_cosmology is False:
            continue
        full_filepath = (model_path / Path(filename)).resolve()
        specific_dict = yaml_handler.yaml_read(str(full_filepath))
        params_dict.update(specific_dict)

    return params_dict


def generate_params_dict_from_model_yaml(file_name: str):
    """Generate the params dictionary from the model yaml file

    This function returns the param dictionary
    *including* the cosmological parameters.

    Parameters
    ----------
    file_name: str
        The name of the user model yaml file.

    Returns
    -------
    params_dict: dict
        The params dictionary
    """

    model_dict = load_model_dict_from_yaml(file_name)
    return generate_params_dict_from_model_dict(model_dict, True)


def write_params_yaml_from_model_dict(model_dict: dict):
    """Write the params yaml file from the model dictionary

    This function writes the params yaml
    *excluding* the cosmological parameters.

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
    """Write the params yaml file from the model yaml file

    This function writes the params yaml
    *excluding* the cosmological parameters.

    Parameters
    ----------
    file_name: str
        The name of the user model yaml file.
    """

    model_dict = load_model_dict_from_yaml(file_name)
    write_params_yaml_from_model_dict(model_dict)


def update_cobaya_dict_with_params_dict(cobaya_dict: dict,
                                        params_dict: dict):
    """Update the Cobaya dictionary with the params dictionary

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary.
    params_dict: dict
        The params dictionary.

    Returns
    -------
    cobaya_dict: dict
        The Cobaya dictionary with the updated params dictionary
    """

    cobaya_dict['params'] = params_dict
    return cobaya_dict


def generate_cobaya_dict_from_model_yaml(file_name: str):
    """
    Generate a Cobaya dictionary starting from the model yaml file

    Parameters
    ----------
    file_name: str
        The name of the user model yaml file.

    Returns
    -------
    cobaya_dict: dict
        The Cobaya dictionary.
    """

    cobaya_shell_path = Path(__file__).resolve().parents[2] \
        / '/likelihood/config_yaml/cobaya_shell.yaml'

    model_dict = load_model_dict_from_yaml(file_name)

    options = model_dict['user_options']
    for key, value in options.items():
        if key == 'in_cobaya_shell_path':
            cobaya_shell_path = Path(value).resolve()

    cobaya_dict = load_cobaya_dict_from_yaml(str(cobaya_shell_path))
    params_dict = generate_params_dict_from_model_dict(model_dict)

    return update_cobaya_dict_with_params_dict(cobaya_dict, params_dict)


def generate_data_yaml(data):
    """
    Data Generator function.

    The Cobaya interface requires a data dictionary storing paths
    to data files. This function saves the data dictionary to a yaml file,
    e.g. data.yaml, which is subsequently called inside EuclidLikelihood.yaml

    Parameters
    ----------
    data: dict
        Dictionary containing specifications for data loading and handling.
    """

    parent_path = str(Path(Path(__file__).resolve().parents[1]))
    data_path = parent_path + '/data.yaml'
    if os.path.exists(data_path):
        warnings.warn(
            'Be aware that {} has been overwritten'.format(data_path))

    with open(data_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        print('{} written'.format(data_path))
