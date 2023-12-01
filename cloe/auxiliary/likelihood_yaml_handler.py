"""LIKELIHOOD YAML HANDLER

Contains functions to handle the :obj:`yaml` files and
related dictionaries in CLOE.
"""


from pathlib import Path
from copy import deepcopy
from cloe.auxiliary import yaml_handler
from cloe.auxiliary.logger import log_info, log_warning


def get_default_configs_path():
    """
    Returns the default configs path.

    This function returns a path object with the
    default path of the configs directory.

    Returns
    -------
    Models path: pathlib.PosixPath
        The default path of the configs directory
    """
    return Path(__file__).resolve().parents[2] / 'configs'


def get_default_models_path():
    """
    Returns the default models path.

    This function returns a path object with the
    default path of the models directory.

    Returns
    -------
    Models path: pathlib.PosixPath
        The default path of the models directory
    """
    return get_default_configs_path() / 'models'


def get_default_params_yaml_path():
    """
    Returns the default path of :obj:`params.yaml`.

    This function returns a path object with the
    default path of :obj:`params.yaml`.

    Returns
    -------
    Parameters yaml path: pathlib.PosixPath
        The default path of :obj:`params.yaml`
    """
    return get_default_configs_path() / 'params.yaml'


def load_model_dict_from_yaml(file_name):
    """Loads user model dictionary from :obj:`yaml` file.

    Gets a dictionary from a :obj:`yaml` file,
    checks that it matches a user model dictionary.

    Parameters
    ----------
    file_name: pathlib.PosixPath or str
        The name of the :obj:`yaml` file where to read the dictionary from

    Returns
    -------
    Model dictionary: dict
       The user model dictionary read from the input file
    """

    needed_keys = ['user_models', 'user_options']
    model_dict =\
        yaml_handler.yaml_read_and_check_dict(file_name, needed_keys)

    return model_dict


def update_cobaya_params_from_model_yaml(cobaya_dict, file_name):
    """Updates Cobaya dictionary and :obj:`params.yaml`.

    Notes: :obj:`params.yaml` will be updated only if the
    overwrite key in the model :obj:`yaml` is set to True.

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    file_name: pathlib.PosixPath or str
        The name of the user model :obj:`yaml` file
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
    """Generates the parameters dictionary from a user model dictionary.

    Cobaya requests parameters defined in the theory
    code (i.e: CAMB/CLASS and cosmological parameters)
    and also parameters defined by the likelihood
    (i.e: CLOE and nuisance parameters).

    The function can also generate a params dictionary without
    the cosmological parameters.

    Parameters
    ----------
    model_dict: dict
        The user model dictionary

    include_cosmology: bool
        If true, the cosmological parameters (such as :math:`H_0, n_s` ...)
        are included in the parameters dictionary

    Returns
    -------
    Parameters dictionary: dict
        The parameters dictionary
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
        if (specific_dict is not None):
            params_dict.update(specific_dict)
        else:
            log_warning('{}.yaml is empty. If needed, parameters '
                        'will be set to default values.'.format(key))

    return params_dict


def write_params_yaml_from_model_dict(model_dict):
    """Writes the parameters :obj:`yaml` file from the model dictionary.

    The cosmological parameters are excluded in :obj:`params.yaml`.

    When invoking Cobaya with CLOE, CLOE can
    understand non-cosmology parameters
    only if they are defined in this :obj:`params.yaml` file.

    Parameters
    ----------
    Model dictionary: dict
        The user model dictionary
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
    """Writes the parameters :obj:`yaml` file from the info dictionary.

    The cosmological parameters are excluded in :obj:`params.yaml`.

    When invoking Cobaya with CLOE, CLOE can
    understand non-cosmology parameters
    only if they are defined in this :obj:`params.yaml` file.

    Parameters
    ----------
    info_dict: dict
        The user model dictionary
    """
    if 'params' not in info_dict.keys():
        raise KeyError('No params subdictionary found in info dictionary.')

    params_filepath = get_default_params_yaml_path()
    params_dict = info_dict['params']
    params_no_cosmo = get_params_dict_without_cosmo_params(params_dict)
    yaml_handler.yaml_write(params_filepath, params_no_cosmo, True)


def update_cobaya_dict_with_halofit_version(cobaya_dict):
    """Updates the main Cobaya dictionary with the Halofit version to use.

    The choice of the :obj:`halofit_version` key can be done only
    outside of the :obj:`Euclike` class. This function reads the value
    of the nonlinear flag, and updates the Cobaya dictionary
    accordingly.

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    """

    NL_flag = cobaya_dict['likelihood']['Euclid']['NL_flag_phot_matter']
    Baryon_flag = cobaya_dict['likelihood']['Euclid']['NL_flag_phot_baryon']
    set_halofit_version(cobaya_dict, NL_flag, Baryon_flag)


def set_halofit_version(cobaya_dict: dict, NL_flag: int, Baryon_flag: int):
    r"""Sets the Halofit version of a cobaya dictionary according to two flags

    Selection is done via the conversion of the nonlinear and baryonic feedback
    flags into the overall cobaya_flag.

    | The flag/Halofit relation depends on the combined cobaya_flag
    | cobaya_flag=0: not set (no request for Halofit to the Boltzman solver)
    | cobaya_flag=1: takahashi \
        (Ref: https://arxiv.org/abs/1208.2701)
    | cobaya_flag=2: mead2016 \
        (Ref: https://arxiv.org/abs/1602.02154)
    | cobaya_flag=3: mead2020 \
        (Ref: https://arxiv.org/abs/2009.01858)
    | cobaya_flag=4: mead2020_feedback \
        (Ref: https://arxiv.org/abs/2009.01858)

    This flag is determined from the following table

    +--------------------+-----+-----+-----+-----+-----+
    | NL_flag \ Baryon_flag |  0  |  1  |  2  |  3  |  4  |
    +--------------------+-----+-----+-----+-----+-----+
    |          0         |  0  |  0  |  0  |  0  |  0  |
    +--------------------+-----+-----+-----+-----+-----+
    |          1         |  1  |  1  |  1  |  1  |  1  |
    +--------------------+-----+-----+-----+-----+-----+
    |          2         |  2  |  2  |  4  |  2  |  2  |
    +--------------------+-----+-----+-----+-----+-----+
    |          3         |  3  |  2  |  4  |  3  |  3  |
    +--------------------+-----+-----+-----+-----+-----+
    |          4         |  3  |  2  |  4  |  3  |  3  |
    +--------------------+-----+-----+-----+-----+-----+
    |          5         |  3  |  2  |  4  |  3  |  3  |
    +--------------------+-----+-----+-----+-----+-----+

    A few additional details:

    When NL_flag=0 nothing is requested from cobaya, as we treat
    everything with linear theory.

    For halofit (NL_flag=1) we always request halofit (cobaya_flag=1) from
    cobaya, since that is currently not available externally.

    In all other cases, the priority model to get from cobaya depends on which
    baryonic feedback model is requested, so e.g. if mead2020_feedback is
    requested by choosing Baryon_flag=2, we always get that from cobaya
    (cobaya_flag=4).

    When a matter power spectrum emulator is chosen we always request Mead2020
    from cobaya for extrapolation (unless we require a different baryonic
    feedback model). When emulators are selected we always request the same
    options from cobaya, which is the reason why the two last rows are the same
    (for EE2 and Bacco as matter models), as are the two last columns (for
    BCemu and Bacco baryons).

    Finally, note that this function only controls what is requested from
    cobaya, as the full model used is then built in the nonlinear module based
    on what the user requested, using the matter power spectrum provided by
    cobaya as one of its components. For example, while we always request
    halofit from cobaya when we have halofit as the matter model, the correct
    baryonic feedback model is then obtained inside the nonlinear module. In
    addition, when two different versions of HMcode are selected, the matter
    power spectrum is always calculated inside the nonlinear module, while the
    baryon correction is taken from cobaya.

    Parameters
    ----------
    cobaya_dict: dict
        The Cobaya dictionary
    NL_flag: int
        The nonlinear flag
    Baryon_flag: int
        The baryonic feedback model flag
    """

    if (NL_flag == 0 and Baryon_flag > 0):
        log_warning("You selected a non-zero NL_flag_phot_baryon "
                    "value, while selecting NL_flag_phot_matter = 0. Selected "
                    "baryonic feedback model will be ignored and every "
                    "prediciton will be linear.")

    # Matrix determining value to pass to switcher depending on NL and Bar flag
    # Rows correpond to different NL_flag (0 to 5), columns to different
    # Baryon_flag (0 to 4). Assumed to give 0 whenever NL_flag=0
    NL_Bar_matrix = [[0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 4, 2, 2],
                     [3, 2, 4, 3, 3],
                     [3, 2, 4, 3, 3],
                     [3, 2, 4, 3, 3]]

    cobaya_flag = NL_Bar_matrix[NL_flag][Baryon_flag]

    def switch_halofit_version_in_camb(flag: int) -> str:
        switcher = {
            1: 'takahashi',
            2: 'mead2016',
            3: 'mead2020',
            4: 'mead2020_feedback'
        }
        return switcher.get(flag, 'mead2020')

    def switch_halofit_version_in_classy(flag: int) -> str:
        switcher = {
            1: 'halofit',
            2: 'hmcode',
        }
        if flag in switcher.keys():
            return switcher.get(flag)
        else:
            raise ValueError('The only available options for '
                             'NL_flag_phot_matter when classy is selected '
                             'are 1 (Halofit) and 2 (HMCode2016)')

    if cobaya_flag > 0:
        solver = cobaya_dict['likelihood']['Euclid']['solver']
        if solver == 'camb':
            cobaya_dict['theory']['camb']['extra_args']['halofit_version'] = \
                switch_halofit_version_in_camb(cobaya_flag)
        elif solver == 'classy':
            cobaya_dict['theory']['classy']['extra_args']['non_linear'] = \
                switch_halofit_version_in_classy(cobaya_flag)

    params = cobaya_dict['params']
    if Baryon_flag != 1 and any([par in params.keys() for par in
                                 ['HMCode_A_baryon',
                                  'HMCode_eta_baryon']]) and \
        (params['HMCode_A_baryon'] != 3.13 or
         params['HMCode_eta_baryon'] != 0.603):
        params['HMCode_A_baryon'] = 3.13
        params['HMCode_eta_baryon'] = 0.603
        log_warning('Parameters [HMCode_A_baryon, HMCode_eta_baryon] are '
                    'used only for the Mead2016 baryon feedback model '
                    '(Baryon_flag=1) but have been set to non-default values. '
                    'Setting them to the default values for matter.')


def get_params_dict_without_cosmo_params(params_dict):
    """Parameters dictionary without the cosmological parameters.

    Note: the input params dictionary is NOT modified.

    Parameters
    ----------
    params_dict: dict
        The parameters dictionary (it is not modified)

    Returns
    -------
    New parameters dictionary: dict
       A deep copy of the input :obj:`params_dict`,
       stripped of cosmological parameters

    Raises
    ------
    ValueError
        If params_dict is None
    TypeError
        If params_dict is not a dict
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

    HMCode_params = ['HMCode_A_baryon', 'HMCode_eta_baryon', 'HMCode_logT_AGN']

    for cosmo_param in cosmo_params:
        new_params_dict.pop(cosmo_param, None)
    for HMCode_param in HMCode_params:
        new_params_dict.pop(HMCode_param, None)

    return new_params_dict
