"""likelihood_params_yaml_generator

Contains function that generates params.yaml
"""


import yaml
import os
import sys
from pathlib import Path


def generate_params_yaml(model=['nuisance_bias', 'nuisance_ia']):
    """
    Params Generator function.

    THIS IS A PROOF OF PRINCIPLE

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
    model: list of strings
        Tentative. Select corresponding models.
    """

    parent_path = str(
            Path(
                Path(__file__).resolve().parents[1]))

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
        # If bias and ia nuisance
        # are selected, bias and ia params are added
        params_path_list = [parent_path + '/Models/' + model_element + '.yaml'
                            for model_element in model]
        for params_path_element in params_path_list:
            try:
                with open(params_path_element) as file:
                    params_file = yaml.load(file, Loader=yaml.FullLoader)
                    likelihood_params.update(params_file)
            except OSError as err:
                print(
                    "File {0} not found. Error: {1}".format(
                        params_path_element, err))
                sys.exit(1)
            except BaseException:
                print("an unexpected error occurred")
                sys.exit(1)
    # In the future, new models will be added by more 'if' statements.
    else:
        print("ATTENTION: No other model is available." +
              "Please choose nuisance_bias or nuisance_ia.")

    params_path = parent_path + '/params.yaml'
    if os.path.exists(params_path):
        print('WARNING:\n')
        print("Be aware that {} has been overwritten".format(
            params_path))
    with open(params_path, 'w') as outfile:
        yaml.dump(likelihood_params, outfile, default_flow_style=False)
        print("{} written".format(params_path))
