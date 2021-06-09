"""likelihood_params_yaml_generator

Contains function that generates params.yaml
"""


import yaml
import os
from pathlib import Path


def generate_params_yaml(model=1):
    """
    Params Generator function.

    THIS IS A PROOF OF PRINCIPLE

    Parameters
    ----------
    model: int
        Tentative. Select number corresponding to a model.
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

    if model == 0:
        pass
    if model == 1:
        # If model = 1 is selected, bias and ia params are added
        nuisance_bias_path = parent_path + '/Models/nuisance_bias.yaml'
        nuisance_ia_path = parent_path + '/Models/nuisance_ia.yaml'
        params_path_list = [nuisance_bias_path, nuisance_ia_path]
        for params_path_element in params_path_list:
            try:
                with open(params_path_element) as file:
                    params_file = yaml.load(file, Loader=yaml.FullLoader)
                    likelihood_params.update(params_file)
            except OSError as err:
                print("Cannot open {0}. Error: {1}".format(params_path,
                                                           err))
    else:
        print("ATTENTION: No other model is available. Please choose 1.")

    params_path = parent_path + '/params.yaml'
    if os.path.exists(params_path):
        print('WARNING:\n')
        print("Be aware that {} will be overwritten".format(
            params_path))
    with open(params_path, 'w') as outfile:
        yaml.dump(likelihood_params, outfile, default_flow_style=False)
        print("{} written".format(params_path))
