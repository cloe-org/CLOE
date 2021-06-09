"""model_generator

Contains function that generates params.yaml
"""


import yaml
import os
from pathlib import Path


def params_generator(model=1):
    """
    Model generator function.

    THIS IS A PROOF OF PRINCIPLE

    Parameters
    ----------
    model: int
        Tentative. Select number corresponding to a model.
    """

    parent_path = str(
            Path(
                Path(__file__).resolve().parents[1]))

    # Right now the likelihood expects always the following params
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

        params_list = [nuisance_bias_path, nuisance_ia_path]
        for element in params_list:
            try:
                with open(element) as file:
                    element_file = yaml.load(file, Loader=yaml.FullLoader)
                    likelihood_params.update(element_file)
            except OSError as err:
                print("Cannot open {0}. Error: {1}".format(element,
                                                           err))
    else:
        print("ATTENTION: No other model is available. Please choose 1.")

    # added warning
    params_path = parent_path + '/params.yaml'
    if os.path.exists(params_path):
        print('WARNING:\n')
        print("Be aware that {} will be created or overwritten".format(
            params_path))
    with open(params_path, 'w') as outfile:
        yaml.dump(likelihood_params, outfile, default_flow_style=False)
        print("{} written".format(params_path))
