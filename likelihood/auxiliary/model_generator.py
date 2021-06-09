"""model_generator

Contains function that generates params.yaml
"""


import yaml
from pathlib import Path


def model_generator(model=1):
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
        try:
            with open(nuisance_bias_path) as file:
                nuisance_bias_file = yaml.load(file, Loader=yaml.FullLoader)
                likelihood_params.update(nuisance_bias_file)
        except OSError as err:
            print("Cannot open {0}. Error: {1}".format(nuisance_bias_path,
                                                       err))
        try:
            with open(nuisance_ia_path) as file:
                nuisance_ia_file = yaml.load(file, Loader=yaml.FullLoader)
                likelihood_params.update(nuisance_ia_file)
        except OSError as err:
            print("Cannot open {0}. Error: {1}".format(nuisance_ia_path,
                                                       err))
    else:
        print("ATTENTION: No other model is available. Please choose 1.")

    # added warning
    params_path = parent_path + '/params.yaml'
    print('WARNING:\n')
    print("Be aware that {} will be created or overwritten".format(
        params_path))
    with open(params_path, 'w') as outfile:
        yaml.dump(likelihood_params, outfile, default_flow_style=False)
