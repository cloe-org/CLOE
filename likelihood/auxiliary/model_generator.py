"""model_generator

Contains function that generates params.yaml
"""


import yaml
from pathlib import Path


def model_generator(model=1):
    """
    Model generator function

    THIS IS A TEST

    Parameters
    ----------
    model: int
        Tentative. Select number corresponding to a model
    """

    # Right now the likelihood expects always the following params
    nuisance_params = {
        'like_selection': 1,
        'full_photo': True,
        'NL_flag': False,
        'aia': 1.72,
        'nia': -0.41,
        'bia': 0.0}

    if model == 1:
        nuisance = str(
            Path(
                Path(__file__).resolve().parents[1])) \
            + '/Models/nuisance_bias.yaml'
        with open(nuisance) as file:
            model_file = yaml.load(file, Loader=yaml.FullLoader)
            # print(model_file)
            nuisance_params.update(model_file)
    params = str(Path(Path(__file__).resolve().parents[1])) + '/params.yaml'
    with open(params, 'w') as outfile:
        yaml.dump(nuisance_params, outfile, default_flow_style=False)
