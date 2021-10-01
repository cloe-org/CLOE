"""observables_dealer

Contains function to read the observable dictionary
and plot the visualization of the matrix
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings


def observables_visualization(observables_dict, palette='standard'):
    """
    Observables visualization

    Takes the observables dictionary read by Cobaya
    and shows the visualization of the matrix

    Parameters
    ----------
    observables_dict: dict
        dictionary with the observables selection
    palette: string
        name of the visualiation color palette
        Choose between 'standard', 'protanopia',
        'deuteranopia'
    """
    # Checks if everything looks nice
    observables_dict_checked = observables_selection_checker(observables_dict)

    observables_df = pd.DataFrame(
        observables_dict_checked).fillna(-1).astype(int).T

    cmaps = {
        'standard': ('white', '#f4d4d4', '#85c0f9'),
        'protanopia': ('white', '#ae9c45', '#a7b8f8'),
        'deuteranopia': ('white', '#c59434', '#a3b7f9')
    }

    sns.heatmap(observables_df, annot=False,
                cmap=ListedColormap(cmaps[palette]), cbar=False)
    plt.title("matrix with observables selection")
    plt.show()
    return observables_df


def observables_selection_checker(observables_dict):
    """
    Observables selection checker

    Checks and notifies the user if a selection is
    not correct

    Parameters
    ----------
    observables_dict: dict
        dictionary with the observables selection
    """
    if (observables_dict['WL']['GCspectro'] or
            observables_dict['GCphot']['GCspectro']):
        observables_dict['WL']['GCspectro'] = False
        observables_dict['GCphot']['GCspectro'] = False
        warnings.warn(
            'Attention: CLOE only computes cross-correlations '
            'for the photometric survey!')
        warnings.warn(
            "Entries ['WL']['GCphot'] and ['GCphot']['GCspec'] "
            "are changed to False.")
    return observables_dict
