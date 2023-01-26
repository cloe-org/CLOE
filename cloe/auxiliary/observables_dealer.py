"""observables_dealer

Contains function to read the observable dictionary
and plot the visualization of the matrix
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    observables_df = pd.DataFrame(
        observables_dict).fillna(-1).astype(int).T

    cmaps = {
        'standard': ('white', '#f4d4d4', '#85c0f9'),
        'protanopia': ('white', '#ae9c45', '#a7b8f8'),
        'deuteranopia': ('white', '#c59434', '#a3b7f9')
    }
    cmap = LinearSegmentedColormap.from_list(palette,
                                             cmaps[palette],
                                             len(cmaps[palette]))
    ax = sns.heatmap(observables_df, annot=False,
                     cmap=cmap,
                     cbar=True)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.667, 0, 0.667])
    colorbar.set_ticklabels(['No input', 'False', 'True'])
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
            "Entries ['WL']['GCspec'] and ['GCphot']['GCspec'] "
            "are changed to False.")
    return observables_dict


def observables_selection_specifications_checker(observables_dict,
                                                 specifications_dict):
    """
    Observables selection and specifications checker

    Merges in a single dictionary the observable selection
    and the specifications loaded in the cobaya_interface.py

    Parameters
    ----------
    observables_dict: dict
        dictionary with the observables selection
    specifications_dict: dict
        dictionary with the observables specifications
    """
    # First check observables selection
    checked_observables_dict = observables_selection_checker(observables_dict)
    # Define empty dict for merging
    merged_dict = {'selection': {}, 'specifications': {}}
    merged_dict['selection'] = checked_observables_dict
    # Check each entry of the observables selection and
    # add specifications
    if checked_observables_dict['WL']['WL']:
        merged_dict['specifications']['WL'] = specifications_dict['WL']
    if checked_observables_dict['GCphot']['GCphot']:
        merged_dict['specifications']['GCphot'] = \
            specifications_dict['GCphot']
    if checked_observables_dict['GCspectro']['GCspectro']:
        merged_dict['specifications']['GCspectro'] = \
            specifications_dict['GCspectro']
    if checked_observables_dict['WL']['GCphot']:
        merged_dict['specifications']['WL-GCphot'] = \
            specifications_dict['WL-GCphot']
    # At the moment, these quantities below are not computed
    # by CLOE and we are forcing this selection to be False.
    # Therefore, the specifications are not loaded.
    # They are added here for completeness in the future
    if checked_observables_dict['GCphot']['GCspectro']:
        merged_dict['specifications']['GCphot-GCspectro'] = \
            specifications_dict['GCphot-GCspectro']
    if checked_observables_dict['WL']['GCspectro']:
        merged_dict['specifications']['WL-GCspectro'] = \
            specifications_dict['WL-GCspectro']
    return merged_dict
