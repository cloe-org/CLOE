"""PARAMETERS CONVERTER

Functions to switch between the naming convention of CAMB/CLASS
"""

camb_to_classy = {
    'tau': 'tau_reio',
    'ombh2': 'omega_b',
    'omch2': 'omega_cdm',
    'num_nu_massless': 'N_ur',
    'num_nu_massive': 'N_ncdm',
    'mnu': 'm_ncdm',
    'omnuh2': 'omega_ncdm',
    'omk': 'Omega_k',
    'w': 'w0_fld',
    'wa': 'wa_fld',
    'As': 'A_s',
    'ns': 'n_s'}


def convert_params(params, theory, mode):
    r"""Main function to convert cosmological parameters

    Checks the parameters and theory section of an info dictionary and modify
    parameter names according to the choice of Boltzmann solver.

    Parameters
    ----------
    params: dict
        Parameters section of the info dictionary.
    theory: dict
        Theory section of the info dictionary.
    mode: str
        Selected Boltzmann solver. Can be either `camb` or `classy`.

    Raise
    -----
    ValueError
        If mode is neither camb or classy.
    """
    if mode == 'classy':
        classy_set_params(params, theory)
    elif mode == 'camb':
        camb_set_params(params, theory)
    else:
        raise ValueError('Only available Boltzmann solvers are '
                         'CAMB and CLASS')


def classy_set_params(params, theory):
    r"""CLASS parameter converter

    Resets the parameters dictionary in a CLASS-like format.

    Paramters
    ---------
    params: dict
        Parameters section of the info dictionary.
    theory: dict
        Theory section of the info dictionary.
    """
    for camb_param, classy_param in camb_to_classy.items():
        if camb_param in params.keys():
            params[classy_param] = params.pop(camb_param)
    if 'nnu' in params.keys():
        nrad = 4.41e-3
        params['N_ur'] = ((params.pop('nnu') - nrad) *
                          (3.0 - params['N_ncdm']) / 3.0 + nrad)
    if 'use_ppf' in params.keys():
        use_ppf = params.pop('use_ppf')
        theory['extra_args']['use_ppf'] = 'yes' if use_ppf else 'no'
    if 'Omega_Lambda' not in params.keys():
        params['Omega_Lambda'] = 0.0
    if 'omegab' in params.keys():
        if isinstance(params['omegab'], dict):
            if 'derived' in params['omegab']:
                params['omegab']['derived'] = \
                    params['omegab']['derived'].replace('ombh2', 'omega_b')


def camb_set_params(params, theory):
    r"""CAMB parameter converter

    Resets the parameters dictionary in a CAMB-like format.

    Parameters
    ----------
    params: dict
        Parameters section of the info dictionary.
    theory: dict
        Theory section of the info dictionary.
    """
    classy_to_camb = dict(zip(camb_to_classy.values(), camb_to_classy.keys()))
    for classy_param, camb_param in classy_to_camb.items():
        if classy_param in params.keys():
            params[camb_param] = params.pop(classy_param)
    if 'num_nu_massless' in params.keys():
        theory['extra_args']['num_nu_massless'] = params.pop('num_nu_massless')
    if 'num_nu_massive' in params.keys():
        theory['extra_args']['num_nu_massive'] = params.pop('num_nu_massive')
    if 'use_ppf' in params.keys():
        use_ppf = params.pop('use_ppf')
        theory['extra_args']['dark_energy_model'] = (
            'ppf' if use_ppf else 'fluid')
    if 'omegab' in params.keys():
        if isinstance(params['omegab'], dict):
            if 'derived' in params['omegab']:
                params['omegab']['derived'] = \
                    params['omegab']['derived'].replace('omega_b', 'ombh2')
