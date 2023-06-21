"""RUN METHOD

Contains routines to check if the code is run with an interactive
interface (e.g. Python shell, jupyter notebook, etc.) or not.
"""

import __main__ as main


def run_is_interactive():
    r"""Checks if the code runs from a script or interactively.

    The :obj:`__file__` attribute of the :obj:`__main__` object is present
    only when the code runs from a terminal using a script file.

    Returns
    -------
    bool
        Returns :obj:`True` if the code runs with an interactive interface
        (e.g. Python shell, jupyter notebook, etc.), :obj:`False` otherwise
    """
    return not (hasattr(main, '__file__'))
