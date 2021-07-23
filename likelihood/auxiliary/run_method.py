"""RUN METHOD

Contains routines to check how the code is currently being run
"""

import __main__ as main


def run_is_interactive():
    r"""Checks if the code runs from a script or interactively

    The __file__ attribute of the __main__ object is present only when the
    code runs from a terminal using a script file.

    Returns
    -------
    bool
        Returns True if the code runs with an interactive interface
        (e.g. Python shell, jupyter notebook, etc.), False otherwise
    """
    return (not(hasattr(main, '__file__')))
