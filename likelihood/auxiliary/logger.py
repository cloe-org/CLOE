"""Logger

This module contains routines to handle loggers
"""

import sys
import logging


def open_logger(filename):
    """Open an instance of logging.Logger

    Parameters
    ----------
    filename : str
        Log file name

    Returns
    -------
    logging.Logger
        Logging instance
    """
    logging.captureWarnings(True)

    formatter = logging.Formatter(fmt='%(asctime)s [%(name)s] %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S',)

    fh = logging.FileHandler(filename='{0}.log'.format(filename), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    log = logging.getLogger(filename)
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)

    warnings_log = logging.getLogger('py.warnings')
    warnings_log.addHandler(fh)

    return log


def close_logger(log):
    """Close an instance of logging.Logger

    Parameters
    ----------
    log : logging.Logger
        Logging instance
    """
    for log_handler in log.handlers:
        log.removeHandler(log_handler)


def catch_error(exception, log=None):
    """Catch an exception and output the error message.

    If a logger is provided, the error is also logged.

    Parameters
    ----------
    exception : error or str
        Exception message string
    log : logging.Logger, optional
        Logging structure instance (default is ``None``)
    """
    err_txt = 'ERROR'

    stream_txt = '{0}: {1}\n'.format(err_txt, exception)
    sys.stderr.write(stream_txt)

    if not isinstance(log, type(None)):
        log_txt = 'ERROR: {0}\n'.format(exception)
        log.exception(log_txt)
