"""LOGGER

This module contains routines to handle loggers.
"""

import sys
import logging
import pprint
from datetime import datetime


def open_logger(name='CLOE'):
    """Opens an instance of :obj:`logging.Logger`.

    Parameters
    ----------
    name : str
        Log name. The filename is the concatenation of the name
        and the timestamp

    Returns
    -------
    logging.Logger
        Logging instance
    """
    logging.captureWarnings(True)

    formatter = logging.Formatter(fmt='%(asctime)s [%(name)s] '
                                      '%(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S',)

    date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    filename = '{0}-{1}.log'.format(name, date_time)
    fh = logging.FileHandler(filename=filename, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(fh)

    warnings_log = logging.getLogger('py.warnings')
    warnings_log.addHandler(fh)

    return log


def set_logging_level(log, level):
    """
    Sets the logging level.

    Parameters
    ----------
    log : logging.Logger
        Logging instance
    level: str
        Logging level {debug, info, warning, error, critical}
    """
    level = level.lower()
    if level == "debug":
        log.setLevel(logging.DEBUG)
    elif level == "info":
        log.setLevel(logging.INFO)
    elif level == "warning":
        log.setLevel(logging.WARNING)
    elif level == "error":
        log.setLevel(logging.ERROR)
    elif level == "critical":
        log.setLevel(logging.CRITICAL)
    else:
        log.setLevel(logging.INFO)
        log.warning("Unknown logging verbose level: {}".format(level))

    log.info("Logging verbose level: {}".format(
        logging.getLevelName(log.getEffectiveLevel())))


def log_debug(message):
    """
    Logs a message with verbose level debug.

    Parameters
    ----------
    message: str
        Message to be logged
    """
    log = logging.getLogger('CLOE')
    log.debug(message)


def log_info(message):
    """
    Logs a message with verbose level info.

    Parameters
    ----------
    message: str or dict
        Message to be logged
    """
    log = logging.getLogger('CLOE')
    if type(message) is dict:
        message = '\n' + pprint.pformat(message)
    log.info(message)


def log_warning(message):
    """
    Logs a message with verbose level warning.

    Parameters
    ----------
    message: str
        Message to be logged
    """
    log = logging.getLogger('CLOE')
    log.warning(message)


def log_error(message):
    """
    Logs a message with verbose level error.

    Parameters
    ----------
    message: str
        Message to be logged
    """
    log = logging.getLogger('CLOE')
    log.error(message)


def log_critical(message):
    """
    Logs a message with verbose level critical.

    Parameters
    ----------
    message: str
        Message to be logged
    """
    log = logging.getLogger('CLOE')
    log.critical(message)


def close_logger(log):
    """Closes an instance of :obj:`logging.Logger`.

    Parameters
    ----------
    log : logging.Logger
        Logging instance
    """
    for log_handler in log.handlers:
        log.removeHandler(log_handler)


def catch_error(exception, log=None):
    """Catches an exception and output the error message.

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
