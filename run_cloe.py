#!/usr/bin/env python
"""Top level script for running CLOE.

This is the top level script for running the CLOE user interface.
"""
import argparse
import json
import sys
from warnings import warn
from likelihood.user_interface.likelihood_ui import LikelihoodUI
from likelihood.auxiliary.logger import open_logger, close_logger, \
    catch_error, set_logging_level


def run_script(log):
    """Method to run the script

    Parses command line arguments, opens a logger,
    instantiates a likelihood_ui object,
    and invokes the method based on the action parameter.

    Parameters
    ----------
    log: logging.Logger
       Reference for logging.Logger, specifying the logger of the run.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        nargs='?', default=None,
                        help='specify input yaml config file')
    parser.add_argument('-a', '--action',
                        type=str,
                        default='run',
                        help='specify action to be performed')
    parser.add_argument('-ps', '--plot-settings',
                        dest='settings',
                        type=str,
                        default=None,
                        help='specify settings for plotting routines')
    parser.add_argument('-d', '--dict',
                        type=str,
                        default='{}',
                        help='specify additional arguments')
    parser.add_argument('-v', '--verbose',
                        type=str,
                        default='info',
                        help='specify logging verbosity level '
                             '{debug, info, warning, error, critical}')
    args = parser.parse_args()

    log = open_logger('CLOE')
    log.info('Starting CLOE')
    set_logging_level(log, args.verbose)

    user_dict = json.loads(args.dict)

    log.info('Instantiating user interface')
    ui = LikelihoodUI(user_config_file=args.config, user_dict=user_dict)
    if args.action == 'run':
        log.info('Selected RUN mode')
        ui.run()
    elif args.action == 'process':
        log.info('Selected PROCESS mode')
        ui.process_chain()
    elif args.action == 'plot':
        log.info('Selected PLOT mode')
        ui.plot(args.settings)
    else:
        warn('Specified action not supported.')

    log.info('Exiting CLOE')
    close_logger(log)


def main():
    """Main function.

    Calls run_script
    """

    log = None
    try:
        run_script(log)
        return 0
    except Exception as err:
        catch_error(err, log)
        return 1


if __name__ == "__main__":
    sys.exit(main())
