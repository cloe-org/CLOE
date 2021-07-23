#!/usr/bin/env python
"""Top level script for running CLOE.

This is the top level script for running the CLOE user interface.
"""
import argparse
import json
from likelihood.user_interface.likelihood_ui import LikelihoodUI
from os import sys
from warnings import warn


def main():
    """Main function.

    Parse command line arguments, instantiate a likelihood_ui object and invoke
    the run() method.
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
    args = parser.parse_args()

    dict=json.loads(args.dict)

    ui = LikelihoodUI(user_config_file=args.config, user_dict=dict)
    if (args.action=='run'):
        ui.run()
    elif (args.action=='plot'):
        ui.plot(args.settings)
    else:
        warn('Specified action not supported.')

    return 0


if __name__ == "__main__":
    sys.exit(main())
