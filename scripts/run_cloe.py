#!/usr/bin/env python
"""Top level script for running CLOE.

This is the top level script for running the CLOE user interface.
"""
import argparse
from likelihood.user_interface.likelihood_ui import LikelihoodUI
from os import sys


def main():
    """Main function.

    Parse command line arguments, instantiate a likelihood_ui object and invoke
    the run() method.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help='specify input yaml config file',
                        type=str)

    args = parser.parse_args()
    ui = LikelihoodUI(user_config_file=args.config)
    ui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
