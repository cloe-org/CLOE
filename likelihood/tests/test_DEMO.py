"""UNIT TESTS FOR DEMO

This module contains unit tests for the DEMO jupyter notebook
=======

"""

from unittest import TestCase
import numpy.testing as npt
import subprocess
import os
import shlex


class DEMOTestCase(TestCase):

    def setUp(self):
        # set path to notebook
        self.notebook_path = os.getcwd() + '/notebooks/'
        self.notebook_name = 'DEMO'
        # if DEMO runs with no error, os returns 0
        self.test = 0

    def tearDown(self):
        self.notebook_path = None
        self.notebook_name = None
        self.test = None

    def test_DEMO_convert(self):
        # Create .py script from the ipynb notebook
        test_convert = subprocess.call(
            shlex.split(
                'jupyter nbconvert --to script {}'.format(
                    self.notebook_path +
                    self.notebook_name +
                    '.ipynb')))
        npt.assert_equal(
            test_convert,
            self.test,
            err_msg='DEMO notebook could not be converted to python script')

    def test_DEMO_execute(self):
        # Execute the DEMO script
        test_execute = subprocess.call(
            shlex.split(
                'ipython {}'.format(
                    self.notebook_path +
                    self.notebook_name +
                    '.py')))
        npt.assert_equal(test_execute, self.test,
                         err_msg='Error in DEMO! Open it up and CHECK')
