"""UNIT TESTS FOR PARAMETERS CONVERTER

This module contains unit tests
for the params_converter module.
=======

"""

import numpy as np
from unittest import TestCase
import cloe.auxiliary.params_converter as pc


class ParamsConverterTestCase(TestCase):

    def setUp(self):
        self.params_camb = {key: None for key in pc.camb_to_classy.keys()}
        self.params_classy = {key: None for key in pc.camb_to_classy.values()}
        self.theory = {'extra_args': {}}

    def tearDown(self):
        self.params_camb = None
        self.params_classy = None

    def test_camb_to_classy_conversion(self):
        pc.convert_params(self.params_camb, self.theory, 'classy')
        self.params_camb.pop('Omega_Lambda')
        if not np.all([key in self.params_classy.keys()
                       for key in self.params_camb.keys()]):
            raise KeyError('Unrecognised key while converting CAMB parameters '
                           'to CLASS parameters.')

    def test_classy_to_camb_conversion(self):
        pc.convert_params(self.params_classy, self.theory, 'camb')
        if not np.all([key in self.params_camb.keys()
                       for key in self.params_classy.keys()]):
            raise KeyError('Unrecognised key while converting CLASS '
                           'parameters to CAMB parameters.')
