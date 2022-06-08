from unittest import TestCase
from unittest.mock import patch
from cloe.auxiliary.likelihood_yaml_handler import *


class yaml_handler_test(TestCase):

    def setUp(self):
        self.file_name = '/dev/null'
        self.model_dict = \
            {'user_models': {
                'cosmology': self.file_name,
                'foo': self.file_name
            }, 'user_options': {
                'overwrite': False}
            }
        self.model_dict_overwrite = \
            {'user_models': {
                'cosmology': self.file_name,
                'foo': self.file_name
            }, 'user_options': {
                'overwrite': True}
            }
        self.cobaya_dict = {'force': None, 'likelihood': None,
                            'output': None,
                            'params': {'ns': 1, 'p0': 0},
                            'sampler': None, 'theory': None}

        self.deep_dict = {'ns': 1., 'p_deep': {'p1': 1., 'p2': 2.}}

    def tearDown(self):
        self.file_name = None
        self.model_dict = None
        self.cobaya_dict = None
        self.deep_dict = None

    @patch('yaml.load')
    def test_load_model_dict_from_yaml(self, load_mock):
        load_mock.return_value = self.model_dict
        load_model_dict_from_yaml(self.file_name)
        self.assertEqual(load_mock.call_count, 1)

    def test_get_params_dict_without_cosmo_params(self):
        original_dict = deepcopy(self.deep_dict)
        new_dict = get_params_dict_without_cosmo_params(self.deep_dict)
        # check that get_cobaya_dict_without_cosmo_params
        # does not modify the input dictionary
        self.assertDictEqual(original_dict, self.deep_dict)
        # check that the cosmological parameter ns is not in the new_dict
        expected_dict = {'p_deep': {'p1': 1., 'p2': 2.}}
        self.assertDictEqual(new_dict, expected_dict)

    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    def test_generate_params_dict_from_model_dict(self, read_mock):
        read_mock.return_value = self.deep_dict
        generate_params_dict_from_model_dict(self.model_dict)
        self.assertEqual(read_mock.call_count, 2)

    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    def test_generate_params_dict_from_model_dict_nocosmo(self, read_mock):
        read_mock.return_value = {}
        generate_params_dict_from_model_dict(self.model_dict, False)
        # check that yaml_read has been called only one
        # (for the only non-cosmology file in self.model_dict)
        self.assertEqual(read_mock.call_count, 1)

    @patch('cloe.auxiliary.yaml_handler.yaml_write')
    @patch('cloe.auxiliary.yaml_handler.yaml_read')
    def test_write_params_yaml_from_model_dict(self, r_mock, w_mock):
        r_mock.return_value = self.deep_dict
        write_params_yaml_from_model_dict(self.model_dict)
        # check that yaml_read has been called only one
        # (for the only non-cosmology file in self.model_dict)
        self.assertEqual(r_mock.call_count, 1)
        self.assertEqual(w_mock.call_count, 1)

    @patch('cloe.auxiliary.likelihood_yaml_handler.'
           'load_model_dict_from_yaml')
    @patch('cloe.auxiliary.likelihood_yaml_handler.'
           'generate_params_dict_from_model_dict')
    def test_update_cobaya_params_from_model_yaml(self, g_mock, l_mock):
        l_mock.return_value = self.model_dict
        g_mock.return_value = self.deep_dict
        update_cobaya_params_from_model_yaml(self.cobaya_dict, self.file_name)
        params_dict = self.cobaya_dict['params']
        self.assertIsNotNone(params_dict)
        self.assertDictEqual(params_dict, self.deep_dict)
        self.assertEqual(l_mock.call_count, 1)

    @patch('cloe.auxiliary.likelihood_yaml_handler.'
           'load_model_dict_from_yaml')
    @patch('cloe.auxiliary.likelihood_yaml_handler.'
           'generate_params_dict_from_model_dict')
    @patch('cloe.auxiliary.yaml_handler.yaml_write')
    def test_update_cobaya_params_from_model_yaml_overwrite(self, w_mock,
                                                            g_mock, l_mock):
        l_mock.return_value = self.model_dict_overwrite
        g_mock.return_value = self.deep_dict
        update_cobaya_params_from_model_yaml(self.cobaya_dict, self.file_name)
        params_dict = self.cobaya_dict['params']
        self.assertIsNotNone(params_dict)
        self.assertDictEqual(params_dict, self.deep_dict)
        self.assertEqual(l_mock.call_count, 1)
        self.assertEqual(w_mock.call_count, 1)
