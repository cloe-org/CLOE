from unittest import TestCase
from unittest.mock import patch
from likelihood.auxiliary.likelihood_yaml_generator import *


class yaml_handler_test(TestCase):

    def setUp(self):
        self.file_name = '/dev/null'
        self.model_dict = \
            {'user_models': {
                'cosmology': self.file_name,
                'foo': self.file_name
            }, 'user_options': {
                'model_path': '/',
                'in_cobaya_shell_path': self.file_name,
                'out_params_filename': self.file_name,
                'overwrite': False}
            }
        self.cobaya_shell_dict = {'force': None, 'likelihood': None,
                                  'output': None, 'params': None,
                                  'sampler': None, 'theory': None}

        self.deep_dict = {'p0': 0., 'p_deep': {'p1': 1., 'p2': 2.}}

    def tearDown(self):
        self.file_name = None
        self.model_dict = None
        self.cobaya_shell_dict = None
        self.deep_dict = None

    @patch('yaml.load')
    def test_load_model_dict_from_yaml(self, load_mock):
        load_mock.return_value = self.model_dict
        load_model_dict_from_yaml(self.file_name)
        self.assertEqual(load_mock.call_count, 1)

    @patch('yaml.load')
    def test_load_cobaya_dict_from_yaml(self, load_mock):
        load_mock.return_value = self.cobaya_shell_dict
        load_cobaya_dict_from_yaml(self.file_name)
        self.assertEqual(load_mock.call_count, 1)

    @patch('likelihood.auxiliary.yaml_handler.yaml_read')
    def test_generate_params_dict_from_model_dict(self, read_mock):
        read_mock.return_value = self.deep_dict
        generate_params_dict_from_model_dict(self.model_dict)
        self.assertEqual(read_mock.call_count, 2)

    @patch('likelihood.auxiliary.yaml_handler.yaml_read')
    def test_generate_params_dict_from_model_dict_nocosmo(self, read_mock):
        read_mock.return_value = {}
        generate_params_dict_from_model_dict(self.model_dict, False)
        # check that yaml_read has been called only one
        # (for the only non-cosmology file in self.model_dict)
        self.assertEqual(read_mock.call_count, 1)

    @patch('likelihood.auxiliary.yaml_handler.yaml_read')
    @patch('likelihood.auxiliary.yaml_handler.yaml_read_and_check_dict')
    def test_generate_params_dict_from_model_yaml(self, rc_mock, r_mock):
        rc_mock.return_value = self.model_dict
        r_mock.return_value = {}
        generate_params_dict_from_model_yaml(self.file_name)
        self.assertEqual(rc_mock.call_count, 1)
        self.assertEqual(r_mock.call_count, 2)

    @patch('likelihood.auxiliary.yaml_handler.yaml_write')
    @patch('likelihood.auxiliary.yaml_handler.yaml_read')
    def test_write_params_yaml_from_model_dict(self, r_mock, w_mock):
        r_mock.return_value = self.deep_dict
        write_params_yaml_from_model_dict(self.model_dict)
        # check that yaml_read has been called only one
        # (for the only non-cosmology file in self.model_dict)
        self.assertEqual(r_mock.call_count, 1)
        self.assertEqual(w_mock.call_count, 1)

    @patch('likelihood.auxiliary.yaml_handler.yaml_write')
    @patch('likelihood.auxiliary.yaml_handler.yaml_read')
    @patch('likelihood.auxiliary.yaml_handler.yaml_read_and_check_dict')
    def test_write_params_yaml_from_model_yaml(self, rc_mock,
                                               r_mock, w_mock):
        rc_mock.return_value = self.model_dict
        r_mock.return_value = self.deep_dict
        write_params_yaml_from_model_yaml(self.file_name)
        self.assertEqual(rc_mock.call_count, 1)
        self.assertEqual(r_mock.call_count, 1)
        self.assertEqual(w_mock.call_count, 1)

    def test_update_cobaya_dict_with_params_dict(self):
        update_cobaya_dict_with_params_dict(self.cobaya_shell_dict,
                                            self.deep_dict)
        params_dict = self.cobaya_shell_dict['params']
        self.assertIsNotNone(params_dict)
        self.assertDictEqual(params_dict, self.deep_dict)

    @patch('likelihood.auxiliary.yaml_handler.yaml_read')
    @patch('likelihood.auxiliary.likelihood_yaml_generator.'
           'load_cobaya_dict_from_yaml')
    @patch('likelihood.auxiliary.likelihood_yaml_generator.'
           'load_model_dict_from_yaml')
    def test_generate_cobaya_dict_from_model_yaml(self, lm_mock,
                                                  lc_mock, yr_mock):
        lm_mock.return_value = self.model_dict
        lc_mock.return_value = self.cobaya_shell_dict
        yr_mock.return_value = self.deep_dict
        cobaya_dict = generate_cobaya_dict_from_model_yaml(self.file_name)
        params_dict = cobaya_dict['params']
        self.assertIsNotNone(params_dict)
        self.assertDictEqual(params_dict, self.deep_dict)
        self.assertEqual(lm_mock.call_count, 1)
        self.assertEqual(lc_mock.call_count, 1)
        self.assertEqual(yr_mock.call_count, 2)
