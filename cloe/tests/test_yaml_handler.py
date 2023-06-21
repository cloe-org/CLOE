"""TEST YAML HANDLER

This module contains functions for handling the :obj:`yaml`
files (loading, writing..).
"""


from unittest import TestCase
from unittest.mock import patch
from cloe.auxiliary.yaml_handler import yaml_write, yaml_read
from cloe.auxiliary.yaml_handler import yaml_read_and_check_dict


class yaml_handler_test(TestCase):

    def setUp(self):
        self.file_name = '/dev/null'
        self.config = {'name': 'anything', 'type': 'whatever'}
        self.bad_input_type = 'a string'
        self.overwrite = True

    def tearDown(self):
        self.fine_name = None
        self.config = None
        self.bad_input_type = None

    @patch('yaml.dump')
    def test_write(self, dump_mock):
        dump_mock.return_value = 'whatever'
        yaml_write(self.file_name, self.config, self.overwrite)
        self.assertEqual(dump_mock.call_count, 1)

    def test_write_bad_input_type(self):
        self.assertRaises(
            TypeError,
            yaml_write,
            self.file_name,
            self.bad_input_type,
            self.overwrite
        )

    def test_write_overwrite_exception(self):
        self.assertRaises(
            RuntimeError,
            yaml_write,
            self.file_name,
            self.config
        )

    @patch('yaml.load')
    def test_read(self, load_mock):
        load_mock.return_value = "whatever"
        yaml_read(self.file_name)
        self.assertEqual(load_mock.call_count, 1)

    @patch('yaml.load')
    def test_read_and_check_dict(self, load_mock):
        load_mock.return_value = self.config
        yaml_read_and_check_dict(self.file_name, ['name', 'type'])
        self.assertEqual(load_mock.call_count, 1)

    @patch('yaml.load')
    def test_read_and_check_dict_bad_type(self, load_mock):
        load_mock.return_value = self.bad_input_type
        self.assertRaises(
            TypeError,
            yaml_read_and_check_dict,
            self.file_name,
            None
        )

    @patch('yaml.load')
    def test_read_and_check_dict_bad_dict(self, load_mock):
        load_mock.return_value = self.config
        self.assertRaises(
            KeyError,
            yaml_read_and_check_dict,
            self.file_name,
            ['missing key']
        )
