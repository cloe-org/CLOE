from unittest import TestCase
from unittest.mock import patch
from likelihood.auxiliary.yaml_handler import yaml_write, yaml_read
import os


class yaml_handler_test(TestCase):

    def setUp(self):
        self.file_name = '/dev/null'
        self.config = {'name': 'anything', 'type': 'whatever'}
        self.bad_input_type = 'a string'

    def tearDown(self):
        self.fine_name = None
        self.config = None
        self.bad_input_type = None

    @patch('yaml.dump')
    def test_write(self, dump_mock):
        dump_mock.return_value = 'whatever'
        yaml_write(self.file_name, self.config)
        self.assertEqual(dump_mock.call_count, 1)

    def test_write_bad_input_type(self):
        self.assertRaises(
            TypeError,
            yaml_write,
            self.file_name,
            self.bad_input_type
        )

    @patch('yaml.load')
    def test_read(self, load_mock):
        load_mock.return_value = "whatever"
        yaml_read(self.file_name)
        self.assertEqual(load_mock.call_count, 1)
