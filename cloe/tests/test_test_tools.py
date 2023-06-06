"""UNIT TESTS FOR TEST TOOLS

This module contains unit tests for :obj:`test_tools`.

"""

from unittest import TestCase
import os
import pickle
import numpy as np
import numpy.testing as npt
from cloe.tests.test_tools import test_data_handler as tdh
from cloe.tests.test_tools import mock_objects as mo


class TestDataHandlerTestCase(TestCase):

    def setUp(self) -> None:
        # Check values
        self.datfile = 'test.dat'
        self.npyfile = 'test.npy'
        self.picklefile = 'test.pickle'
        self.dat = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.npy = np.arange(5)
        self.pickle = {'a': 1, 'b': 2, 'c': 3}
        self.dict = {'d': 4, 'e': 5, 'f': 6}

    def tearDown(self):
        self.datfile = None
        self.npyfile = None
        self.picklefile = None
        self.dat = None
        self.npy = None
        self.pickle = None
        self.dict = None

    def test_load_test_npy(self):
        npt.assert_equal(
            tdh.load_test_npy(self.datfile),
            self.dat,
            err_msg='Numpy text file not loaded correctly.',
        )
        npt.assert_equal(
            tdh.load_test_npy(self.npyfile),
            self.npy,
            err_msg='Numpy binary file not loaded correctly.',
        )
        npt.assert_raises(
            ValueError,
            tdh.load_test_npy,
            self.picklefile,
        )

    def test_load_test_pickle(self):
        npt.assert_equal(
            tdh.load_test_pickle(self.picklefile),
            self.pickle,
            err_msg='Pickle file not loaded correctly.',
        )

    def test_save_test_pickle(self):
        file_name = 'temp.pickle'
        try:
            tdh.save_test_pickle(file_name, self.dict, path='.')
            with open(file_name, 'rb') as pickle_file:
                content = pickle.load(pickle_file)
        finally:
            os.remove(file_name)
        npt.assert_equal(
            content,
            self.dict,
            err_msg='Pickle file not saved correctly.',
        )
