from unittest import TestCase
from unittest.mock import patch
import likelihood.auxiliary.matrix_manipulator as matrix_manipulator
import numpy


class MatrixManipulator_test(TestCase):

    def setUp(self):
        dimMin = 10
        dimMax = 50
        self.nRows1 = numpy.random.randint(dimMin, dimMax)
        self.nCols1 = numpy.random.randint(dimMin, dimMax)
        self.nRows2 = numpy.random.randint(dimMin, dimMax)
        self.nCols2 = numpy.random.randint(dimMin, dimMax)
        self.mat1 = numpy.zeros((self.nRows1, self.nCols1))
        self.mat2 = numpy.zeros((self.nRows2, self.nCols2))
        for row in range(0, self.nRows1):
            for col in range(0, self.nCols1):
                self.mat1[row, col] = numpy.random.randn()
        for row in range(0, self.nRows2):
            for col in range(0, self.nCols2):
                self.mat2[row, col] = numpy.random.randn()

    def tearDown(self):
        pass

    # test that the size of the output matrix is as expected, and that
    # 100 random entries match the expectation
    def test_merge_matrices(self):
        outMatrix = matrix_manipulator.merge_matrices(self.mat1, self.mat2)
        expectedRows = self.nRows1 + self.nRows2
        expectedCols = self.nCols1 + self.nCols2
        self.assertEqual(outMatrix.shape[0], expectedRows,
                         f'Unexpected number of rows in output matrix')
        self.assertEqual(outMatrix.shape[1], expectedCols,
                         f'Unexpected number of columns in output matrix')
        for iter in range(0, 100):
            rndRow = numpy.random.randint(0, outMatrix.shape[0])
            rndCol = numpy.random.randint(0, outMatrix.shape[1])
            actualValue = outMatrix[rndRow, rndCol]
            if (rndRow < self.nRows1 and rndCol < self.nCols1):
                expected = self.mat1[rndRow, rndCol]
            elif (rndRow >= self.nRows1 and rndCol >= self.nCols1):
                expected = (
                    self.mat2[rndRow - self.nRows1, rndCol - self.nCols1])
            else:
                expected = 0
            self.assertEqual(expected, actualValue,
                             f'Unexpected element in output matrix at'
                             f' position ({rndRow},{rndCol})')
