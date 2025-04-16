"""Test cases for math utilities"""

import unittest

import numpy as np

from pyastrobee.utils.math_utils import normalize, is_diagonal


class MathTest(unittest.TestCase):
    """Contains test cases to validate math utility functions"""

    def test_normalize(self):
        a = np.random.rand(3)
        a = normalize(a)
        np.testing.assert_almost_equal(1.0, np.linalg.norm(a))
        a = np.random.rand(1)
        a = normalize(a)
        np.testing.assert_almost_equal(1.0, np.linalg.norm(a))
        with self.assertRaises(ZeroDivisionError):
            a = [0, 0, 0]
            normalize(a)

    def test_is_diagonal(self):
        a = np.diag([1, 2, 3])
        self.assertTrue(is_diagonal(a))
        a = np.random.rand(2, 2)
        self.assertFalse(is_diagonal(a))
        a = np.zeros((3, 3))
        self.assertTrue(is_diagonal(a))
        # Raises an error if it is a scalar
        with self.assertRaises(ValueError):
            self.assertTrue(is_diagonal(1))
        # But, if it is a 1x1 matrix, it should work
        a = np.array([1])
        self.assertTrue(is_diagonal(a))
        a = np.array([[1]])
        self.assertTrue(is_diagonal(a))


if __name__ == "__main__":
    unittest.main()
