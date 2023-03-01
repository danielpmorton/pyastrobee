"""Test cases for quaternions

TODO
- Quaternion distance
- Quaternion interpolation
"""

import unittest
import numpy as np

from pyastrobee.utils import rotations as rts
from pyastrobee.utils.math_utils import normalize
from pyastrobee.utils.quaternion import Quaternion


class QuaternionTest(unittest.TestCase):
    """Contains test cases for the Quaternion class and all quaternion-associated functions"""

    def test_quaternion_class(self):
        q0 = normalize([0.1, 0.2, 0.3, 0.4])
        q = Quaternion(wxyz=q0)
        np.testing.assert_equal(q0, q.wxyz)
        np.testing.assert_equal([*q0[1:], q0[0]], q.xyzw)

    def test_quaternion_conversions(self):
        q = normalize([0.1, 0.2, 0.3, 0.4])
        axis, angle = rts.quat_to_axis_angle(q)
        np.testing.assert_array_almost_equal(rts.axis_angle_to_quat(axis, angle), q)
        rmat = rts.quat_to_rmat(q)
        np.testing.assert_array_almost_equal(rts.rmat_to_quat(rmat), q)
        euler_xyz = rts.quat_to_euler_xyz(q)
        np.testing.assert_array_almost_equal(rts.euler_xyz_to_quat(euler_xyz), q)
        fixed_xyz = rts.quat_to_fixed_xyz(q)
        np.testing.assert_array_almost_equal(rts.fixed_xyz_to_quat(fixed_xyz), q)

    def test_quaternion_combination(self):
        euler_1 = np.array([0, 0, 0])
        euler_2 = np.array([0.1, 0.2, 0.3])
        q1 = rts.euler_xyz_to_quat(euler_1)
        q2 = rts.euler_xyz_to_quat(euler_2)
        q3 = rts.combine_quaternions(q1, q2)
        euler_3 = rts.quat_to_euler_xyz(q3)
        np.testing.assert_array_almost_equal(euler_3, euler_1 + euler_2)

    def test_quaternion_interpolation(self):
        # TODO. Need to think of a good way to evaluate this test case
        pass

    def test_quaternion_distance(self):
        # TODO. Need to think of a good way to evaluate this test case
        pass


if __name__ == "__main__":
    unittest.main()
