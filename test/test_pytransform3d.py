"""Test cases for pytransform3d validation

(confirming that their conventions and input parameters match what we expect)
"""

import unittest
import numpy as np
import pytransform3d.rotations as rt
import pytransform3d.transformations as tr

from pyastrobee.archive import my_rotations as my_rt
from pyastrobee.archive import my_transformations as my_tr


class Pytransform3dTest(unittest.TestCase):
    """Contains test cases to compare the pytransform3d functions against the known math"""

    def test_rotation_equivalents(self):
        np.testing.assert_array_almost_equal(
            my_rt.euler_xyz_to_rmat(0.1, 0.2, 0.3),
            rt.matrix_from_euler([0.1, 0.2, 0.3], 0, 1, 2, False),
        )
        R = my_rt.euler_xyz_to_rmat(0.1, 0.2, 0.3)
        np.testing.assert_array_almost_equal(
            my_rt.rmat_to_euler_xyz(R), rt.euler_from_matrix(R, 0, 1, 2, False)
        )
        axis = np.array([0.1, 0.2, 0.3]) / np.linalg.norm([0.1, 0.2, 0.3])
        angle = 0.1
        np.testing.assert_array_almost_equal(
            my_rt.axis_angle_to_rmat(axis, angle),
            rt.matrix_from_axis_angle([*axis, angle]),
        )
        axis, angle = my_rt.rmat_to_axis_angle(R)
        np.testing.assert_array_almost_equal(
            [*axis, angle], rt.axis_angle_from_matrix(R)
        )
        wxyz = rt.quaternion_from_matrix(R)
        xyzw = [*wxyz[1:], wxyz[0]]
        np.testing.assert_array_almost_equal(my_rt.rmat_to_quat(R), xyzw)
        np.testing.assert_array_almost_equal(
            my_rt.quat_to_rmat(xyzw), rt.matrix_from_quaternion(wxyz)
        )

    def test_transform_equivalents(self):
        R = my_rt.euler_xyz_to_rmat(0.1, 0.2, 0.3)
        p = np.array([0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(
            my_tr.make_transform_mat(R, p), tr.transform_from(R, p)
        )
        T = my_tr.make_transform_mat(R, p)
        np.testing.assert_array_almost_equal(
            my_tr.invert_transform_mat(T), tr.invert_transform(T)
        )


if __name__ == "__main__":
    unittest.main()
