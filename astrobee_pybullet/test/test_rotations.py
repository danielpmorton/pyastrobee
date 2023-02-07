"""Test cases for rotations

TODO add some test cases to test against pytransform3d
"""

import unittest
import numpy as np

from astrobee_pybullet.utils import rotations as rot


class RotationsTest(unittest.TestCase):
    """Contains test cases for the rotations utility functions"""

    def test_simple_rmats(self):
        self.assertTrue(np.array_equal(rot.Rx(0), np.eye(3)))
        self.assertTrue(np.array_equal(rot.Ry(0), np.eye(3)))
        self.assertTrue(np.array_equal(rot.Rz(0), np.eye(3)))
        np.testing.assert_array_almost_equal(
            rot.Rx(np.deg2rad(90)), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        )
        np.testing.assert_array_almost_equal(
            rot.Ry(np.deg2rad(90)), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        )
        np.testing.assert_array_almost_equal(
            rot.Rz(np.deg2rad(90)), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        )

    def test_euler_angles(self):
        np.testing.assert_array_almost_equal(
            rot.euler_angles_to_rmat("x", np.deg2rad(90)), rot.Rx(np.deg2rad(90))
        )
        np.testing.assert_array_almost_equal(
            rot.euler_angles_to_rmat("y", np.deg2rad(90)), rot.Ry(np.deg2rad(90))
        )
        np.testing.assert_array_almost_equal(
            rot.euler_angles_to_rmat("z", np.deg2rad(90)), rot.Rz(np.deg2rad(90))
        )
        np.testing.assert_array_almost_equal(
            rot.euler_angles_to_rmat("xy", np.deg2rad(45), np.deg2rad(45)),
            np.array(
                [
                    [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                    [0.5, np.sqrt(2) / 2, -0.5],
                    [-0.5, np.sqrt(2) / 2, 0.5],
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            rot.euler_angles_to_rmat(
                "xyz", np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)
            ),
            np.array(
                [
                    [0.7500000, -0.4330127, 0.5000000],
                    [0.6495190, 0.6250000, -0.4330127],
                    [-0.1250000, 0.6495190, 0.7500000],
                ]
            ),
            decimal=6,
        )

        with self.assertRaises(ValueError):
            rot.euler_angles_to_rmat("xyz", 90)
        with self.assertRaises(ValueError):
            rot.euler_angles_to_rmat("x", 1, 2, 3)

    def test_rmat_check(self):
        self.assertTrue(rot.check_rotation_mat(rot.Rx(1)))
        self.assertTrue(
            rot.check_rotation_mat(rot.euler_angles_to_rmat("xyz", 1, 2, 3))
        )
        self.assertFalse(rot.check_rotation_mat(np.random.rand(3, 3)))
        self.assertFalse(rot.check_rotation_mat(np.random.rand(4, 4)))
        self.assertFalse(rot.check_rotation_mat(np.random.rand(1, 3)))

    def test_rmat_angle_conversion(self):
        input_angles = [0.1, 0.2, 0.3]
        # Euler ZYX
        R = rot.euler_angles_to_rmat("zyx", *input_angles)
        output_angles = rot.rmat_to_euler_zyx(R)
        np.testing.assert_array_almost_equal(input_angles, output_angles)
        # Fixed XYZ
        R2 = rot.fixed_angles_to_rmat("xyz", *input_angles)
        output_angles_2 = rot.rmat_to_fixed_xyz(R2)
        np.testing.assert_array_almost_equal(input_angles, output_angles_2)
        # Euler XYZ
        R3 = rot.euler_angles_to_rmat("xyz", *input_angles)
        output_angles_3 = rot.rmat_to_euler_xyz(R3)
        np.testing.assert_array_almost_equal(input_angles, output_angles_3)
        # Fixed ZYX
        R4 = rot.fixed_angles_to_rmat("zyx", *input_angles)
        output_angles_4 = rot.rmat_to_fixed_zyx(R4)
        np.testing.assert_array_almost_equal(input_angles, output_angles_4)

    def test_euler_param_conversion(self):
        R_in = rot.euler_angles_to_rmat("xyz", 0.1, 0.2, 0.3)
        params = rot.rmat_to_euler_params(R_in)
        R_out = rot.euler_params_to_rmat(*params)
        np.testing.assert_array_almost_equal(R_in, R_out)

    def test_rmat_axis_angle_conversion(self):
        R_in = rot.euler_angles_to_rmat("xyz", 0.1, 0.2, 0.3)
        axis, angle = rot.rmat_to_axis_angle(R_in)
        R_out = rot.axis_angle_to_rmat(axis, angle)
        np.testing.assert_array_almost_equal(R_in, R_out)


if __name__ == "__main__":
    unittest.main()
