"""Test cases for pose representation conversions"""

import unittest

import numpy as np

from pyastrobee.utils.poses import Pose, tmat_to_pos_euler_xyz
from pyastrobee.utils import rotations as rts
from pyastrobee.utils import transformations as tfs


class PoseTest(unittest.TestCase):
    """Contains test cases to ensure that Pose instances work properly"""

    def test_conversions(self):
        angles = [0.1, 0.2, 0.3]
        R = rts.euler_xyz_to_rmat(angles)
        p = [4, 5, 6]
        q = rts.euler_xyz_to_quat(angles)
        T = tfs.make_transform_mat(R, p)
        pos_euler_xyz = np.array([*p, *angles])
        pos_quat = np.array([*p, *q])
        pose_1 = Pose(pos_euler_xyz=pos_euler_xyz)
        pose_2 = Pose(pos_quat=pos_quat)
        pose_3 = Pose(tmat=T)
        np.testing.assert_array_almost_equal(pose_1.pos_euler_xyz, pose_2.pos_euler_xyz)
        np.testing.assert_array_almost_equal(pose_2.pos_euler_xyz, pose_3.pos_euler_xyz)
        np.testing.assert_array_almost_equal(pose_1.pos_quat, pose_2.pos_quat)
        np.testing.assert_array_almost_equal(pose_2.pos_quat, pose_3.pos_quat)
        np.testing.assert_array_almost_equal(pose_1.tmat, pose_2.tmat)
        np.testing.assert_array_almost_equal(pose_2.tmat, pose_3.tmat)

    def test_reassignment(self):
        # Create a Pose with some initial pose
        angles = [0.1, 0.2, 0.3]
        p = [4, 5, 6]
        pos_euler_xyz = np.array([*p, *angles])
        pose = Pose(pos_euler_xyz=pos_euler_xyz)
        new_angles = [0.3, 0.2, 0.1]
        _ = pose.tmat  # Store a tmat calculation
        # Create a new pose and update the Pose object
        new_pos_euler_xyz = np.array([*p, *new_angles])
        pose.pos_euler_xyz = new_pos_euler_xyz
        new_tmat = pose.tmat
        # The pose.tmat should have been reset since we changed the value via another convention
        comparison_pos_euler_xyz = tmat_to_pos_euler_xyz(new_tmat)
        np.testing.assert_array_almost_equal(
            new_pos_euler_xyz, comparison_pos_euler_xyz
        )


if __name__ == "__main__":
    unittest.main()
