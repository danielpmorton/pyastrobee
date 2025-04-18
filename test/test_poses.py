"""Test cases for pose representation conversions"""

import unittest

import numpy as np

from pyastrobee.utils import poses
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
        # fmt: off
        np.testing.assert_array_almost_equal(pos_euler_xyz, poses.pos_quat_to_pos_euler_xyz(pos_quat))
        np.testing.assert_array_almost_equal(pos_euler_xyz, poses.tmat_to_pos_euler_xyz(T))
        np.testing.assert_array_almost_equal(pos_quat, poses.pos_euler_xyz_to_pos_quat(pos_euler_xyz))
        np.testing.assert_array_almost_equal(pos_quat, poses.tmat_to_pos_quat(T))
        np.testing.assert_array_almost_equal(T, poses.pos_quat_to_tmat(pos_quat))
        np.testing.assert_array_almost_equal(T, poses.pos_euler_xyz_to_tmat(pos_euler_xyz))
        # fmt: on

    def test_adding_delta(self):
        # Evaluate the deltas based on position/euler since it is easier to see if
        # the composition worked properly (since we don't need to add quaternions)
        pose_1 = poses.pos_euler_xyz_to_pos_quat([0, 0, 0, 0, 0, 0])
        pose_1_delta = poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        new_pose_1 = poses.add_global_pose_delta(pose_1, pose_1_delta)
        np.testing.assert_array_almost_equal(
            new_pose_1, poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        )

        # We should be able to evaluate the local delta in the same way as global
        # *for this example only*, because the initial pose is all 0
        pose_2 = poses.pos_euler_xyz_to_pos_quat([0, 0, 0, 0, 0, 0])
        pose_2_delta = poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        new_pose_2 = poses.add_local_pose_delta(pose_2, pose_2_delta)
        np.testing.assert_array_almost_equal(
            new_pose_2, poses.pos_euler_xyz_to_pos_quat([4, 5, 6, 0.1, 0.2, 0.3])
        )

        # Now, do a non-trivial addition in the local frame
        pose_3 = poses.pos_euler_xyz_to_pos_quat([0, 0, 0, 0, 0, np.pi / 4])
        pose_3_delta = poses.pos_euler_xyz_to_pos_quat([1, 0, 0, 0, 0, 0])
        new_pose_3 = poses.add_local_pose_delta(pose_3, pose_3_delta)
        np.testing.assert_array_almost_equal(
            new_pose_3,
            poses.pos_euler_xyz_to_pos_quat(
                [np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0, 0, np.pi / 4]
            ),
        )


if __name__ == "__main__":
    unittest.main()
