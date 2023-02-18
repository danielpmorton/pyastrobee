from enum import Enum

import numpy as np

from pyastrobee.utils.rotations import euler_xyz_to_rmat, rmat_to_euler_zyx, quat_to_rmat, Quaternion

# Does it even make sense to have the xyz notation in euler? We can maybe just assume we are dealing with XYZ
# Likewise, do we even need to worry about XYZW vs WXYZ quaternions, or can we just stick to XYZW and call it a day?

def pos_euler_xyz_to_tmat(pose):
    raise NotImplementedError

def pos_euler_xyz_to_pos_quat(pose):
    raise NotImplementedError

def tmat_to_pos_euler_xyz(tmat):
    raise NotImplementedError

def tmat_to_pos_quat(tmat):
    raise NotImplementedError

def pos_quat_to_tmat(pose):
    raise NotImplementedError

def pos_quat_to_pos_euler_xyz(pose):
    raise NotImplementedError

def add_poses(pose1, pose2):
    raise NotImplementedError

def subtract_poses(pose1, pose2):
    raise NotImplementedError



class Pose:
    class Convention(Enum):
        POS_EULER_XYZ = 1
        POS_QUAT = 2
        TMAT = 3

    def __init__(self, pose):
        self._pose = pose

        self._pos_euler_xyz = None
        self._pos_quat = None
        self._tmat = None

        if len(pose) == 6:
            # TODO conventions might not be needed anymore?
            self._convention = Pose.Convention.POS_EULER_XYZ
            self._pos_euler_xyz = pose
        elif len(pose) == 7:
            self._convention = Pose.Convention.POS_QUAT
            self._pos_quat = pose
        elif isinstance(pose, np.ndarray) and pose.shape == (4,4):
            self._convention = Pose.Convention.TMAT
            self._tmat = pose
        else:
            raise ValueError(f"Invalid pose type.\nGot: {pose}")

    @property
    def pos_euler_xyz(self):
        if not self._pos_euler_xyz:
            if self._convention == Pose.Convention.POS_QUAT:
                self._pos_euler_xyz = pos_quat_to_pos_euler_xyz(self._pose)
            elif self._convention == Pose.Convention.TMAT:
                self._pos_euler_xyz = tmat_to_pos_euler_xyz(self._pose)
        return self._pos_euler_xyz

    @property
    def pos_quat(self):
        if not self._pos_quat:
            if self._convention == Pose.Convention.POS_EULER_XYZ:
                self._pos_quat = pos_euler_xyz_to_pos_quat(self._pose)
            elif self._convention    == Pose.Convention.TMAT:
                self._pos_quat = tmat_to_pos_quat(self._pose)
        return self._pos_quat

    @property
    def tmat(self):
        if not self._tmat:
            if self._convention == Pose.Convention.POS_EULER_XYZ:
                self._tmat = pos_euler_xyz_to_tmat(self._pose)
            elif self._convention == Pose.Convention.POS_QUAT:
                self._tmat = pos_quat_to_tmat(self._pose)
        return self._tmat




class ArmPose(Pose):
    pass

class RobotPose(Pose):
    pass
    