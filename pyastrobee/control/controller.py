"""Constraint controller: Simple, non-physically-realistic position control with Pybullet soft constraints

Useful for debugging, but not for real control
"""

import time
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.trajectories.simple_trajectories import point_and_move_pose_traj


class ConstraintController:
    """A non-physically-realistic controller for Astrobee using Pybullet soft constraints

    Args:
        robot (Astrobee): The Astrobee being controlled
    """

    def __init__(self, robot: Astrobee):
        self.robot = robot
        self._pose_command = None
        self.constraint_id = pybullet.createConstraint(
            self.robot.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0)
        )

    @property
    def pose_command(self) -> np.ndarray:
        """Position and xyzw quaternion pose"""
        return self._pose_command

    @pose_command.setter
    def pose_command(self, cmd: npt.ArrayLike) -> np.ndarray:
        if not len(cmd) == 7:
            raise ValueError(f"Invalid pose. Got: {cmd}")
        self._pose_command = cmd

    def go_to_pose(
        self,
        pose: npt.ArrayLike,
        pos_stepsize: float = 0.01,
        orn_stepsize: float = 0.01,
        max_force: float = 500,
        sleep: Optional[float] = None,
    ) -> None:
        """Navigates to a new pose

        Current method (nothing fancy at the moment):
        - Orient towards the goal position
        - Move along a straight line to the goal position
        - Orient towards the goal orientation

        Args:
            pose (npt.ArrayLike): Desired new pose (position + quaternion) for the astrobee
            pos_stepsize (float, optional): Stepsize for consecutive points in the straight-line motion part
                of the trajectory. Defaults to 0.01
            orn_stepsize (float, optional): Stepsize for consecutive points in the orientation parts of the trajectory.
                Defaults to 0.01
            max_force (float, optional): Maximum force to apply to the constraint. Defaults to 500.
            sleep (float, optional): Time to sleep between each step. Defaults to None.
        """
        assert isinstance(pos_stepsize, (float, int)) and pos_stepsize > 0
        assert isinstance(orn_stepsize, (float, int)) and orn_stepsize > 0
        assert isinstance(max_force, (float, int)) and max_force > 0
        assert isinstance(sleep, ((float, int), type(None))) and (
            sleep is None or sleep > 0
        )
        traj = point_and_move_pose_traj(
            self.robot.pose, pose, pos_stepsize, orn_stepsize
        )
        for i in range(traj.shape[0]):
            pybullet.changeConstraint(
                self.constraint_id, traj[i, :3], traj[i, 3:], maxForce=max_force
            )
            pybullet.stepSimulation()
            if sleep is not None:
                time.sleep(sleep)

    def delete_constraint(self) -> None:
        """Deletes the constraint between the Astrobee and the world"""
        pybullet.removeConstraint(self.constraint_id)


def main():
    # Simple controller test
    pybullet.connect(pybullet.GUI)
    robot = Astrobee()
    constraint_controller = ConstraintController(robot)
    constraint_controller.go_to_pose(
        [0.446, -1.338, 0.446, 0.088, 0.067, -0.787, 0.606], sleep=1 / 240
    )
    print("Position control complete")
    input("Press enter to exit")
    pybullet.disconnect()


if __name__ == "__main__":
    main()
