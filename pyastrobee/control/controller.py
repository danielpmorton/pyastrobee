"""Position, velocity, and force controllers for Astrobee motion

TODO
- !! Check on the 1/5 scaling issue with velocities?
  https://github.com/bulletphysics/bullet3/issues/2237
- Add ability to command deltas
- Add logic/planning/PID to control more than just velocity with velocity, force with force, ...
- Unify function between controllers with a follow_trajectory() function?

TODO (Long-horizon)
- Add support for multiple robots (maybe this is already viable if we just initialize another controller for robot #2)

NOTE
- Currently working with position control as "pose control", linking position + orientation
  IDK if it is valuable to control these two things separately, but we may want to include this
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

import pybullet
import numpy as np
from numpy.typing import ArrayLike

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.config.astrobee_motion import (
    LINEAR_ACCEL_LIMIT,
    LINEAR_SPEED_LIMIT,
    ANGULAR_ACCEL_LIMIT,
    ANGULAR_SPEED_LIMIT,
    MAX_FORCE,
    MAX_TORQUE,
)  # Change this import handling?
from pyastrobee.control.physics_models import drag_force_model
from pyastrobee.utils.python_utils import print_red
from pyastrobee.control.planner import point_and_move_pose_traj


class Controller(ABC):
    """Abstract class for controllers

    Position/velocity/force controllers should inherit from this to share a common structure
    """

    def __init__(self, robot: Astrobee):
        self.robot = robot

        # Initialize variables for properties (TODO decide if these should be in the inherited classes)
        # Note that the "default value" for force/torque should be 0,
        # but we don't want to set the pose or velocity to 0 accidentally, so we use None instead
        self._pose_command = None
        self._velocity_command = None
        self._angular_velocity_command = None
        self._force_command = np.array([0, 0, 0])
        self._torque_command = np.array([0, 0, 0])

        self.dt = 1 / 120  # Timestep (TODO should this be in a different place?)
        self.dv_max = LINEAR_ACCEL_LIMIT * self.dt  # TODO move this to controller
        self.dw_max = ANGULAR_ACCEL_LIMIT * self.dt
        self.step_count = 0  # Initialization

    @abstractmethod
    def update(self):
        """Runs a single-timestep update on whatever parameter is being controlled (position, velocity, force)

        This function should be implemented in the controllers inheriting from this abstract class
        """
        pass

    def step(self):
        """Updates the controller parameters and steps through the simulation"""
        self.update()
        pybullet.stepSimulation()
        self.step_count += 1
        # time.sleep(self.dt) # DECIDE IF NEEDED HERE

    def run(self, max_iter: Optional[int] = None):
        """Runs the simulation for the controller for a specified number of steps or indefinitely

        Args:
            max_iter (Optional[int]): Maximum number of iterations to run the simulation. Defaults to None,
                in which case the simulation will run indefinitely
        """
        if max_iter is None:
            max_iter = float("inf")
        while self.step_count < max_iter:
            self.step()
            time.sleep(self.dt)  # Should this be in step?

    def _validate_cmds(
        self,
        pose: Optional[ArrayLike] = None,
        vel: Optional[ArrayLike] = None,
        ang_vel: Optional[ArrayLike] = None,
        force: Optional[ArrayLike] = None,
        torque: Optional[ArrayLike] = None,
    ):
        """Confirms that inputs are the correct shape and within the Astrobee's force/speed limits

        - All commands are defined in world frame
        - TODO decide if this should be implemented in the inherited classes

        Args:
            pose (Optional[ArrayLike]): Pose command (position + XYZW quaternion). Defaults to None.
            vel (Optional[ArrayLike]): Linear velocity command ([vx, vy, vz]). Defaults to None.
            ang_vel (Optional[ArrayLike]): Angular velocity command ([wx, wy, wz]). Defaults to None.
            force (Optional[ArrayLike]): Force command ([Fx, Fy, Fz]). Defaults to None.
            torque (Optional[ArrayLike]): Torque command ([Tx, Ty, Tz]). Defaults to None.

        Raises:
            ValueError: If any of the inputs are invalid
        """
        if pose is not None:
            if len(pose) != 7:
                raise ValueError(f"Invalid pose. Got: {pose}")
        if vel is not None:
            if not len(vel) == 3:
                raise ValueError(f"Invalid velocity vector.\nGot: {vel}")
            if np.linalg.norm(vel) > LINEAR_SPEED_LIMIT:
                raise ValueError(
                    "Commanded velocity exceeds the speed limit.\n"
                    + f"Got: {vel}\n"
                    + f"Limit: {LINEAR_SPEED_LIMIT}"
                )
        if ang_vel is not None:
            if not len(ang_vel) == 3:
                raise ValueError(f"Invalid angular velocity vector.\nGot: {ang_vel}")
            if np.linalg.norm(ang_vel) > ANGULAR_SPEED_LIMIT:
                raise ValueError(
                    "Commanded angular velocity exceeds the speed limit.\n"
                    + f"Got: {ang_vel}\n"
                    + f"Limit: {ANGULAR_SPEED_LIMIT}"
                )
        if force is not None:
            if not len(force) == 3:
                raise ValueError(f"Invalid force vector.\nGot: {force}")
            R_W2R = self.robot.rmat.T  # World to robot
            local_force_cmd = R_W2R @ force
            if np.any(local_force_cmd > MAX_FORCE):
                # TODO decide if this should be a warning instead, and then clip the value?
                raise ValueError(
                    "Commanded force exceeds the limit.\n"
                    + f"World frame command: {force}\n"
                    + f"Local command: {local_force_cmd}\n"
                    + f"Limit: {MAX_FORCE}"
                )
        if torque is not None:
            if not len(torque) == 3:
                raise ValueError(f"Invalid torque vector.\nGot: {torque}")
            R_W2R = self.robot.rmat.T  # World to robot
            local_torque_cmd = R_W2R @ torque
            if np.any(local_torque_cmd > MAX_TORQUE):
                # TODO decide if this should be a warning instead, and then clip the value?
                raise ValueError(
                    "Commanded torque exceeds the limit.\n"
                    + f"World frame command: {torque}\n"
                    + f"Local command: {local_torque_cmd}\n"
                    + f"Limit: {MAX_TORQUE}"
                )


class PoseController:
    """Position control for the Astrobee

    Args:
        robot (Astrobee): The Astrobee being controlled

    TODO
    - This should be structured in the same way as the other controllers (Inherit from Controller(), and define the
      update() function), so that we can call the controllers in the same way
    - This will likely require us to define states such as IDLE, ALIGNING, FOLLOWING, and do different things in
      the update() function based on the state and transitions
    - Rename to ConstraintController?
    """

    def __init__(self, robot: Astrobee):
        self.robot = robot
        self._pose_command = None

        self.orn_tol = 0.03  # TODO update this
        self.pos_tol = 0.1  # TODO update this. Rename to stepsize?

        # Copying over the original constraint from the Astrobee class
        # TODO still need to play around with the input parameters here
        self.constraint_id = pybullet.createConstraint(
            self.robot.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0)
        )

    @property
    def pose_command(self) -> np.ndarray:
        return self._pose_command

    @pose_command.setter
    def pose_command(self, cmd: ArrayLike) -> np.ndarray:
        # TODO inherit the validation function from Controller!!
        if not len(cmd) == 7:
            raise ValueError(f"Invalid pose. Got: {cmd}")
        self._pose_command = cmd

    def go_to_pose(self, pose: ArrayLike, max_force: float = 500) -> None:
        """Navigates to a new pose

        Current method (nothing fancy at the moment):
        - Orient towards the goal position
        - Move along a straight line to the goal position
        - Orient towards the goal orientation

        Args:
            pose (ArrayLike): Desired new pose (position + quaternion) for the astrobee
            max_force (float, optional): Maximum force to apply to the constraint. Defaults to 500
        """
        pos_stepsize = 0.01  # TODO move this
        orn_stepsize = 0.01
        traj = point_and_move_pose_traj(
            self.robot.pose, pose, pos_stepsize, orn_stepsize
        )
        for i in range(traj.shape[0]):
            pybullet.changeConstraint(
                self.constraint_id, traj[i, :3], traj[i, 3:], maxForce=max_force
            )
            pybullet.stepSimulation()
            time.sleep(1 / 120)

    def delete_constraint(self) -> None:
        """Deletes the constraint between the Astrobee and the world"""
        pybullet.removeConstraint(self.constraint_id)


class VelocityController(Controller):
    """Velocity control for the Astrobee

    Args:
        robot (Astrobee): The Astrobee being controlled
    """

    def __init__(self, robot: Astrobee):
        super().__init__(robot)
        check_for_constraint(robot)

    @property
    def velocity_command(self) -> np.ndarray:
        """Desired velocity ([vx, vy, vz], world frame) of the Astrobee base

        When setting this value, velocities must be within the operating mode's speed limits
        """
        return self._velocity_command

    @velocity_command.setter
    def velocity_command(self, cmd: ArrayLike):
        self._validate_cmds(vel=cmd)
        self._velocity_command = np.array(cmd)

    @property
    def angular_velocity_command(self) -> np.ndarray:
        """Desired angular velocity ([wx, wy, wz], world frame) of the Astrobee base

        When setting this value, velocities must be within the operating mode's speed limits
        """
        return self._angular_velocity_command

    @angular_velocity_command.setter
    def angular_velocity_command(self, cmd: ArrayLike):
        self._validate_cmds(ang_vel=cmd)
        self._angular_velocity_command = np.array(cmd)

    def update(self):
        """Updates the velocities on the robot for a single timestep"""

        # TODO see if any of the logic here can be simplified
        # Updates the current velocity of the robot for a single timestep
        cur_vel = self.robot.velocity
        cur_ang_vel = self.robot.angular_velocity
        if self.velocity_command is not None:
            dv = self.velocity_command - cur_vel
            # Enforce the maximum linear acceleration constraint
            # NOTE this seems super slow right now
            if np.linalg.norm(dv) > self.dv_max:
                vel_cmd = cur_vel + self.dv_max * dv / np.linalg.norm(dv)
            else:
                vel_cmd = self.velocity_command
            # vel_cmd = self.velocity_command  # TODO remove
        else:
            vel_cmd = cur_vel
        if self.angular_velocity_command is not None:
            dw = self.angular_velocity_command - cur_ang_vel
            # Enforce the maximum angular acceleration constraint
            if np.linalg.norm(dw) > self.dw_max:
                ang_vel_cmd = cur_ang_vel + self.dw_max * dw / np.linalg.norm(dw)
            else:
                ang_vel_cmd = self.angular_velocity_command
            # ang_vel_cmd = self.angular_velocity_command  # TODO remove
        else:
            ang_vel_cmd = cur_ang_vel
        # print("sending velocity", vel_cmd)
        pybullet.resetBaseVelocity(self.robot.id, vel_cmd, ang_vel_cmd)

    def set_local_velocity_cmds(
        self,
        linear: Optional[ArrayLike] = None,
        angular: Optional[ArrayLike] = None,
    ) -> None:
        """Sets a desired linear/angular velocity command in the local (robot) frame

        Args:
            linear (Optional[ArrayLike]): Desired linear velocity along the robot's reference frame axes.
                Defaults to None, AKA "don't change the current command"
            angular (Optional[ArrayLike]): Desired angular velocity about the robot's reference frame axes.
                Defaults to None, AKA "don't change the current command"
        """
        # Transform the velocity vectors from robot frame to world frame, then set the command
        R_R2W = self.robot.rmat
        self.velocity_command = R_R2W @ linear
        self.angular_velocity_command = R_R2W @ angular


class ForceController(Controller):
    """Force control for the Astrobee

    Args:
        robot (Astrobee): The Astrobee being controlled
    """

    def __init__(self, robot: Astrobee):
        super().__init__(robot)
        check_for_constraint(robot)

    @property
    def force_command(self) -> np.ndarray:
        """Desired thrust force ([Fx, Fy, Fz], world frame) to be applied to the Astrobee base

        When setting this value, forces must be within the fans' maximum applied thrust limits
        """
        return self._force_command

    @force_command.setter
    def force_command(self, cmd: ArrayLike):
        self._validate_cmds(force=cmd)
        self._force_command = np.array(cmd)

    @property
    def torque_command(self) -> np.ndarray:
        """Desired torque ([Tx, Ty, Tz], world frame) to be applied to the Astrobee base

        When setting this value, torques must be within the fans' maximum applied torque limits
        """
        return self._torque_command

    @torque_command.setter
    def torque_command(self, cmd: ArrayLike):
        self._validate_cmds(torque=cmd)
        self._torque_command = cmd

    def update(self):
        """Updates the forces on the robot for a single timestep

        TODO
        - Make sure we don't exceed the speed limit
          (For the accel limit, we can assume that adhering to the maximum force limits are enough)
        - Include torque effect of drag?
        - Include more robust force model for the fan thrust
        """

        cur_vel = self.robot.velocity
        drag_force = drag_force_model(cur_vel)
        net_force = self.force_command + drag_force

        # TODO this needs a lot of testing
        # Decide if we should have a better idea of world frame vs local frame
        base_idx = -1  # Apply the force/torque to the Astrobee's base
        point = np.array([0.0, 0.0, 0.0])  # Point in base frame where force is applied
        pybullet.applyExternalForce(
            self.robot.id, base_idx, net_force, point, pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(
            self.robot.id, base_idx, self.torque_command, pybullet.WORLD_FRAME
        )

    def set_local_force_cmds(
        self,
        force: Optional[ArrayLike] = None,
        torque: Optional[ArrayLike] = None,
    ) -> None:
        """Sets a desired force/torque command in the local (robot) frame

        Args:
            force (Optional[ArrayLike]): Desired force along the robot's reference frame axes.
                Defaults to None, AKA "don't change the current command"
            torque (Optional[ArrayLike]): Desired torque about the robot's reference frame axes.
                Defaults to None, AKA "don't change the current command"
        """
        # Transform the force/torque vectors from robot frame to world frame, then set the command
        R_R2W = self.robot.rmat
        self.force_command = R_R2W @ force
        self.torque_command = R_R2W @ torque


def check_for_constraint(robot: Astrobee):
    """Checks to see if a constraint between the robot/world is currently enabled, and prints a warning if so

    (Constraints are used for position control, but if it is still active when we use a different method,
    it will not allow the robot to respond to velocities/forces)

    Args:
        robot (Astrobee): The Astrobee being controlled
    """
    n = pybullet.getNumConstraints()
    # There may be other constraints in use (for instance, between robot/bag), so check explicitly between robot/world
    for constraint_id in range(1, n + 1):  # These seem to be 1-indexed
        info = pybullet.getConstraintInfo(constraint_id)
        parent_id = info[0]
        child_id = info[2]
        if parent_id == robot.id and child_id == -1:
            print_red(
                "WARNING: There seems to be a constraint between the robot and world!\n"
                + "This might be left-over from a previous control method\n"
                + "Confirm that this should still be active!"
            )


if __name__ == "__main__":
    # Some assorted commands just to see the behavior of each controller
    pybullet.connect(pybullet.GUI)
    bee = Astrobee()
    position_controller = PoseController(bee)
    position_controller.go_to_pose([0.446, -1.338, 0.446, 0.088, 0.067, -0.787, 0.606])
    print("Position control complete")
    position_controller.delete_constraint()
    vel_controller = VelocityController(bee)
    vel_controller.velocity_command = np.array([0.2, 0.2, 0.2])
    vel_controller.run(max_iter=2000)
    print("Velocity control complete")
    force_controller = ForceController(bee)
    force_controller.force_command = np.array([0.5, 0, 0])
    force_controller.run(max_iter=2000)
    print("Force control complete")
