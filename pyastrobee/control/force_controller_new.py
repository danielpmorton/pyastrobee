"""Force/torque control

TODO integrate this with the PID class I made
TODO update the way mass and inertia are handled (attributes of Astrobee?)
TODO see if we can bring back the matrix forms of these gains
TODO add stopping tolerances as inputs?
TODO unify variable naming and argument positioning
TODO enforce max force/torque limits, convert between frames
TODO decide if the stopping tolerance parameters should be in a different location
     (OR, figure out a way to define this for the bag as well)
"""

from typing import Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.trajectories.trajectory import (
    Trajectory,
    TrajectoryLogger,
    stopping_criteria,
)
from pyastrobee.control.controller import ControlLogger
from pyastrobee.utils.quaternions import quaternion_angular_error


class ForcePIDController:
    """PID-style force/torque control

    Args:
        robot_id (int): Pybullet ID of the robot to control
        mass (float): Mass of the robot
        inertia (np.ndarray): Inertia tensor for the robot, shape (3, 3)
        kp (float): Gain for position error
        kv (float): Gain for velocity error
        kq (float): Gain for orientation (quaternion) error
        kw (float): Gain for angular velocity (omega) error
        dt (float): Timestep
        pos_tol (float, optional): Stopping tolerance on position error magnitude. Defaults to 1e-2.
        orn_tol (float, optional): Stopping tolerance on quaternion distance between cur/des. Defaults to 1e-2.
        vel_tol (float, optional): Stopping tolerance on linear velocity error magnitude. Defaults to 1e-2.
        ang_vel_tol (float, optional): Stopping tolerance on angular velocity error magnitude. Defaults to 5e-3.
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    def __init__(
        self,
        robot_id: int,
        mass: float,
        inertia: np.ndarray,
        kp: float,
        kv: float,
        kq: float,
        kw: float,
        dt: float,
        pos_tol: float = 1e-2,
        orn_tol: float = 1e-2,
        vel_tol: float = 1e-2,
        ang_vel_tol: float = 5e-3,
        client: Optional[BulletClient] = None,
    ):
        self.id = robot_id
        self.mass = mass
        self.inertia = inertia
        self.kp = kp
        self.kv = kv
        self.kq = kq
        self.kw = kw
        self.dt = dt
        self.pos_tol = pos_tol
        self.orn_tol = orn_tol
        self.vel_tol = vel_tol
        self.ang_vel_tol = ang_vel_tol
        self.traj_log = TrajectoryLogger()
        self.control_log = ControlLogger()
        self.client: pybullet = pybullet if client is None else client

    def get_force(
        self,
        cur_pos: npt.ArrayLike,
        cur_vel: npt.ArrayLike,
        des_pos: npt.ArrayLike,
        des_vel: npt.ArrayLike,
        des_accel: npt.ArrayLike,
    ) -> np.ndarray:
        """Calculates the required force to achieve a desired pos/vel/accel

        TODO figure out how world/robot frame should be handled

        Args:
            cur_pos (npt.ArrayLike): Current position, shape (3,)
            cur_vel (npt.ArrayLike): Current velocity, shape (3,)
            des_pos (npt.ArrayLike): Desired position, shape (3,)
            des_vel (npt.ArrayLike): Desired velocity, shape (3,)
            des_accel (npt.ArrayLike): Desired acceleration, shape (3,)

        Returns:
            np.ndarray: Force, (Fx, Fy, Fz), shape (3,)
        """
        M = self.mass * np.eye(3)
        pos_err = np.subtract(cur_pos, des_pos)
        vel_err = np.subtract(cur_vel, des_vel)
        return M @ np.asarray(des_accel) - self.kv * vel_err - self.kp * pos_err

    def get_torque(
        self,
        cur_q: npt.ArrayLike,
        cur_w: npt.ArrayLike,
        des_q: npt.ArrayLike,
        des_w: npt.ArrayLike,
        des_a: npt.ArrayLike,
    ) -> np.ndarray:
        """Calculates the required torque to achieve a desired orientation/ang vel/ang accel

        Args:
            cur_q (npt.ArrayLike): Current orientation (XYZW quaternion), shape (4,)
            cur_w (npt.ArrayLike): Current angular velocity (omega), shape (3,)
            des_q (npt.ArrayLike): Desired orientation (XYZW quaternion), shape (4,)
            des_w (npt.ArrayLike): Desired angular velocity (omega), shape (3,)
            des_a (npt.ArrayLike): Desired angular acceleration (alpha), shape (3,)

        Returns:
            np.ndarray: Torque, (Tx, Ty, Tz), shape (3,)
        """
        ang_err = quaternion_angular_error(cur_q, des_q)
        ang_vel_err = cur_w - des_w
        # Standard 3D free-body torque equation based on desired ang. accel and current ang. vel
        torque = self.inertia @ des_a + np.cross(cur_w, self.inertia @ cur_w)
        # Add in the proportional and derivative terms
        return torque - self.kw * ang_vel_err - self.kq * ang_err

    def follow_traj(
        self,
        traj: Trajectory,
        stop_at_end: bool = True,
        max_stop_iters: Optional[int] = None,
    ) -> None:
        """Use PID force/torque control to follow a trajectory

        Args:
            traj (Trajectory): Trajectory with position, orientation, and derivative info across time
            stop_at_end (bool, optional): Whether or not to command the robot to come to a stop at the last
                pose in the trajectory. Defaults to True
            max_stop_iters (int, optional): If stop_at_end is True, this gives a maximum number of iterations to
                allow for stopping. Defaults to None (keep controlling until stopped)
        """
        for i in range(traj.num_timesteps):
            pos, orn, lin_vel, ang_vel = self.get_current_state()
            self.step(
                pos,
                lin_vel,
                orn,
                ang_vel,
                traj.positions[i, :],
                traj.linear_velocities[i, :],
                traj.linear_accels[i, :],
                traj.quaternions[i, :],
                traj.angular_velocities[i, :],
                traj.angular_accels[i, :],
            )
        if stop_at_end:
            self.stop(traj.positions[-1, :], traj.quaternions[-1, :], max_stop_iters)

    def stop(
        self,
        des_pos: npt.ArrayLike,
        des_quat: npt.ArrayLike,
        max_iters: Optional[int] = None,
    ) -> None:
        """Controls the robot to stop at a desired position/orientation

        Args:
            des_pos (npt.ArrayLike): Desired position, shape (3,)
            des_quat (npt.ArrayLike): Desired orientation (XYZW quaternion), shape (4,)
            max_iters (int, optional): Maximum number of control iterations to allow for stopping.
                Defaults to None (keep controlling until stopped)
        """
        des_vel = np.zeros(3)
        des_accel = np.zeros(3)
        des_omega = np.zeros(3)
        des_alpha = np.zeros(3)
        iters = 0
        while True:
            pos, orn, lin_vel, ang_vel = self.get_current_state()
            if stopping_criteria(
                pos,
                orn,
                lin_vel,
                ang_vel,
                des_pos,
                des_quat,
                self.pos_tol,
                self.orn_tol,
                self.vel_tol,
                self.ang_vel_tol,
            ):
                return
            if max_iters is not None and iters >= max_iters:
                print("Maximum iterations reached, stopping unsuccessful")
                return
            self.step(
                pos,
                lin_vel,
                orn,
                ang_vel,
                des_pos,
                des_vel,
                des_accel,
                des_quat,
                des_omega,
                des_alpha,
            )
            iters += 1

    def step(
        self,
        pos: npt.ArrayLike,
        vel: npt.ArrayLike,
        orn: npt.ArrayLike,
        omega: npt.ArrayLike,
        des_pos: npt.ArrayLike,
        des_vel: npt.ArrayLike,
        des_accel: npt.ArrayLike,
        des_orn: npt.ArrayLike,
        des_omega: npt.ArrayLike,
        des_alpha: npt.ArrayLike,
    ) -> None:
        """Steps the controller and the simulation

        Args:
            pos (npt.ArrayLike): Current position, shape (3,)
            vel (npt.ArrayLike): Current linear velocity, shape (3,)
            orn (npt.ArrayLike): Current orientation (XYZW quaternion), shape (4,)
            omega (npt.ArrayLike): Current angular velocity, shape (3,)
            des_pos (npt.ArrayLike): Desired position, shape (3,)
            des_vel (npt.ArrayLike): Desired linear velocity, shape (3,)
            des_accel (npt.ArrayLike): Desired linear acceleration, shape (3,)
            des_orn (npt.ArrayLike): Desired orientation (XYZW quaternion), shape (4,)
            des_omega (npt.ArrayLike): Desired angular velocity, shape (3,)
            des_alpha (npt.ArrayLike): Desired angular acceleration, shape (3,)
        """
        force = self.get_force(pos, vel, des_pos, des_vel, des_accel)
        torque = self.get_torque(orn, omega, des_orn, des_omega, des_alpha)
        self.control_log.log_control(force, torque, self.dt)
        self.client.applyExternalForce(
            self.id, -1, force, list(pos), pybullet.WORLD_FRAME
        )
        self.client.applyExternalTorque(self.id, -1, list(torque), pybullet.WORLD_FRAME)
        self.client.stepSimulation()

    def get_current_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determines the current dynamics state of the robot

        Returns:
            Tuple of:
                np.ndarray: Current position, shape (3,)
                np.ndarray: Current orientation (XYZW quaternion), shape (4,)
                np.ndarray: Current linear velocity, shape (3,)
                np.ndarray: Current angular velocity, shape (3,)
        """
        pos, quat = self.client.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = self.client.getBaseVelocity(self.id)
        self.traj_log.log_state(pos, quat, lin_vel, ang_vel, self.dt)
        # These are originally tuples, so convert to numpy
        return np.array(pos), np.array(quat), np.array(lin_vel), np.array(ang_vel)
