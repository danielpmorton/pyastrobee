"""Force + torque PID control of the Astrobee body

Note: Forces and torques are currently applied in world frame
"""

# TODO: Add better handling of body-frame forces and torques (more aligned with how Astrobee's thrusters operate)
# TODO: Operational space control

from typing import Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyastrobee.trajectories.trajectory import (
    Trajectory,
    TrajectoryLogger,
    stopping_criteria,
)
from pyastrobee.utils.quaternions import quaternion_angular_error
from pyastrobee.utils.rotations import quat_to_rmat


class ForceTorqueController:
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
        max_force (Optional[float]): Limit on the applied force magnitude. Defaults to None (no limit)
        max_torque (Optional[float]): Limit on the applied torque magnitude. Defaults to None (no limit)
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
        max_force: Optional[float] = None,
        max_torque: Optional[float] = None,
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
        self.max_force = max_force
        self.max_torque = max_torque
        self.traj_log = TrajectoryLogger()
        self.control_log = ControlLogger()
        self.client: pybullet = pybullet if client is None else client
        # Parameters for checking if there was a quaternion sign flip
        self.last_quat = np.array(self.client.getBasePositionAndOrientation(self.id)[1])
        self.quat_sign = 1

    # TODO figure out how world/robot frame should be handled
    def get_force(
        self,
        cur_pos: npt.ArrayLike,
        cur_vel: npt.ArrayLike,
        des_pos: npt.ArrayLike,
        des_vel: npt.ArrayLike,
        des_accel: npt.ArrayLike,
    ) -> np.ndarray:
        """Calculates the required force to achieve a desired pos/vel/accel

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
        if np.allclose(cur_q, -1 * self.last_quat, atol=1e-3):
            print("Quaternion flip detected")
            self.quat_sign *= -1
        ang_err = quaternion_angular_error(cur_q * self.quat_sign, des_q)
        self.last_quat = cur_q
        ang_vel_err = cur_w - des_w
        R = quat_to_rmat(cur_q)
        world_inertia = R @ self.inertia @ R.T
        # Standard 3D free-body torque equation based on desired ang. accel and current ang. vel
        # Note: this ignores the m * r x a term
        torque = world_inertia @ des_a + np.cross(cur_w, world_inertia @ cur_w)
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
        step_sim: bool = True,
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
            step_sim (bool, optional): Whether to step the sim or not (This should almost always be true except for if
                there are multiple active controllers in the simulation. In that case, the sim must be stepped manually
                with this flag as False on each controller). Defaults to True.
        """
        force = self.get_force(pos, vel, des_pos, des_vel, des_accel)
        torque = self.get_torque(orn, omega, des_orn, des_omega, des_alpha)
        # Clamp the maximum force/torque if needed
        if self.max_force is not None:
            force_mag = np.linalg.norm(force)
            if force_mag > self.max_force:
                force = self.max_force * (force / force_mag)
        if self.max_torque is not None:
            torque_mag = np.linalg.norm(torque)
            if torque_mag > self.max_torque:
                torque = self.max_torque * (torque / torque_mag)
        self.control_log.log_control(force, torque, self.dt)
        self.client.applyExternalForce(
            self.id, -1, force, list(pos), pybullet.WORLD_FRAME
        )
        self.client.applyExternalTorque(self.id, -1, list(torque), pybullet.WORLD_FRAME)
        if step_sim:
            self.client.stepSimulation()

    def get_current_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determines the current dynamics state of the robot

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


class ControlLogger:
    """Class for maintaining a history of control inputs for plottting or further analysis

    Any conversions between world frame / robot frame should be done before storing the force/torque
    data, depending on what is of interest
    """

    def __init__(self):
        self._forces = []
        self._torques = []
        self._times = []

    @property
    def forces(self) -> np.ndarray:
        return np.atleast_2d(self._forces)

    @property
    def torques(self) -> np.ndarray:
        return np.atleast_2d(self._torques)

    @property
    def times(self) -> np.ndarray:
        return np.array(self._times)

    def log_control(
        self, force: npt.ArrayLike, torque: npt.ArrayLike, dt: Optional[float] = None
    ) -> None:
        """Logs the forces and torques applied in a simulation step

        Args:
            force (npt.ArrayLike): Applied force (Fx, Fy, Fz), shape (3,)
            torque (npt.ArrayLike): Applied torque (Tx, Ty, Tz), shape (3,)
            dt (Optional[float]): Time elapsed since the previous step. Defaults to None.
        """
        self._forces.append(force)
        self._torques.append(torque)
        if dt is not None and len(self._times) == 0:
            self._times.append(0.0)
        elif dt is not None:
            self._times.append(self._times[-1] + dt)

    def plot(
        self,
        max_force: Optional[npt.ArrayLike] = None,
        max_torque: Optional[npt.ArrayLike] = None,
        show: bool = True,
    ) -> Figure:
        """Plot the stored history of control inputs

        Args:
            max_force (Optional[npt.ArrayLike]): Applied force limits (Fx_max, Fy_max, Fz_max), shape (3,)
                Defaults to None (Don't indicate the limit on the plots)
            max_torque (Optional[npt.ArrayLike]): Applied torque limits (Tx_max, Ty_max, Tz_max), shape (3,)
                Defaults to None (Don't indicate the limit on the plots)
            show (bool, optional): Whether or not to show the plot. Defaults to True

        Returns:
            Figure: Matplotlib figure containing the plots
        """
        return plot_control(
            self.forces, self.torques, self.times, max_force, max_torque, show
        )


def plot_control(
    forces: np.ndarray,
    torques: np.ndarray,
    times: Optional[np.ndarray] = None,
    max_force: Optional[npt.ArrayLike] = None,
    max_torque: Optional[npt.ArrayLike] = None,
    show: bool = True,
    fmt: str = "k-",
) -> Figure:
    """Plots a recorded history of force/torque control inputs

    Args:
        forces (np.ndarray): Sequence of force inputs (Fx, Fy, Fz), shape (n, 3)
        torques (np.ndarray): Sequence of torque inputs (Tx, Ty, Tz), shape (n, 3)
        times (Optional[np.ndarray], optional): Times corresponding to control inputs, shape (n,).
            Defaults to None, in which case control inputs will be plotted against timesteps
        max_force (Optional[npt.ArrayLike]): Applied force limits (Fx_max, Fy_max, Fz_max), shape (3,)
            Defaults to None (Don't indicate the limit on the plots)
        max_torque (Optional[npt.ArrayLike]): Applied torque limits (Tx_max, Ty_max, Tz_max), shape (3,)
            Defaults to None (Don't indicate the limit on the plots)
        show (bool, optional): Whether or not to display the plot. Defaults to True.
        fmt (str, optional): Matplotlib line specification. Defaults to "k-"

    Returns:
        Figure: Matplotlib figure containing the plots
    """
    fig = plt.figure()
    if times is not None:
        x_axis = times
        x_label = "Time, s"
    else:
        x_axis = np.arange(forces.shape[0])
        x_label = "Timesteps"
    subfigs = fig.subfigures(2, 1)
    top_axes = subfigs[0].subplots(1, 3)
    bot_axes = subfigs[1].subplots(1, 3)
    force_labels = ["Fx", "Fy", "Fz"]
    torque_labels = ["Tx", "Ty", "Tz"]
    # Plot force info on the top axes
    for i, ax in enumerate(top_axes):
        ax.plot(x_axis, forces[:, i], fmt)
        if max_force is not None:
            ax.plot(x_axis, max_force[i] * np.ones_like(x_axis), "--")
        ax.set_title(force_labels[i])
        ax.set_xlabel(x_label)
    # Plot torque info on the bottom axes
    for i, ax in enumerate(bot_axes):
        ax.plot(x_axis, torques[:, i], fmt)
        if max_torque is not None:
            ax.plot(x_axis, max_torque[i] * np.ones_like(x_axis), "--")
        ax.set_title(torque_labels[i])
        ax.set_xlabel(x_label)
    if show:
        plt.show()
    return fig
