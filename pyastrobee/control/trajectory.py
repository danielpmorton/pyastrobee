"""Trajectory class for Astrobee control, and trajectory-associated helpers"""

from typing import Optional, Union

import pybullet
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyastrobee.utils.poses import batched_pos_quats_to_tmats
from pyastrobee.utils.debug_visualizer import visualize_frame


class Trajectory:
    """Trajectory class: Keeps track of a sequence of poses/velocities over a period of time

    - All arguments can be omitted as needed (for instance, a pose-only trajectory without velocities,
        or a trajectory without time information)
    - All positions/orientations/velocities are assumed to be defined in world frame

    Args:
        positions (Optional[npt.ArrayLike]): Sequence of XYZ positions, shape (n, 3)
        quats (Optional[npt.ArrayLike]): Sequence of XYZW quaternions, shape (n, 4)
        lin_vels (Optional[npt.ArrayLike]): Sequence of (vx, vy, vz) linear velocities, shape (n, 3)
        ang_vels (Optional[npt.ArrayLike]): Sequence of (wx, wy, wz) angular velocities, shape (n, 3)
        times (Optional[npt.ArrayLike]): Times corresponding to each trajectory entry, shape (n)
    """

    def __init__(
        self,
        positions: Optional[npt.ArrayLike] = None,
        quats: Optional[npt.ArrayLike] = None,
        lin_vels: Optional[npt.ArrayLike] = None,
        ang_vels: Optional[npt.ArrayLike] = None,
        times: Optional[npt.ArrayLike] = None,
    ):
        # TODO decide if there is ever a use-case for not passing in the positions
        self.positions = np.atleast_2d(positions) if positions is not None else None
        self.quaternions = np.atleast_2d(quats) if quats is not None else None
        self.linear_velocities = (
            np.atleast_2d(lin_vels) if lin_vels is not None else None
        )
        self.angular_velocities = (
            np.atleast_2d(ang_vels) if ang_vels is not None else None
        )
        self.times = np.atleast_1d(times) if times is not None else None
        self.num_timesteps = self.positions.shape[0]

    @property
    def poses(self) -> np.ndarray:
        """Pose array (position + xyzw quaternion), shape (n, 7)"""
        if self.positions is None:
            raise ValueError("No position information available")
        if self.quaternions is None:
            raise ValueError("No orientation information available")
        return np.column_stack([self.positions, self.quaternions])

    @property
    def tmats(self) -> np.ndarray:
        """Poses expressed as transformation matrices, shape (n, 4, 4)"""
        return batched_pos_quats_to_tmats(self.poses)

    @property
    def velocities(self) -> np.ndarray:
        """Array of linear and angular velocities, shape (n, 6)"""
        if self.linear_velocities is None:
            raise ValueError("No linear velocity information available")
        if self.angular_velocities is None:
            raise ValueError("No angular velocity information available")
        return np.column_stack([self.linear_velocities, self.angular_velocities])

    def visualize(self) -> None:
        """View the trajectory in Pybullet"""
        connection_status = pybullet.isConnected()
        # Bring up the Pybullet GUI if needed
        if not connection_status:
            pybullet.connect(pybullet.GUI)
        visualize_traj(self)
        input("Press Enter to continue")
        # Disconnect Pybullet if we originally weren't connected
        if not connection_status:
            pybullet.disconnect()

    def plot(self, show: bool = True) -> Figure:
        """Plot the trajectory components over time

        Args:
            show (bool, optional): Whether or not to display the plot. Defaults to True.

        Returns:
            Figure: Matplotlib figure containing the plots
        """
        return plot_traj(self, show)


def plot_traj(traj: Trajectory, show: bool = True) -> Figure:
    """Plot the trajectory components over time

    Args:
        show (bool, optional): Whether or not to display the plot. Defaults to True.

    Returns:
        Figure: Matplotlib figure containing the plots
    """

    def _plot_positions(axes: np.ndarray[plt.Axes]):
        """Helper function to plot position information on the desired axes"""
        if traj.positions is None:
            return
        labels = ["x", "y", "z"]
        for i, ax in enumerate(axes):
            ax.plot(x_axis, traj.positions[:, i])
            ax.set_title(labels[i])
            ax.set_xlabel(x_label)

    def _plot_quaternions(axes: np.ndarray[plt.Axes]):
        """Helper function to plot quaternion information on the desired axes"""
        if traj.quaternions is None:
            return
        labels = ["qx", "qy", "qz", "qw"]
        for i, ax in enumerate(axes):
            ax.plot(x_axis, traj.quaternions[:, i])
            ax.set_title(labels[i])
            ax.set_xlabel(x_label)

    def _plot_linear_velocities(axes: np.ndarray[plt.Axes]):
        """Helper function to plot linear velocity information on the desired axes"""
        if traj.linear_velocities is None:
            return
        labels = ["vx", "vy", "vz"]
        for i, ax in enumerate(axes):
            ax.plot(x_axis, traj.linear_velocities[:, i])
            ax.set_title(labels[i])
            ax.set_xlabel(x_label)

    def _plot_angular_velocities(axes: np.ndarray[plt.Axes]):
        """Helper function to plot angular velocity information on the desired axes"""
        if traj.angular_velocities is None:
            return
        labels = ["wx", "wy", "wz"]
        for i, ax in enumerate(axes):
            ax.plot(x_axis, traj.angular_velocities[:, i])
            ax.set_title(labels[i])
            ax.set_xlabel(x_label)

    if traj.times is not None:
        x_axis = traj.times
        x_label = "Time, s"
    else:
        x_axis = range(traj.num_timesteps)
        x_label = "Timesteps"

    fig = plt.figure()
    # Make an extended version of the plot if we have velocity information
    if traj.linear_velocities is not None or traj.angular_velocities is not None:
        subfigs = fig.subfigures(2, 2)
        top_left = subfigs[0, 0].subplots(1, 3)
        _plot_positions(top_left)
        top_right = subfigs[0, 1].subplots(1, 3)
        _plot_linear_velocities(top_right)
        bot_left = subfigs[1, 0].subplots(1, 4)
        _plot_quaternions(bot_left)
        bot_right = subfigs[1, 1].subplots(1, 3)
        _plot_angular_velocities(bot_right)
    else:  # Only make a section for the position/orientation information
        subfigs = fig.subfigures(2, 1)
        top = subfigs[0].subplots(1, 3)
        _plot_positions(top)
        bot = subfigs[1].subplots(1, 4)
        _plot_quaternions(bot)

    if show:
        plt.show()
    return fig


def visualize_traj(traj: Union[Trajectory, npt.ArrayLike]) -> list[int]:
    """Visualizes a trajectory's sequence of poses on the Pybullet GUI

    Args:
        traj (Union[Trajectory, npt.ArrayLike]): Trajectory to visualize (must contain at least
            position + orientation info), or an array of position + quaternion poses, shape (n, 7)

    Returns:
        list[int]: Pybullet IDs for the lines drawn onto the GUI
    """
    if isinstance(traj, Trajectory):
        traj = traj.poses
    else:
        # If there is more information (velocity, time) in our array, only use the pose info
        traj = np.atleast_2d(traj)[:, :7]
    tmats = batched_pos_quats_to_tmats(traj)
    ids = []
    for i in range(tmats.shape[0]):
        ids += visualize_frame(tmats[i, :, :])
    return ids
