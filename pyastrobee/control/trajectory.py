"""Trajectory class for Astrobee control, and trajectory-associated helpers

TODO decide if trajectory attributes should be read-only? (properties)
"""

from typing import Optional, Union

import pybullet
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyastrobee.utils.poses import batched_pos_quats_to_tmats
from pyastrobee.utils.debug_visualizer import visualize_frame


class Trajectory:
    """Trajectory class: Keeps track of a sequence of poses/velocities/accels over a period of time

    - All arguments can be omitted as needed (for instance, a pose-only trajectory without velocities,
        or a trajectory without time information)
    - All positions/orientations/velocities... are assumed to be defined in world frame

    Args:
        positions (Optional[npt.ArrayLike]): Sequence of XYZ positions, shape (n, 3)
        quats (Optional[npt.ArrayLike]): Sequence of XYZW quaternions, shape (n, 4)
        lin_vels (Optional[npt.ArrayLike]): Sequence of (vx, vy, vz) linear velocities, shape (n, 3)
        ang_vels (Optional[npt.ArrayLike]): Sequence of (wx, wy, wz) angular velocities, shape (n, 3)
        lin_accels (Optional[npt.ArrayLike]): Sequence of (ax, ay, az) linear accelerations, shape (n, 3)
        ang_accels (Optional[npt.ArrayLike]): Sequence of (al_x, al_y, al_z) angular accelerations, shape (n, 3)
        times (Optional[npt.ArrayLike]): Times corresponding to each trajectory entry, shape (n)
    """

    def __init__(
        self,
        positions: Optional[npt.ArrayLike] = None,
        quats: Optional[npt.ArrayLike] = None,
        lin_vels: Optional[npt.ArrayLike] = None,
        ang_vels: Optional[npt.ArrayLike] = None,
        lin_accels: Optional[npt.ArrayLike] = None,
        ang_accels: Optional[npt.ArrayLike] = None,
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
        self.linear_accels = lin_accels if lin_accels is not None else None
        self.angular_accels = ang_accels if ang_accels is not None else None
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

    # Indexing helper variables
    POSITION = 0
    ORIENTATION = 1
    VELOCITY = 2
    ANGULAR_VELOCITY = 3
    ACCELERATION = 4
    ANGULAR_ACCEL = 5

    traj_data = {
        POSITION: traj.positions,
        ORIENTATION: traj.quaternions,
        VELOCITY: traj.linear_velocities,
        ANGULAR_VELOCITY: traj.angular_velocities,
        ACCELERATION: traj.linear_accels,
        ANGULAR_ACCEL: traj.angular_accels,
    }

    traj_labels = {
        POSITION: ["x", "y", "z"],
        ORIENTATION: ["qx", "qy", "qz", "qw"],
        VELOCITY: ["vx", "vy", "vz"],
        ANGULAR_VELOCITY: ["wx", "wy", "wz"],
        ACCELERATION: ["ax", "ay", "az"],
        ANGULAR_ACCEL: ["al_x", "al_y", "al_z"],
    }

    def _plot(axes: np.ndarray[plt.Axes], component: int):
        """Helper sub-function for plotting trajectory components

        Args:
            axes (np.ndarray[plt.Axes]): Matplotlib axes for subplots within a subfigure
            component (int): Type of trajectory information to plot. One of POSITION, ORIENTATION, VELOCITY, ...
                (see the indexing helper variables)
        """
        # Extract trajectory info and the relevant labels for plotting
        data = traj_data[component]
        labels = traj_labels[component]
        # If the trajectory doesn't contain the info, don't plot it
        if data is None:
            return
        # Set the x axis based on if we have time information in the trajectory
        if traj.times is not None:
            x_axis = traj.times
            x_label = "Time, s"
        else:
            x_axis = range(traj.num_timesteps)
            x_label = "Timesteps"
        # Number of components to plot (for instance, position: n = 3: x, y, z)
        n = data.shape[1]
        assert n == len(labels)
        assert n == len(axes)
        # Plot each component of the trajectory information on a separate axis
        for i, ax in enumerate(axes):
            ax.plot(x_axis, data[:, i])
            ax.set_title(labels[i])
            ax.set_xlabel(x_label)

    fig = plt.figure()
    # Large plot if we have up second-derivative information
    if traj.linear_accels is not None or traj.angular_accels is not None:
        subfigs = fig.subfigures(2, 3)
        top_left = subfigs[0, 0].subplots(1, 3)
        _plot(top_left, POSITION)
        top_middle = subfigs[0, 1].subplots(1, 3)
        _plot(top_middle, VELOCITY)
        top_right = subfigs[0, 2].subplots(1, 3)
        _plot(top_right, ACCELERATION)
        bot_left = subfigs[1, 0].subplots(1, 4)
        _plot(bot_left, ORIENTATION)
        bot_middle = subfigs[1, 1].subplots(1, 3)
        _plot(bot_middle, ANGULAR_VELOCITY)
        bot_right = subfigs[1, 2].subplots(1, 3)
        _plot(bot_right, ANGULAR_ACCEL)
    # Medium size plot if we only have first-derivative info
    elif traj.linear_velocities is not None or traj.angular_velocities is not None:
        subfigs = fig.subfigures(2, 2)
        top_left = subfigs[0, 0].subplots(1, 3)
        _plot(top_left, POSITION)
        top_right = subfigs[0, 1].subplots(1, 3)
        _plot(top_right, VELOCITY)
        bot_left = subfigs[1, 0].subplots(1, 4)
        _plot(bot_left, ORIENTATION)
        bot_right = subfigs[1, 1].subplots(1, 3)
        _plot(bot_right, ANGULAR_VELOCITY)
    # Small plot for just plotting position + orientation
    else:
        subfigs = fig.subfigures(2, 1)
        top = subfigs[0].subplots(1, 3)
        _plot(top, POSITION)
        bot = subfigs[1].subplots(1, 4)
        _plot(bot, ORIENTATION)

    if show:
        plt.show()
    return fig


def visualize_traj(
    traj: Union[Trajectory, npt.ArrayLike], n: Optional[int] = None
) -> list[int]:
    """Visualizes a trajectory's sequence of poses on the Pybullet GUI

    Args:
        traj (Union[Trajectory, npt.ArrayLike]): Trajectory to visualize (must contain at least
            position + orientation info), or an array of position + quaternion poses, shape (n, 7)
        n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
            Defaults to None (plot all frames)

    Returns:
        list[int]: Pybullet IDs for the lines drawn onto the GUI
    """
    if isinstance(traj, Trajectory):
        traj = traj.poses
    else:
        # If there is more information (velocity, time) in our array, only use the pose info
        traj = np.atleast_2d(traj)[:, :7]
    n_frames = traj.shape[0]
    # If desired, sample frames evenly across the trajectory to plot a subset
    if n is not None and n < n_frames:
        # This indexing ensures that the first and last frames are plotted
        idx = np.round(np.linspace(0, n_frames - 1, n, endpoint=True)).astype(int)
        traj = traj[idx, :]
    tmats = batched_pos_quats_to_tmats(traj)
    ids = []
    for i in range(tmats.shape[0]):
        ids += visualize_frame(tmats[i, :, :])
    return ids
