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
from pyastrobee.utils.quaternions import quaternion_dist


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
        self._positions = positions if positions is not None else []
        self._quats = quats if quats is not None else []
        self._lin_vels = lin_vels if lin_vels is not None else []
        self._ang_vels = ang_vels if ang_vels is not None else []
        self._lin_accels = lin_accels if lin_accels is not None else []
        self._ang_accels = ang_accels if ang_accels is not None else []
        self._times = times if times is not None else []
        self._poses = None  # Init
        self._tmats = None  # Init

    @property
    def positions(self):
        return np.atleast_2d(self._positions)

    @property
    def quaternions(self):
        return np.atleast_2d(self._quats)

    @property
    def linear_velocities(self):
        return np.atleast_2d(self._lin_vels)

    @property
    def angular_velocities(self):
        return np.atleast_2d(self._ang_vels)

    @property
    def linear_accels(self):
        return np.atleast_2d(self._lin_accels)

    @property
    def angular_accels(self):
        return np.atleast_2d(self._ang_accels)

    @property
    def times(self):
        return np.asarray(self._times)

    @property
    def num_timesteps(self):
        return self.positions.shape[0]

    @property
    def poses(self) -> np.ndarray:
        """Pose array (position + xyzw quaternion), shape (n, 7)"""
        # if self._poses is not None:
        #     return self._poses  # Only calculate this once
        if self._positions is None:
            raise ValueError("No position information available")
        if self._quats is None:
            raise ValueError("No orientation information available")
        self._poses = np.column_stack([self.positions, self.quaternions])
        return self._poses

    @property
    def tmats(self) -> np.ndarray:
        """Poses expressed as transformation matrices, shape (n, 4, 4)"""
        # if self._tmats is not None:
        #     return self._tmats  # Only calculate this once
        self._tmats = batched_pos_quats_to_tmats(self.poses)
        return self._tmats

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
        return plot_traj(self, show=show)

    def log_state(
        self,
        pos: npt.ArrayLike,
        quat: npt.ArrayLike = None,
        lin_vel: Optional[npt.ArrayLike] = None,
        ang_vel: Optional[npt.ArrayLike] = None,
        dt: Optional[float] = None,
    ):
        # TODO refine this functionality
        # Can log position/orientation info without velocity if needed
        # These values can be None because matplotlib will just not plot them
        # Most importantly, we want to make sure that things correspond in time
        # (There shouldn't be an instance where we have histories of different lengths)
        self._positions.append(pos)
        self._quats.append(quat)
        self._lin_vels.append(lin_vel)
        self._ang_vels.append(ang_vel)
        if dt is not None and len(self._times) == 0:
            self._times.append(0.0)
        elif dt is not None:
            self._times.append(self._times[-1] + dt)


def plot_traj(traj: Trajectory, show: bool = True) -> Figure:
    """Plot the trajectory components over time

    Args:
        traj (Trajectory): The Trajectory object to plot
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
        if data == [] or data is None:
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


# NEEDS CLEANUP - integrate into main plotting function
def _plot_data(
    axes: np.ndarray[plt.Axes], data, labels, x_axis, x_label, *args, **kwargs
):
    """Helper function for plotting trajectory components

    Args:
        axes (np.ndarray[plt.Axes]): Matplotlib axes for subplots within a subfigure
        component (int): Type of trajectory information to plot. One of POSITION, ORIENTATION, VELOCITY, ...
            (see the indexing helper variables)
    """
    # If the trajectory doesn't contain the info, don't plot it
    if data == [] or data is None:
        return
    # Number of components to plot (for instance, position: n = 3: x, y, z)
    n = data.shape[1]
    assert n == len(labels)
    assert n == len(axes)
    # Plot each component of the trajectory information on a separate axis
    for i, ax in enumerate(axes):
        ax.plot(x_axis, data[:, i], *args, **kwargs)
        ax.set_title(labels[i])
        ax.set_xlabel(x_label)


# NEEDS SIGNIFICANT CLEANUP - should be totally eliminated or dramatically simplified
# Merge this with the main trajectory plotting function
def compare_trajs(traj_1: Trajectory, traj_2: Trajectory, show: bool = True):
    # Indexing helper variables
    POS = 0
    ORN = 1
    LIN_VEL = 2
    ANG_VEL = 3
    LIN_ACCEL = 4
    ANG_ACCEL = 5

    labels = {
        POS: ["x", "y", "z"],
        ORN: ["qx", "qy", "qz", "qw"],
        LIN_VEL: ["vx", "vy", "vz"],
        ANG_VEL: ["wx", "wy", "wz"],
        LIN_ACCEL: ["ax", "ay", "az"],
        ANG_ACCEL: ["al_x", "al_y", "al_z"],
    }

    fig = plt.figure()
    if traj_1.times is not None:
        x_axis = traj_1.times
        x_label = "Time, s"
    else:
        x_axis = range(traj_1.num_timesteps)
        x_label = "Timesteps"
    # Large plot if we have up second-derivative information
    if traj_1.linear_accels is not None or traj_1.angular_accels is not None:
        subfigs = fig.subfigures(2, 3)
        top_left = subfigs[0, 0].subplots(1, 3)
        _plot_data(top_left, traj_1.positions, labels[POS], x_axis, x_label, "k-")
        _plot_data(top_left, traj_2.positions, labels[POS], x_axis, x_label, "b-")
        top_middle = subfigs[0, 1].subplots(1, 3)
        _plot_data(
            top_middle, traj_1.linear_velocities, labels[LIN_VEL], x_axis, x_label, "k-"
        )
        _plot_data(
            top_middle, traj_2.linear_velocities, labels[LIN_VEL], x_axis, x_label, "b-"
        )
        top_right = subfigs[0, 2].subplots(1, 3)
        _plot_data(
            top_right, traj_1.linear_accels, labels[LIN_ACCEL], x_axis, x_label, "k-"
        )
        # TODO ADD A BETTER CHECK
        # _plot_data(
        #     top_right, traj_2.linear_accels, labels[LIN_ACCEL], x_axis, x_label, "b-"
        # )
        bot_left = subfigs[1, 0].subplots(1, 4)
        _plot_data(bot_left, traj_1.quaternions, labels[ORN], x_axis, x_label, "k-")
        _plot_data(bot_left, traj_2.quaternions, labels[ORN], x_axis, x_label, "b-")
        bot_mid = subfigs[1, 1].subplots(1, 3)
        _plot_data(
            bot_mid, traj_1.angular_velocities, labels[ANG_VEL], x_axis, x_label, "k-"
        )
        _plot_data(
            bot_mid, traj_2.angular_velocities, labels[ANG_VEL], x_axis, x_label, "b-"
        )
        bot_right = subfigs[1, 2].subplots(1, 3)
        _plot_data(
            bot_right, traj_1.angular_accels, labels[ANG_ACCEL], x_axis, x_label, "k-"
        )
        # TODO ADD A BETTER CHECK
        # _plot_data(
        #     bot_right, traj_2.angular_accels, labels[ANG_ACCEL], x_axis, x_label, "b-"
        # )
    # Medium size plot if we only have first-derivative info
    elif traj_1.linear_velocities is not None or traj_1.angular_velocities is not None:
        subfigs = fig.subfigures(2, 2)
        top_left = subfigs[0, 0].subplots(1, 3)
        _plot_data(top_left, traj_1.positions, labels[POS], x_axis, x_label, "k-")
        _plot_data(top_left, traj_2.positions, labels[POS], x_axis, x_label, "b-")
        top_right = subfigs[0, 1].subplots(1, 3)
        _plot_data(
            top_right, traj_1.linear_velocities, labels[LIN_VEL], x_axis, x_label, "k-"
        )
        _plot_data(
            top_right, traj_2.linear_velocities, labels[LIN_VEL], x_axis, x_label, "b-"
        )
        bot_left = subfigs[1, 0].subplots(1, 4)
        _plot_data(bot_left, traj_1.quaternions, labels[ORN], x_axis, x_label, "k-")
        _plot_data(bot_left, traj_2.quaternions, labels[ORN], x_axis, x_label, "b-")
        bot_right = subfigs[1, 1].subplots(1, 3)
        _plot_data(
            bot_right, traj_1.angular_velocities, labels[ANG_VEL], x_axis, x_label, "k-"
        )
        _plot_data(
            bot_right, traj_2.angular_velocities, labels[ANG_VEL], x_axis, x_label, "b-"
        )
    # Small plot for just plotting position + orientation
    else:
        subfigs = fig.subfigures(2, 1)
        top = subfigs[0].subplots(1, 3)
        _plot_data(top, traj_1.positions, labels[POS], x_axis, x_label, "k-")
        _plot_data(top, traj_2.positions, labels[POS], x_axis, x_label, "b-")
        bot = subfigs[1].subplots(1, 4)
        _plot_data(bot, traj_1.quaternions, labels[ORN], x_axis, x_label, "k-")
        _plot_data(bot, traj_2.quaternions, labels[ORN], x_axis, x_label, "b-")

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


def stopping_criteria(
    pos: npt.ArrayLike,
    quat: npt.ArrayLike,
    lin_vel: npt.ArrayLike,
    ang_vel: npt.ArrayLike,
    pos_des: npt.ArrayLike,
    quat_des: npt.ArrayLike,
    dp: float = 1e-2,
    dq: float = 1e-2,
    dv: float = 1e-2,
    dw: float = 5e-3,
) -> bool:
    """Determine if the Astrobee has fully stopped, based on its current dynamics state

    Args:
        pos (npt.ArrayLike): Current position, shape (3,)
        quat (npt.ArrayLike): Current XYZW quaternion orientation, shape (4,)
        lin_vel (npt.ArrayLike): Current linear velocity, shape (3,)
        ang_vel (npt.ArrayLike): Current angular velocity, shape (3,)
        pos_des (npt.ArrayLike): Desired position, shape (3,)
        quat_des (npt.ArrayLike): Desired XYZW quaternion orientation, shape (4,)
        dp (float, optional): Tolerance on position error magnitude. Defaults to 1e-2.
        dq (float, optional): Tolerance on quaternion distance between cur/des. Defaults to 1e-2.
        dv (float, optional): Tolerance on linear velocity error magnitude. Defaults to 1e-2.
        dw (float, optional): Tolerance on angular velocity error magnitude. Defaults to 5e-3.

    Returns:
        bool: If the Astrobee have successfully stopped at its desired goal pose
    """
    p_check = np.linalg.norm(pos - pos_des) <= dp
    q_check = quaternion_dist(quat, quat_des) <= dq
    v_check = np.linalg.norm(lin_vel) <= dv
    w_check = np.linalg.norm(ang_vel) <= dw
    checks = [p_check, q_check, v_check, w_check]
    return np.all(checks)
