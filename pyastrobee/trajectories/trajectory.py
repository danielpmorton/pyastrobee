"""Trajectory class for Astrobee control, and trajectory-associated helpers"""

# TODO decide if trajectory attributes should be read-only? (properties)
# TODO make a subclass of Trajectory with the log() method
# TODO make it clearer what happens if some of the traj components are None
#      (Numerically calculate the gradient? Remove the option to set these as None?)
# TODO add max vel/accel lines into the plots?
# TODO remove the ability to not give time information... this makes no sense.

from typing import Optional, Union

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pyastrobee.utils.poses import batched_pos_quats_to_tmats
from pyastrobee.utils.debug_visualizer import visualize_frame, visualize_path
from pyastrobee.utils.quaternions import quaternion_dist, quats_to_angular_velocities
from pyastrobee.utils.boxes import Box


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
        self._num_timesteps = None  # Init

    @property
    def positions(self) -> np.ndarray:
        return np.atleast_2d(self._positions)

    @property
    def quaternions(self) -> np.ndarray:
        return np.atleast_2d(self._quats)

    @property
    def linear_velocities(self) -> np.ndarray:
        return np.atleast_2d(self._lin_vels)

    @property
    def angular_velocities(self) -> np.ndarray:
        return np.atleast_2d(self._ang_vels)

    @property
    def linear_accels(self) -> np.ndarray:
        return np.atleast_2d(self._lin_accels)

    @property
    def angular_accels(self) -> np.ndarray:
        return np.atleast_2d(self._ang_accels)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._times)

    @property
    def timestep(self) -> float | None:
        if np.size(self._times) == 0:
            return None
        return self._times[1] - self._times[0]

    @property
    def num_timesteps(self) -> int:
        if self._num_timesteps is None:
            if np.size(self.positions) > 0:
                self._num_timesteps = self.positions.shape[0]
            elif np.size(self.quaternions) > 0:
                self._num_timesteps = self.quaternions.shape[0]
        # If there is no position or orientation info, trajectory is empty (None)
        return self._num_timesteps

    @property
    def duration(self) -> float:
        return self._times[-1] - self._times[0]

    @property
    def poses(self) -> np.ndarray:
        """Pose array (position + xyzw quaternion), shape (n, 7)"""
        # if self._poses is not None:
        #     return self._poses  # Only calculate this once
        if self.positions.size == 0:
            raise ValueError("No position information available")
        if self.quaternions.size == 0:
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

    @property
    def contains_pos_only(self) -> bool:
        """Whether the trajectory contains only position info"""
        return self.positions.size > 0 and self.quaternions.size == 0

    @property
    def contains_orn_only(self) -> bool:
        """Whether the trajectory contains only orientation info"""
        return self.positions.size == 0 and self.quaternions.size > 0

    @property
    def contains_pos_and_orn(self) -> bool:
        """Whether the trajectory contains both position and orientation info"""
        return self.positions.size > 0 and self.quaternions.size > 0

    @property
    def is_empty(self) -> bool:
        """Whether the trajectory contains no position/orientation info"""
        return self.positions.size == 0 and self.quaternions.size == 0

    def visualize(
        self,
        n: Optional[int] = None,
        size: float = 0.5,
        client: Optional[BulletClient] = None,
    ) -> list[int]:
        """View the trajectory in Pybullet

        Args:
            n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
                Defaults to None (plot all frames)
            size (float, optional): Length of the lines to plot for each frame. Defaults to 0.5 (this gives a good scale
                with respect to the dimensions of the Astrobee)
            client (BulletClient, optional): If connecting to multiple physics servers, include the client
                (the class instance, not just the ID) here. Defaults to None (use default connected client)

        Returns:
            list[int]: Pybullet IDs for the lines drawn onto the GUI
        """
        client: pybullet = pybullet if client is None else client
        connection_status = client.isConnected()
        # Bring up the Pybullet GUI if needed
        if not connection_status:
            client.connect(pybullet.GUI)
        if self.contains_pos_and_orn:
            ids = visualize_traj(self, n, size, client=client)
        elif self.contains_pos_only:
            print("Trajectory only contains position info. Showing path instead")
            ids = visualize_path(self.positions, n, client=client)
        elif self.contains_orn_only:
            raise NotImplementedError(
                "Visualizing a sequence of purely orientations is not implemented yet"
            )
        else:  # Empty trajectory
            raise ValueError("No trajectory information to visualize")
        # Disconnect Pybullet if we originally weren't connected
        if not connection_status:
            input("Press Enter to disconnect Pybullet")
            client.disconnect()
        return ids

    def plot(self, show: bool = True) -> Figure:
        """Plot the trajectory components over time

        Args:
            show (bool, optional): Whether or not to display the plot. Defaults to True.

        Returns:
            Figure: Matplotlib figure containing the plots
        """
        return plot_traj(self, show=show)

    def get_segment(
        self, start_index: int, end_index: int, reset_time: bool = True
    ) -> "Trajectory":
        """Construct a trajectory segment from a larger trajectory

        Args:
            start_index (int): Starting index of the larger trajectory to extract the segment
            end_index (int): Ending index of the larger trajectory to extract the segment
            reset_time (bool): Whether to maintain the time association with the original trajectory,
                or reset the start time back to 0. Defaults to True (reset start time back to 0)

        Returns:
            Trajectory: A new trajectory representing a segment of the original trajectory
        """
        # TODO: add check for invalid slicing indices? Or just leave it up to numpy

        # Time needs to get handled differently because the trajectory may or may not have time info
        if np.size(self.times) == 0:  # No time info
            new_times = None
        else:
            new_times = self.times[start_index:end_index]
            if reset_time:
                new_times -= new_times[0]

        return Trajectory(
            self.positions[start_index:end_index],
            self.quaternions[start_index:end_index],
            self.linear_velocities[start_index:end_index],
            self.angular_velocities[start_index:end_index],
            self.linear_accels[start_index:end_index],
            self.angular_accels[start_index:end_index],
            new_times,
        )

    def get_segment_between_times(self, start_time, end_time, reset_time: bool = True):
        # start_index = np.searchsorted(self.times, start_time)
        # end_index = np.searchsorted(self.times, end_time)
        raise NotImplementedError("TODO")


class TrajectoryLogger(Trajectory):
    """Class for maintaining a history of a robot's state over time"""

    def __init__(self):
        # Create an empty Trajectory which we will iteratively append to
        super().__init__(None, None, None, None, None, None, None)

    def log_state(
        self,
        pos: npt.ArrayLike,
        quat: npt.ArrayLike,
        lin_vel: Optional[npt.ArrayLike] = None,
        ang_vel: Optional[npt.ArrayLike] = None,
        dt: Optional[float] = None,
    ):
        """Record the robot state at a given timestep

        If velocity information is not available (for instance, with a softbody),
        we can log just the position + orientation information

        Args:
            pos (npt.ArrayLike): Current position, shape (3,)
            quat (npt.ArrayLike): Current orientation (XYZW quaternion), shape (4,)
            lin_vel (Optional[npt.ArrayLike]): Current linear velocity, shape (3,). Defaults to None.
            ang_vel (Optional[npt.ArrayLike]): Current angular velocity, shape (3,). Defaults to None.
            dt (Optional[float]): Time elapsed since the previous step. Defaults to None.
        """
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

    def clear_log(self) -> None:
        """Clears the log of all trajectory data"""
        self._positions = []
        self._quats = []
        self._lin_vels = []
        self._ang_vels = []
        self._times = []


# TODO improve this... make it match up with the normal class
# TODO velocity/force control?
# TODO use a dictionary for the angles instead? map joint indices => their angles over time
# TODO should we always define the trajectory for all of the joints?
class ArmTrajectory:
    """Maintains information on the positions of the arm's joint angles over time

    Currently this assumes we are working with position control

    Args:
        angles (np.ndarray): Joint angles over time, shape (n_timesteps, n_controlled_joints)
        joint_ids (npt.ArrayLike): Indices of the joints being controlled with position control. Unspecified joints
            will be left at their default control value (Either manually set or defined at robot initialization)
        times (Optional[npt.ArrayLike]): Time information for the trajectory, shape (n_timesteps). Defaults to None.
        key_times (Optional[dict[str, float]]): Information about when key transitions occur in the trajectory,
            for instance, when we start/stop an arm deployment motion. Defaults to None.

    Raises:
        ValueError: If the shape of the angles input does not match the expected number of joints being controlled,
            or the number of timesteps in the time input
        KeyError: If the key_times dictionary contains unexpected keys. Currently supported: "begin_drag_motion",
            "end_drag_motion", "begin_grasp_motion", "end_grasp_motion"
    """

    def __init__(
        self,
        angles: np.ndarray,
        joint_ids: npt.ArrayLike,
        times: Optional[npt.ArrayLike] = None,
        key_times: Optional[dict[str, float]] = None,
    ):
        if np.ndim(angles) == 1:
            angles = angles.reshape(-1, 1)
        joint_ids = np.atleast_1d(joint_ids)
        n_steps, n_joints = angles.shape
        if n_joints != len(joint_ids):
            raise ValueError(
                "Mismatched inputs: The second dimension of the angles must match the number of joints.\n"
                + f"Got shape: {angles.shape} for {len(joint_ids)} joint indices: {joint_ids}"
            )
        if times is not None and len(times) != n_steps:
            raise ValueError(
                "Mismatched input dimensions: Time info is not the same length as the angles"
            )
        self.angles = angles
        self.joint_ids = joint_ids
        self.times = times if times is not None else []
        self.num_timesteps = n_steps
        # Hacky way to store information on the times/indices when the arm moves back or forward
        # This somewhat assumes that we're going to move the arm to "drag" and then back to "grasp"
        # TODO make this more general
        # TODO improve how we handle the keys that we'd expect to see
        self.key_times = key_times
        expected_keys = {
            "begin_drag_motion",
            "end_drag_motion",
            "begin_grasp_motion",
            "end_grasp_motion",
        }
        for key in self.key_times.keys():
            if key not in expected_keys:
                raise KeyError("Unexpected key time name: ", key)

    def plot(self):
        if self.times is None or np.size(self.times) == 0:
            x_axis = range(self.num_timesteps)
            x_label = "Timesteps"
        else:
            x_axis = self.times
            x_label = "Time, s"
        n_subplots = len(self.joint_ids)
        subplot_shape = num_subplots_to_shape(n_subplots)
        fig = plt.figure()
        for j, index in enumerate(self.joint_ids):
            plt.subplot(*subplot_shape, j + 1)
            plt.plot(x_axis, self.angles[:, j])
            plt.title(f"Joint {index}")
            plt.xlabel(x_label)
            plt.ylabel("Angle")
        plt.show()

    # TODO!! USE NEW CONCATENATE AND EXTRAPOLATE METHODS
    def get_segment(
        self, start_index: int, end_index: int, reset_time: bool = True
    ) -> "ArmTrajectory":
        """Construct an arm trajectory segment from a larger trajectory

        Args:
            start_index (int): Starting index of the larger trajectory to extract the segment
            end_index (int): Ending index of the larger trajectory to extract the segment
            reset_time (bool): Whether to maintain the time association with the original trajectory,
                or reset the start time back to 0. Defaults to True (reset start time back to 0)

        Returns:
            ArmTrajectory: A new trajectory representing a segment of the original trajectory
        """
        # TODO: add check for invalid slicing indices? Or just leave it up to numpy

        # HACK HACK HACK !!!!!!!!!!!!!!!!!!!!!!

        n_steps = len(self.times)
        # print(f"GETTING ARM SEGMENT: Start: {start_index}, End: {end_index}")
        if start_index < n_steps and end_index < n_steps:
            # NORMAL

            # Time needs to get handled differently because the trajectory may or may not have time info
            if np.size(self.times) == 0:  # No time info
                new_times = None
            else:
                new_times = self.times[start_index:end_index]

            new_key_times = {}
            for name, time in self.key_times.items():
                if new_times[0] <= time <= new_times[-1]:
                    new_key_times[name] = time

            if reset_time and new_times is not None:
                new_times -= new_times[0]
                for name, time in new_key_times.items():
                    new_key_times[name] -= new_times[0]

            return ArmTrajectory(
                self.angles[start_index:end_index, :],
                self.joint_ids,
                new_times,
                new_key_times,
            )
        elif start_index < n_steps and end_index >= n_steps:
            angles = np.vstack(
                [
                    self.angles[start_index:, :],
                    np.ones((end_index - n_steps, len(self.joint_ids)))
                    * self.angles[-1],
                ]
            )
            dt = self.times[1] - self.times[0]
            new_times = (
                self.times[start_index] + np.arange(end_index - start_index) * dt
            )
            new_key_times = {}
            for name, time in self.key_times.items():
                if new_times[0] <= time <= new_times[-1]:
                    new_key_times[name] = time
            if reset_time:
                new_times -= new_times[0]
                for name, time in new_key_times.items():
                    new_key_times[name] -= new_times[0]
            return ArmTrajectory(angles, self.joint_ids, new_times, new_key_times)
        else:  # Both over the limit
            angles = self.angles[-1] * np.ones(
                (end_index - start_index, len(self.joint_ids))
            )
            dt = self.times[1] - self.times[0]
            new_times = self.times[-1] + np.arange(end_index - start_index) * dt
            new_key_times = {}
            for name, time in self.key_times.items():
                if new_times[0] <= time <= new_times[-1]:
                    new_key_times[name] = time
            if reset_time:
                new_times -= new_times[0]
                for name, time in new_key_times.items():
                    new_key_times[name] -= new_times[0]
            return ArmTrajectory(angles, self.joint_ids, new_times, new_key_times)

    def get_segment_from_times(
        self, start_time: float, end_time: float, reset_time: bool = True
    ) -> "ArmTrajectory":
        """Construct an arm trajectory segment from a larger trajectory

        Args:
            start_time (float): Starting time of the segment
            end_time (float): Ending time of the segment
            reset_time (bool): Whether to maintain the time association with the original trajectory,
                or reset the start time back to 0. Defaults to True (reset start time back to 0)

        Returns:
            ArmTrajectory: A new trajectory representing a segment of the original trajectory
        """
        start_index = np.searchsorted(self.times, start_time)
        end_index = np.searchsorted(self.times, end_time)
        if end_index == len(self.times):  # Past the max time... somewhat of a HACK
            dt = self.times[-1] - self.times[-2]
            time_after = end_time - self.times[-1]
            end_index = len(self.times) - 1 + round(time_after / dt)
        return self.get_segment(start_index, end_index, reset_time)


# TODO see if we can incorporate a sequence of Boxes for the position constraints
# on a spline trajectory (rather than just a single Box constraint for a single curve)
def plot_traj_constraints(
    traj: Trajectory,
    pos_lims: Optional[Union[Box, npt.ArrayLike]] = None,
    max_vel: Optional[float] = None,
    max_accel: Optional[float] = None,
    max_omega: Optional[float] = None,
    max_alpha: Optional[float] = None,
    show: bool = True,
) -> Figure:
    """Plot trajectory info to visualize how it satisfies constraints

    Args:
        traj (Trajectory): Trajectory to plot
        pos_lims (Optional[Union[Box, npt.ArrayLike]]): Lower and upper limits on the XYZ position. Defaults to None.
        max_vel (Optional[float]): Maximum velocity magnitude. Defaults to None.
        max_accel (Optional[float]): Maximum acceleration magnitude. Defaults to None.
        max_omega (Optional[float]): Maximum angular velocity magnitude. Defaults to None.
        max_alpha (Optional[float]): Maximum angular acceleration magnitude. Defaults to None.
        show (bool, optional): Whether or not to display the plot. Defaults to True.

    Returns:
        Figure: The plot
    """
    fig = plt.figure()
    if traj.times is None or np.size(traj.times) == 0:
        x_axis = range(traj.num_timesteps)
        x_label = "Timesteps"
    else:
        x_axis = traj.times
        x_label = "Time, s"

    fmt = "k-"
    lim_fmt = "r--"
    # Top row is position info, bottom row is orientation info
    # Columns give derivative info
    subfigs = fig.subfigures(2, 3)
    # Position
    top_left = subfigs[0, 0].subplots(1, 3)
    if traj.positions.size > 0:
        for i, ax in enumerate(top_left):
            ax.plot(x_axis, traj.positions[:, i], fmt)
            ax.set_title(["x", "y", "z"][i])
            ax.set_xlabel(x_label)
        if pos_lims is not None:
            lower_pos_lim, upper_pos_lim = pos_lims
            for i, ax in enumerate(top_left):
                ax.plot(x_axis, lower_pos_lim[i] * np.ones_like(x_axis), lim_fmt)
                ax.plot(x_axis, upper_pos_lim[i] * np.ones_like(x_axis), lim_fmt)
    # Linear velocity
    if traj.linear_velocities.size > 0:
        top_middle = subfigs[0, 1].subplots(1, 1)
        top_middle.plot(x_axis, np.linalg.norm(traj.linear_velocities, axis=1), fmt)
        top_middle.set_title("||vel||")
        top_middle.set_xlabel(x_label)
        if max_vel is not None:
            top_middle.plot(x_axis, max_vel * np.ones_like(x_axis), lim_fmt)
    # Linear acceleration
    if traj.linear_accels.size > 0:
        top_right = subfigs[0, 2].subplots(1, 1)
        top_right.plot(x_axis, np.linalg.norm(traj.linear_accels, axis=1), fmt)
        top_right.set_title("||accel||")
        top_right.set_xlabel(x_label)
        if max_accel is not None:
            top_right.plot(x_axis, max_accel * np.ones_like(x_axis), lim_fmt)
    # Quaternions
    # These are unconstrained so it's the same plotting method as in the standard plot traj function
    bot_left = subfigs[1, 0].subplots(1, 4)
    if traj.quaternions.size > 0:
        _plot(
            bot_left, traj.quaternions, ["qx", "qy", "qz", "qw"], x_axis, x_label, fmt
        )
    # Angular velocity
    bot_middle = subfigs[1, 1].subplots(1, 1)
    if traj.angular_velocities.size > 0:
        bot_middle.plot(x_axis, np.linalg.norm(traj.angular_velocities, axis=1), fmt)
        bot_middle.set_title("||omega||")
        bot_middle.set_xlabel(x_label)
        if max_omega is not None:
            bot_middle.plot(x_axis, max_omega * np.ones_like(x_axis), lim_fmt)
    # Angular acceleration
    bot_right = subfigs[1, 2].subplots(1, 1)
    if traj.angular_accels.size > 0:
        bot_right.plot(x_axis, np.linalg.norm(traj.angular_accels, axis=1), fmt)
        bot_right.set_title("||alpha||")
        bot_right.set_xlabel(x_label)
        if max_alpha is not None:
            bot_right.plot(x_axis, max_alpha * np.ones_like(x_axis), lim_fmt)
    if show:
        plt.show()
    return fig


def plot_traj(traj: Trajectory, show: bool = True, fmt: str = "k-") -> Figure:
    """Plot the trajectory components over time

    Args:
        traj (Trajectory): The Trajectory object to plot
        show (bool, optional): Whether or not to display the plot. Defaults to True.
        fmt (str, optional): Matplotlib line specification. Defaults to "k-"

    Returns:
        Figure: Matplotlib figure containing the plots
    """

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
    if traj.times is None or np.size(traj.times) == 0:
        x_axis = range(traj.num_timesteps)
        x_label = "Timesteps"
    else:
        x_axis = traj.times
        x_label = "Time, s"
    # Top row is position info, bottom row is orientation info
    # Columns give derivative info
    subfigs = fig.subfigures(2, 3)
    top_left = subfigs[0, 0].subplots(1, 3)
    _plot(top_left, traj.positions, labels[POS], x_axis, x_label, fmt)
    top_middle = subfigs[0, 1].subplots(1, 3)
    _plot(top_middle, traj.linear_velocities, labels[LIN_VEL], x_axis, x_label, fmt)
    top_right = subfigs[0, 2].subplots(1, 3)
    _plot(top_right, traj.linear_accels, labels[LIN_ACCEL], x_axis, x_label, fmt)
    bot_left = subfigs[1, 0].subplots(1, 4)
    _plot(bot_left, traj.quaternions, labels[ORN], x_axis, x_label, fmt)
    bot_middle = subfigs[1, 1].subplots(1, 3)
    _plot(bot_middle, traj.angular_velocities, labels[ANG_VEL], x_axis, x_label, fmt)
    bot_right = subfigs[1, 2].subplots(1, 3)
    _plot(bot_right, traj.angular_accels, labels[ANG_ACCEL], x_axis, x_label, fmt)
    if show:
        plt.show()
    return fig


def _plot(
    axes: np.ndarray[plt.Axes],
    data: np.ndarray,
    labels: list[str],
    x_axis: np.ndarray,
    x_label: str,
    *args,
    **kwargs,
):
    """Helper function for plotting trajectory components

    Args:
        axes (np.ndarray[plt.Axes]): Matplotlib axes for subplots within a subfigure, length = n
        data (np.ndarray): Trajectory information to plot, shape (m, n) where m is the number of timesteps
            and n refers to the number of components of that trajectory info (for instance, position has
            data for x, y, and z, so n = 3)
        labels (list[str]): Labels for each of the components of the trajectory data, length = n
        x_axis (np.ndarray): X-axis data to plot the trajectory against, length = m
        x_label (str): Label for the x-axis (for instance, "Time" or "Steps")
    """
    # If the trajectory doesn't contain the info, don't plot it
    if np.size(data) == 0 or data is None:
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


# TODO all of the plotting logic here is extremely similar to the single-plot method
# Figure out a way to simplify the code
def compare_trajs(
    traj_1: Trajectory,
    traj_2: Trajectory,
    show: bool = True,
    fmt_1: str = "k-",
    fmt_2: str = "b-",
) -> Figure:
    """Compares two trajectories by plotting them on the same axes

    Args:
        traj_1 (Trajectory): First trajectory to plot
        traj_2 (Trajectory): Second trajectory to plot
        show (bool, optional): . Defaults to True.
        fmt_1 (str, optional): Matplotlib line specification for the first traj. Defaults to "k-".
        fmt_2 (str, optional): Matplotlib line specification for the second traj. Defaults to "b-".

    Returns:
        Figure: Matplotlib figure containing the plots
    """
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
    # TODO this check is kinda weird right now
    # what happens if one has time info and the other doesn't??
    if traj_1.times is None or np.size(traj_1.times) == 0:
        x_axis_1 = range(traj_1.num_timesteps)
        x_label = "Timesteps"
    else:
        x_axis_1 = traj_1.times
        x_label = "Time, s"
    if traj_2.times is None or np.size(traj_2.times) == 0:
        x_axis_2 = range(traj_2.num_timesteps)
    else:
        x_axis_2 = traj_2.times
    # Top row is position info, bottom row is orientation info
    # Columns give derivative info
    # fmt: off
    subfigs = fig.subfigures(2, 3)
    top_left = subfigs[0, 0].subplots(1, 3)
    _plot(top_left, traj_1.positions, labels[POS], x_axis_1, x_label, fmt_1)
    _plot(top_left, traj_2.positions, labels[POS], x_axis_2, x_label, fmt_2)
    top_middle = subfigs[0, 1].subplots(1, 3)
    _plot(top_middle, traj_1.linear_velocities, labels[LIN_VEL], x_axis_1, x_label, fmt_1)
    _plot(top_middle, traj_2.linear_velocities, labels[LIN_VEL], x_axis_2, x_label, fmt_2)
    top_right = subfigs[0, 2].subplots(1, 3)
    _plot(top_right, traj_1.linear_accels, labels[LIN_ACCEL], x_axis_1, x_label, fmt_1)
    _plot(top_right, traj_2.linear_accels, labels[LIN_ACCEL], x_axis_2, x_label, fmt_2)
    bot_left = subfigs[1, 0].subplots(1, 4)
    _plot(bot_left, traj_1.quaternions, labels[ORN], x_axis_1, x_label, fmt_1)
    _plot(bot_left, traj_2.quaternions, labels[ORN], x_axis_2, x_label, fmt_2)
    bot_mid = subfigs[1, 1].subplots(1, 3)
    _plot(bot_mid, traj_1.angular_velocities, labels[ANG_VEL], x_axis_1, x_label, fmt_1)
    _plot(bot_mid, traj_2.angular_velocities, labels[ANG_VEL], x_axis_2, x_label, fmt_2)
    bot_right = subfigs[1, 2].subplots(1, 3)
    _plot(bot_right, traj_1.angular_accels, labels[ANG_ACCEL], x_axis_1, x_label, fmt_1)
    _plot(bot_right, traj_2.angular_accels, labels[ANG_ACCEL], x_axis_2, x_label, fmt_2)
    # fmt: on
    if show:
        plt.show()
    return fig


def visualize_traj(
    traj: Union[Trajectory, npt.ArrayLike],
    n: Optional[int] = None,
    size: float = 0.5,
    client: Optional[BulletClient] = None,
) -> list[int]:
    """Visualizes a trajectory's sequence of poses on the Pybullet GUI

    Args:
        traj (Union[Trajectory, npt.ArrayLike]): Trajectory to visualize (must contain at least
            position + orientation info), or an array of position + quaternion poses, shape (n, 7)
        n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
            Defaults to None (plot all frames)
        size (float, optional): Length of the lines to plot for each frame. Defaults to 0.5 (this gives a good scale
            with respect to the dimensions of the Astrobee)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        list[int]: Pybullet IDs for the lines drawn onto the GUI
    """
    client: pybullet = pybullet if client is None else client
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
        ids += visualize_frame(tmats[i, :, :], size, client=client)
    return ids


# TODO Move this to a more dynamics-relevant location?
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


def merge_trajs(pos_traj: Trajectory, orn_traj: Trajectory) -> Trajectory:
    """Merge the position component from one trajectory with the orientation component from another

    Args:
        pos_traj (Trajectory): Trajectory with desired position component
        orn_traj (Trajectory): Trajectory with desired orientation component

    Returns:
        Trajectory: Merged trajectory with both position + orientation info
    """
    assert np.size(pos_traj.positions) > 0
    assert np.size(orn_traj.quaternions) > 0
    positions = pos_traj.positions
    quaternions = pos_traj.quaternions
    # Time
    # If both trajectories have time info, make sure they match up
    if np.size(pos_traj.times) > 0:
        if np.size(orn_traj.times) > 0:
            if not np.allclose(pos_traj.times, orn_traj.times):
                raise ValueError("Mismatched time information")
        times = pos_traj.times
        dt = pos_traj.timestep
    elif np.size(orn_traj.times) > 0:
        times = orn_traj.times
        dt = orn_traj.timestep
    else:
        times = None
        dt = None
    # If either trajectory is missing derivative info, compute it if we can
    # Position
    if np.size(pos_traj.linear_velocities) == 0 and dt is not None:
        lin_vels = np.gradient(pos_traj.positions, dt, axis=0)
    else:
        lin_vels = pos_traj.linear_velocities
    if np.size(pos_traj.linear_accels) == 0 and dt is not None:
        lin_accels = np.gradient(lin_vels, dt, axis=0)
    else:
        lin_accels = pos_traj.linear_accels
    # Orientation
    if np.size(orn_traj.angular_velocities) == 0 and dt is not None:
        ang_vels = quats_to_angular_velocities(orn_traj.quaternions, dt)
    else:
        ang_vels = orn_traj.angular_velocities
    if np.size(orn_traj.angular_accels) == 0 and dt is not None:
        ang_accels = np.gradient(ang_vels, dt, axis=0)
    else:
        ang_accels = orn_traj.angular_accels

    return Trajectory(
        positions, quaternions, lin_vels, ang_vels, lin_accels, ang_accels, times
    )


def concatenate_trajs(traj_1: Trajectory, traj_2: Trajectory) -> Trajectory:
    """Combine two trajectories one after the other

    This will follow the first trajectory until its end, then follow the second one until it ends

    Args:
        traj_1 (Trajectory): First trajectory
        traj_2 (Trajectory): Second trajectory

    Returns:
        Trajectory: Combined trajectory
    """
    # Ensure continuity in time
    # TODO check for continuity in all other components?
    # TODO this assumes that both have time information
    dt = traj_1.times[-1] - traj_1.times[-2]
    if np.isclose(traj_2.times[0], 0):
        times = np.concatenate([traj_1.times, traj_2.times + traj_1.times[-1] + dt])
    elif np.isclose(traj_2.times[0], traj_1.times[-1] + dt):
        times = np.concatenate([traj_1.times, traj_2.times])
    return Trajectory(
        np.vstack([traj_1.positions, traj_2.positions]),
        np.vstack(
            [traj_1.quaternions, traj_2.quaternions],
        ),
        np.vstack([traj_1.linear_velocities, traj_2.linear_velocities]),
        np.vstack([traj_1.angular_velocities, traj_2.angular_velocities]),
        np.vstack([traj_1.linear_accels, traj_2.linear_accels]),
        np.vstack([traj_1.angular_accels, traj_2.angular_accels]),
        times,
    )


def concatenate_arm_trajs(
    traj_1: ArmTrajectory, traj_2: ArmTrajectory
) -> ArmTrajectory:
    """Combine two arm trajectories one after the other

    This will follow the first trajectory until its end, then follow the second one until it ends

    Args:
        traj_1 (ArmTrajectory): First trajectory
        traj_2 (ArmTrajectory): Second trajectory

    Returns:
        ArmTrajectory: Combined trajectory
    """
    if traj_1.joint_ids != traj_2.joint_ids:
        raise NotImplementedError(
            "Controlled joint indices must match between the two trajectories (for now)"
        )
    # Ensure continuity in time, and merge the key time information
    # TODO check for continuity in all other components?
    # TODO this assumes that both have time information
    dt = traj_1.times[-1] - traj_1.times[-2]
    new_key_times = traj_1.key_times
    if np.isclose(traj_2.times[0], 0):
        times = np.concatenate([traj_1.times, traj_2.times + traj_1.times[-1] + dt])
        for name, time in traj_2.key_times.items():
            new_key_times[name] = time + traj_1.times[-1] + dt
    elif np.isclose(traj_2.times[0], traj_1.times[-1] + dt):
        times = np.concatenate([traj_1.times, traj_2.times])
        new_key_times |= traj_2.key_times
    return ArmTrajectory(
        np.vstack([traj_1.angles, traj_2.angles]),
        traj_1.joint_ids,
        times,
        new_key_times,
    )


def extrapolate_traj(
    traj: Trajectory, duration: Optional[float] = None, n_steps: Optional[int] = None
) -> Trajectory:
    """Extrapolates a trajectory by an amount of time

    This assumes that the (pre-extrapolation) trajectory comes to a rest at the end

    Either the extrapolation duration or number of timesteps must be specified. Not both, not neither

    Args:
        traj (Trajectory): Original trajectory
        duration (Optional[float], optional): Amount of time to extrapolate by. Defaults to None
            (n_steps input must be provided)
        n_steps (Optional[int], optional): Number of timesteps to extrapolate by. Defaults to None
            (duration input must be provided)

    Returns:
        Trajectory: Extrapolated trajectory
    """
    dt = traj.times[-1] - traj.times[-2]
    if duration is None and n_steps is None:
        raise ValueError(
            "Must provide either a duration to extrapolate by, or a fixed number of timesteps"
        )
    elif duration is not None and n_steps is not None:
        raise ValueError("Provide either a duration or a number of timesteps, not both")
    elif duration is not None:
        n_steps = round(duration / dt)
    assert n_steps is not None
    if not np.isclose(np.linalg.norm(traj.linear_velocities[-1]), 0) or not np.isclose(
        np.linalg.norm(traj.angular_velocities[-1]), 0
    ):
        raise NotImplementedError(
            "Cannot extrapolate the trajectory: Only implemented for trajectories that end at a stopped pose"
        )
    return Trajectory(
        np.vstack([traj.positions, traj.positions[-1] * np.ones((n_steps, 1))]),
        np.vstack([traj.quaternions, traj.quaternions[-1] * np.ones((n_steps, 1))]),
        np.vstack([traj.linear_velocities, np.zeros((n_steps, 3))]),
        np.vstack([traj.angular_velocities, np.zeros((n_steps, 3))]),
        np.vstack([traj.linear_accels, np.zeros((n_steps, 3))]),
        np.vstack([traj.angular_accels, np.zeros((n_steps, 3))]),
        np.concatenate([traj.times, traj.times[-1] + dt + np.arange(n_steps) * dt]),
    )


def extrapolate_arm_traj(
    traj: ArmTrajectory, duration: Optional[float] = None, n_steps: Optional[int] = None
) -> ArmTrajectory:
    """Extrapolates an arm trajectory by an amount of time

    This assumes that the (pre-extrapolation) trajectory comes to a rest at the end

    Either the extrapolation duration or number of timesteps must be specified. Not both, not neither

    Args:
        traj (ArmTrajectory): Original arm trajectory
        duration (Optional[float], optional): Amount of time to extrapolate by. Defaults to None
            (n_steps input must be provided)
        n_steps (Optional[int], optional): Number of timesteps to extrapolate by. Defaults to None
            (duration input must be provided)

    Returns:
        ArmTrajectory: Extrapolated arm trajectory
    """
    dt = traj.times[-1] - traj.times[-2]
    if duration is None and n_steps is None:
        raise ValueError(
            "Must provide either a duration to extrapolate by, or a fixed number of timesteps"
        )
    elif duration is not None and n_steps is not None:
        raise ValueError("Provide either a duration or a number of timesteps, not both")
    elif duration is not None:
        n_steps = round(duration / dt)
    assert n_steps is not None
    return ArmTrajectory(
        np.vstack([traj.angles, traj.angles[-1] * np.ones((n_steps, 1))]),
        traj.joint_ids,
        np.concatenate([traj.times, traj.times[-1] + dt + np.arange(n_steps) * dt]),
        traj.key_times,
    )


def num_subplots_to_shape(n: int) -> tuple[int, int]:
    """Determines the best layout of a number of subplots within a larger figure

    Args:
        n (int): Number of subplots

    Returns:
        tuple[int, int]: Number of rows and columns for the subplot divisions
    """
    n_rows = int(np.sqrt(n))
    n_cols = n // n_rows + (n % n_rows > 0)
    assert n_rows * n_cols >= n
    return (n_rows, n_cols)
