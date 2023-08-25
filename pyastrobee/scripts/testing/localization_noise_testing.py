"""Script to help visualize the effect of diferent localization noise values"""

import time

import pybullet
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.math_utils import spherical_vonmises_sampling
from pyastrobee.utils.rotations import quat_to_fixed_xyz
from pyastrobee.utils.debug_visualizer import visualize_points, remove_debug_objects


def generate_samples(
    pos_mean: npt.ArrayLike,
    pos_cov: np.ndarray,
    quat_mean: npt.ArrayLike,
    quat_kappa: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample postions and quaternions from a distribution about provided mean values

    Args:
        pos_mean (npt.ArrayLike): Mean position, shape (3,)
        pos_cov (np.ndarray): Position covariance matrix, shape (3, 3)
        quat_mean (npt.ArrayLike): Mean quaternion, shape (4,)
        quat_kappa (float): Concentration parameter of von Mises distribution
        n (int): Number of samples

    Returns:
        tuple[np.ndarray, np.ndarray]:
            np.ndarray: Sampled positions, shape (n, 3)
            np.ndarray: Sampled quaternions, shape (n, 4)
    """
    sampled_positions = np.random.multivariate_normal(pos_mean, pos_cov, n)
    sampled_quats = spherical_vonmises_sampling(quat_mean, quat_kappa, n)
    return sampled_positions, sampled_quats


def test_plot_distributions():
    """Quick test to sample different positions/orientations and plot their distributnions"""
    position_mean = np.array([1, 2, 3])
    position_covariance = np.diag([0.1, 0.2, 0.3])
    orientation_mean = np.array([0, 0, 0, 1])
    orientation_variance = 0.01
    orientation_kappa = 1 / orientation_variance  # Approx
    n_samples = 100
    sampled_positions, sampled_quats = generate_samples(
        position_mean,
        position_covariance,
        orientation_mean,
        orientation_kappa,
        n_samples,
    )
    eulers = np.array([quat_to_fixed_xyz(q) for q in sampled_quats])

    n_bins = 20
    fig = plt.figure()
    subfigs = fig.subfigures(3, 1)
    top = subfigs[0].subplots(1, 3)  # For positions
    mid = subfigs[1].subplots(1, 4)  # For quaternions
    bot = subfigs[2].subplots(1, 3)  # For euler angles

    pos_labels = ["x", "y", "z"]
    quat_labels = ["qx", "qy", "qz", "qw"]
    euler_labels = ["roll", "pitch", "yaw"]
    for i, ax in enumerate(top):
        ax.hist(sampled_positions[:, i], n_bins)
        ax.set_title(pos_labels[i])
    for i, ax in enumerate(mid):
        ax.hist(sampled_quats[:, i], n_bins)
        ax.set_title(quat_labels[i])
    for i, ax in enumerate(bot):
        ax.hist(eulers[:, i], n_bins)
        ax.set_title(euler_labels[i])
    plt.show()


def test_with_astrobee():
    """Quick test to show how the position/orientation of the Astrobee can vary under certain noise values"""
    pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
    robot = Astrobee()
    position_mean = robot.position
    position_covariance = np.diag([0.01, 0.02, 0.03])
    orientation_mean = robot.orientation
    orientation_variance = 0.01
    orientation_kappa = 1 / orientation_variance  # Approx
    n_samples = 100
    sampled_positions, sampled_quats = generate_samples(
        position_mean,
        position_covariance,
        orientation_mean,
        orientation_kappa,
        n_samples,
    )
    delay = 1 / 10  # To make it easier to see each sampled state
    input("Press Enter to visualize position + orientation noise together")
    for pos, orn in zip(sampled_positions, sampled_quats):
        robot.reset_to_base_pose([*pos, *orn])
        time.sleep(delay)
    input("Press Enter to visualize position noise alone")
    points = visualize_points(sampled_positions, [1, 0, 0], 5)
    for pos in sampled_positions:
        robot.reset_to_base_pose([*pos, *orientation_mean])
        time.sleep(delay)
    input("Press Enter to visualize orientation noise alone")
    remove_debug_objects(points)
    # TODO: add debug lines to visualize the pointing direction of the astrobee?
    # (Not a full description of orientation, but could be useful to see)
    for orn in sampled_quats:
        robot.reset_to_base_pose([*position_mean, *orn])
        time.sleep(delay)
    pybullet.disconnect()


if __name__ == "__main__":
    test_plot_distributions()
    test_with_astrobee()
