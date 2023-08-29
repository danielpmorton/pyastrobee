"""Methods for defining rewards / penalties based on how well a trajectory is tracked

Currently, we just compare terminal states in the penalty function and don't consider integrated tracking error
(TODO make this a separate method)
"""

from typing import Union, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colors

from pyastrobee.utils.boxes import Box, plot_2D_box, is_in_box
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.quaternions import quaternion_angular_error


def deviation_penalty(
    cur_pos: npt.ArrayLike,
    cur_orn: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    cur_ang_vel: npt.ArrayLike,
    des_pos: npt.ArrayLike,
    des_orn: npt.ArrayLike,
    des_vel: npt.ArrayLike,
    des_ang_vel: npt.ArrayLike,
    pos_penalty: float,
    orn_penalty: float,
    vel_penalty: float,
    ang_vel_penalty: float,
) -> float:
    """Evaluate a loss/penalty for deviations between the current and desired dynamics state

    Args:
        cur_pos (npt.ArrayLike): Current position, shape (3,)
        cur_orn (npt.ArrayLike): Current XYZW quaternion orientation, shape (4,)
        cur_vel (npt.ArrayLike): Current linear velocity, shape (3,)
        cur_ang_vel (npt.ArrayLike): Current angular velocity, shape (3,)
        des_pos (npt.ArrayLike): Desired position, shape (3,)
        des_orn (npt.ArrayLike): Desired XYZW quaternion orientation, shape (4,)
        des_vel (npt.ArrayLike): Desired linear velocity, shape (3,)
        des_ang_vel (npt.ArrayLike): Desired angular velocity, shape (3,)
        pos_penalty (float): Penalty scaling factor for position error
        orn_penalty (float): Penalty scaling factor for orientation/angular error
        vel_penalty (float): Penalty scaling factor for linear velocity error
        ang_vel_penalty (float): Penalty scaling factor for angular velocity error

    Returns:
        float: Loss/penalty value
    """
    pos_err = np.subtract(cur_pos, des_pos)
    orn_err = quaternion_angular_error(cur_orn, des_orn)
    vel_err = np.subtract(cur_vel, des_vel)
    ang_vel_err = np.subtract(cur_ang_vel, des_ang_vel)
    # NOTE Would an L1 norm be better here?
    return (
        pos_penalty * np.linalg.norm(pos_err)
        + orn_penalty * np.linalg.norm(orn_err)
        + vel_penalty * np.linalg.norm(vel_err)
        + ang_vel_penalty * np.linalg.norm(ang_vel_err)
    )


def integrated_deviation(
    des_traj: Trajectory,
    recorded_traj: Trajectory,
    pos_penalty: float,
    orn_penalty: float,
    vel_penalty: float,
    ang_vel_penalty: float,
):
    pass


def safe_set_deviation(
    position: npt.ArrayLike,
    boxes: Union[Box, list[Box]],
    padding: Optional[npt.ArrayLike] = None,
    collision_penalty: float = 0,
) -> float:
    """Evaluate the cost function for a given position within an environment composed of safe boxes

    In general, the cost function rewards being centrally located within the safe set, with the cost increasing as the
    position approaches the wall. Costs are negative for being inside the safe set, and positive outside (in collision).
    We can add an additional large penalty for being in collision, to help ensure collision avoidance

    This essentially works out to be an L1 signed distance field

    Args:
        position (npt.ArrayLike): Position to evaluate the cost function
        boxes (Union[Box, list[Box]]): Local description of the safe set. This does NOT need to be every box in the
            environment, just the ones in the proximity of the position being evaluated
        padding (Optional[npt.ArrayLike]): Distance(s) to pad the safe set by (for instance, if we evaluate the central
            point of a spherical robot). Defaults to None.
        collision_penalty (float, optional): Additional penalty to increase the cost function if collision occurs.
            Defaults to 0.

    Returns:
        float: Cost function evaluation
    """
    dim = len(position)
    if isinstance(boxes, Box):
        boxes = [boxes]
    # If we're padding the boxes, go through each one and reduce their size by the padding amount
    if padding is not None:
        if isinstance(padding, (float, int)):
            padding = padding * np.ones(dim)
        else:
            padding = np.ravel(padding).astype(np.float64)
            assert len(padding) == dim
        for i, box in enumerate(boxes):
            boxes[i] = Box(box.lower + padding, box.upper - padding)
    # Keep track of which boxes we're inside, since the penalty depends on if we're in the safe set or not
    boxes_inside: list[Box] = []
    for box in boxes:
        if is_in_box(position, box):
            boxes_inside.append(box)
    # If we are not in collision, we only consider the boxes that we're inside
    if len(boxes_inside) != 0:
        return sum(
            max([*(position - box.upper), *(box.lower - position)])
            for box in boxes_inside
        )
    # If we are in collision, the collision depth is based on the nearest wall, which considers all boxes
    else:
        return collision_penalty + min(
            max([*(position - box.upper), *(box.lower - position)]) for box in boxes
        )


def integrated_safe_set_deviation(
    positions: npt.ArrayLike,
    box_or_boxes: Union[Box, list[Box]],
    padding: Optional[npt.ArrayLike] = None,
    collision_penalty: float = 0,
):
    dev = 0
    for pos in positions:
        dev += safe_set_deviation(pos, box_or_boxes, padding, collision_penalty)
    pass


def _visualize_penalty_test():
    """Create a 2D example of two safe-set boxes and visualize a heatmap of the penalty function"""
    n = 50
    X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    Z = np.empty_like(X)
    box_1 = Box([-1, -0.5], [0.5, 0.5])
    box_2 = Box([0, -1], [1, 1])
    plt.figure()

    padding = np.array([0.1, 0.1])
    penalty = 0.5
    new_boxes = [
        Box(box_1.lower + padding, box_1.upper - padding),
        Box(box_2.lower + padding, box_2.upper - padding),
    ]
    for i in range(n):
        for j in range(n):
            Z[i, j] = safe_set_deviation(
                (X[i, j], Y[i, j]), [box_1, box_2], padding, penalty
            )

    two_slope_norm = colors.TwoSlopeNorm(vmin=np.min(Z), vcenter=0, vmax=np.max(Z))
    heatmap = plt.pcolormesh(X, Y, Z, cmap="Spectral_r", norm=two_slope_norm)
    plot_2D_box(box_1, None, False, "k--")
    plot_2D_box(box_2, None, False, "k--")
    plot_2D_box(new_boxes[0], None, False, "b--")
    plot_2D_box(new_boxes[1], None, False, "b--")
    plt.colorbar(heatmap)
    plt.title("Safe Set Deviation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    _visualize_penalty_test()
