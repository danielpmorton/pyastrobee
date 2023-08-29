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
    box_or_boxes: Union[Box, list[Box]],
    padding: Optional[npt.ArrayLike] = None,
    collision_penalty: float = 0,
):
    # TODO clean up some of this weird input handling...
    dim = len(position)
    if isinstance(box_or_boxes, Box):
        box_or_boxes = [box_or_boxes]
    boxes = []
    if padding is None:
        boxes = box_or_boxes
    else:
        padding = np.ravel(padding).astype(np.float64)
        assert len(padding) == dim
        for box in box_or_boxes:
            boxes.append(Box(box.lower + padding, box.upper - padding))

    boxes_inside: list[Box] = []
    for box in boxes:
        if is_in_box(position, box):
            boxes_inside.append(box)

    # If we are in the safe set, don't penalize for being outside of another box in the safe set
    # (just base this on how close we are to the boundary of the box we are in)
    if len(boxes_inside) != 0:
        # TODO check if this is better as a sum(max()) or a min(max())
        # NOTE sum(max()) seems to be a bit better since we can gurantee the function is continuous across boxes?
        return sum(
            max([*(position - box.upper), *(box.lower - position)])
            for box in boxes_inside
        )

    else:
        # If we are in collision,... TODO explain better
        return collision_penalty + min(
            max([*(position - box.upper), *(box.lower - position)]) for box in boxes
        )


def integrated_safe_set_deviation(
    positions: npt.ArrayLike, box_or_boxes: Union[Box, list[Box]]
):
    pass


def _visualize_penalty_test():
    n = 50
    X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    Z = np.empty_like(X)
    box_1 = Box([-1, -0.5], [0.5, 0.5])
    box_2 = Box([0, -1], [1, 1])
    plt.figure()

    padding = np.array([0.1, 0.1])
    penalty = 0
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
