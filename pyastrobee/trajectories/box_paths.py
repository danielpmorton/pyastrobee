"""Different ways of characterizing paths between regions of the environment

These are mainly used for timing heuristics for splines right now - for instance, 
using a simple method of traversing the environment to create an estimate of how
long we will spend in each region
"""

import time
import numpy as np
import numpy.typing as npt
import cvxpy as cp
import pybullet

from pyastrobee.utils.boxes import Box, intersect_boxes, visualize_3D_box
from pyastrobee.utils.debug_visualizer import visualize_path, visualize_points
from pyastrobee.utils.errors import OptimizationError


def min_length_path(
    start_pt: npt.ArrayLike, end_pt: npt.ArrayLike, boxes: list[Box]
) -> np.ndarray:
    """Determine the minimum-length path between two points through a sequence of safe boxes

    Args:
        start_pt (npt.ArrayLike): Starting XYZ position, shape (3,)
        end_pt (npt.ArrayLike): Ending XYZ position, shape (3,)
        boxes (list[Box]): Sequence of safe boxes to pass through

    Returns:
        np.ndarray: Path from start to end, shape (n_boxes + 1, 3)
    """
    n_boxes = len(boxes)
    points = cp.Variable((n_boxes + 1, 3))
    pathlength = cp.sum(cp.norm2(cp.diff(points, axis=0), axis=1))
    objective = cp.Minimize(pathlength)
    constraints = [points[0] == start_pt, points[-1] == end_pt]
    for i, box in enumerate(boxes):
        lower, upper = box
        constraints.append(points[i] >= lower)
        constraints.append(points[i] <= upper)
        constraints.append(points[i + 1] >= lower)
        constraints.append(points[i + 1] <= upper)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        raise OptimizationError(
            "Unable to find a solution.\n"
            + "Check that the set of boxes is connected and that the start/end points are contained in the boxes"
        )
    return points.value


def centerpoint_path(
    start_pt: npt.ArrayLike, end_pt: npt.ArrayLike, boxes: list[Box]
) -> np.ndarray:
    """Determine a path between two points, using the center points of boxes and their intersections as waypoints

    Args:
        start_pt (npt.ArrayLike): Starting XYZ position, shape (3,)
        end_pt (npt.ArrayLike): Ending XYZ position, shape (3,)
        boxes (list[Box]): Sequence of safe boxes to pass through

    Returns:
        np.ndarray: Path from start to end, shape (2 * n_boxes + 1, 3)
    """
    n_boxes = len(boxes)
    # One point per box, one point per box intersection, and two points for start/end
    n_points = 2 * n_boxes + 1
    points = np.empty((n_points, 3))
    points[0] = start_pt
    points[-1] = end_pt
    for i in range(n_boxes):
        points[2 * i + 1] = boxes[i].center
        if i < n_boxes - 1:  # Not the last box
            points[2 * i + 2] = intersect_boxes(boxes[i], boxes[i + 1]).center
    return points


def intersection_path(
    start_pt: npt.ArrayLike, end_pt: npt.ArrayLike, boxes: list[Box]
) -> np.ndarray:
    """Determine a path between two points, using the intersections between consecutive boxes as waypoints

    Args:
        start_pt (npt.ArrayLike): Starting XYZ position, shape (3,)
        end_pt (npt.ArrayLike): Ending XYZ position, shape (3,)
        boxes (list[Box]): Sequence of safe boxes to pass through

    Returns:
        np.ndarray: Path from start to end, shape (n_boxes + 1, 3)
    """
    n_boxes = len(boxes)
    n_points = n_boxes + 1
    points = np.empty((n_points, 3))
    points[0] = start_pt
    points[-1] = end_pt
    for i in range(n_boxes - 1):
        points[i + 1] = intersect_boxes(boxes[i], boxes[i + 1]).center
    return points


def _test_path_methods():
    start_pt = [0.1, 0.2, 0.3]
    end_pt = [1.5, 5, 1.7]
    boxes = [
        Box((0, 0, 0), (1, 1, 1)),
        Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
        Box((1, 4.5, 1), (2, 5.5, 2)),
    ]
    time_a = time.time()
    min_length_solution = min_length_path(start_pt, end_pt, boxes)
    time_b = time.time()
    centerpoint_solution = centerpoint_path(start_pt, end_pt, boxes)
    time_c = time.time()
    intersection_solution = intersection_path(start_pt, end_pt, boxes)
    time_d = time.time()
    print("min length time: ", time_b - time_a)
    print("centerpoint time: ", time_c - time_b)
    print("intersection time: ", time_d - time_c)
    pybullet.connect(pybullet.GUI)
    min_length_color = (1, 0, 0)
    centerpoint_color = (0, 1, 0)
    intersection_color = (0, 0, 1)
    visualize_path(min_length_solution, color=min_length_color)
    visualize_path(centerpoint_solution, color=centerpoint_color)
    visualize_path(intersection_solution, color=intersection_color)
    visualize_points(min_length_solution, color=min_length_color)
    visualize_points(centerpoint_solution, color=centerpoint_color)
    visualize_points(intersection_solution, color=intersection_color)
    for box in boxes:
        visualize_3D_box(box)
    pybullet.addUserDebugText("Min length", [0, 0, -0.2], min_length_color)
    pybullet.addUserDebugText("Centerpoint", [0, 0, -0.4], centerpoint_color)
    pybullet.addUserDebugText("Intersection", [0, 0, -0.6], intersection_color)
    input("Press Enter to close")


if __name__ == "__main__":
    _test_path_methods()
