"""Functions associated with determining timing parameters for trajectories

- Timing heuristics help us estimate the amount of time allocated to trajectories
- Note: These heuristics are astrobee-specific
- The retiming optimization refines transition timing of spline trajectories
"""
# TODO
# - Some of these heuristics will be relatively poor. Try to figure out better estimates that are
#   not too computationally intensive to solve

import numpy as np
import numpy.typing as npt
import cvxpy as cp

from pyastrobee.utils.boxes import Box
from pyastrobee.trajectories.box_paths import intersection_path
from pyastrobee.config.astrobee_motion import LINEAR_SPEED_LIMIT, ANGULAR_SPEED_LIMIT
from pyastrobee.utils.quaternions import quaternion_angular_error
from pyastrobee.utils.errors import OptimizationError


def bezier_duration_heuristic(start_pt: npt.ArrayLike, end_pt: npt.ArrayLike) -> float:
    """Estimate the total duration of a trajectory comprised of a single Bezier curve

    Args:
        start_pt (npt.ArrayLike): Starting XYZ position, shape (3,)
        end_pt (npt.ArrayLike): Ending XYZ position, shape (3,)

    Returns:
        float: Estimated duration
    """
    dist = np.linalg.norm(end_pt - start_pt)
    return dist / (0.5 * LINEAR_SPEED_LIMIT)


def spline_duration_heuristic(
    start_pt: npt.ArrayLike, end_pt: npt.ArrayLike, boxes: list[Box]
) -> tuple[float, np.ndarray]:
    """Calculate a preliminary estimate for the time allocated to each curve in a spline

    Empirically, this seems to give a decent weighting and improves the solver reliability for retiming

    Args:
        start_pt (npt.ArrayLike): Starting XYZ position, shape (3,)
        end_pt (npt.ArrayLike): Ending XYZ position, shape (3,)
        boxes (list[Box]): Sequence of safe boxes that will be traveled through (Not the entire free space)

    Returns:
        tuple[float, np.ndarray]:
            float: Total duration estimate for the entire curve
            np.ndarray: Fractions of the total duration allocated to each box, shape (num_boxes,)
    """
    # Approximate the lengths of each path segment
    path_points = intersection_path(start_pt, end_pt, boxes)
    path_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    total_length = np.sum(path_lengths)
    fractional_lengths = path_lengths / total_length
    # Allocate a little extra time to the start and end to account for accel/decel
    fractional_lengths[0] *= 1.5
    fractional_lengths[-1] *= 1.5
    # Re-ensure the fractional lengths sum to one
    fractional_lengths /= np.sum(fractional_lengths)
    # Assume constant speed along each segment of half of the speed limit
    constant_speed = 0.5 * LINEAR_SPEED_LIMIT
    total_time = total_length / constant_speed
    return total_time, fractional_lengths


def rotation_duration_heuristic(q0: npt.ArrayLike, qf: npt.ArrayLike) -> float:
    """Calculate an estimate of how long a rotation will take

    Args:
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)

    Returns:
        float: Time estimate, seconds
    """
    err = quaternion_angular_error(q0, qf)
    err_mag = np.linalg.norm(err)
    return err_mag / (0.5 * ANGULAR_SPEED_LIMIT)


def retiming(
    kappa: float,
    costs: dict[int, dict[int, float]],
    durations: npt.ArrayLike,
    retiming_weights: dict[int, dict[int, float]],
) -> tuple[np.ndarray, float]:
    """Run the retiming trust-region-based optimization to generate an improved set of curve durations

    This code is essentially straight from Fast Path Planning with minimal modification since the retiming method
    is a bit complex, and we know that this works

    Args:
        kappa (float): Trust region parameter: Defines the maximum change in adjacent scaling factors
        costs (dict[int, dict[int, float]]): Breakdown of costs per curve and per derivative. Costs[i] gives the info
            for curve i, and costs[i][j] gives the cost associated with the jth derivative curve. Since we deal only
            with min-jerk, we only evaluate the j=3 case. The cost is the squared L2 norm of the jerk
        durations (npt.ArrayLike): Current best known value of the curve durations, shape (n_curves,)
        retiming_weights (dict[int, dict[int, float]]): A combination of Lagrangian multipliers and the last solved
            path. Weights[i] gives the weights associated with the ith differentiability (continuity) constraint, and
            weights[i][j] gives the weight associated with the jth derivative curve continuity. We enforce continuity
            up to the second derivative (j = 1 and 2)

    Returns:
        tuple[np.ndarray, float]:
            np.ndarray: The updated curve durations
            float: The new trust region parameter
    """
    # Decision variables.
    n_boxes = max(costs) + 1
    eta = cp.Variable(n_boxes)
    eta.value = np.ones(n_boxes)
    constr = [durations @ eta == sum(durations)]

    # Scale costs from previous trajectory.
    cost = 0
    for i, ci in costs.items():
        for j, cij in ci.items():
            cost += cij * cp.power(eta[i], 1 - 2 * j)

    # Retiming weights.
    for k in range(n_boxes - 1):
        for i, w in retiming_weights[k].items():
            cost += i * retiming_weights[k][i] * (eta[k + 1] - eta[k])

    # Trust region.
    if not np.isinf(kappa):
        constr.append(eta[1:] - eta[:-1] <= kappa)
        constr.append(eta[:-1] - eta[1:] <= kappa)

    # Solve SOCP and get new durarations.
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        print("Clarabel failed to solve the retiming problem. Retrying with MOSEK")
        prob.solve(solver=cp.MOSEK)
    if prob.status != cp.OPTIMAL:
        raise OptimizationError("Unable to solve the retiming problem")
    new_durations = np.multiply(eta.value, durations)

    # New candidate for kappa.
    kappa_max = max(np.abs(eta.value[1:] - eta.value[:-1]))
    print("Retiming time: ", prob.solver_stats.solve_time)

    return new_durations, kappa_max


def _test_timing_estimate():
    p0 = [0.1, 0.2, 0.3]
    pf = [1.5, 5, 1.7]
    safe_boxes = [
        Box((0, 0, 0), (1, 1, 1)),
        Box((0.5, 0.5, 0.5), (1.5, 5, 1.5)),
        Box((1, 4.5, 1), (2, 5.5, 2)),
    ]
    total_time, time_fractions = spline_duration_heuristic(p0, pf, safe_boxes)
    print("Time estimate: ", total_time)
    print("Fractional breakdown per box: ", time_fractions)
    print("Time per box: ", total_time * time_fractions)


if __name__ == "__main__":
    _test_timing_estimate()
