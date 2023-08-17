"""Preliminary test to see if we can call our planning method with a variable total duration

WORK IN PROGRESS
"""


from typing import Optional
import cvxpy as cp
import numpy as np
import numpy.typing as npt
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.bezier import BezierCurve
from pyastrobee.utils.boxes import Box
from pyastrobee.config.astrobee_motion import LINEAR_SPEED_LIMIT, LINEAR_ACCEL_LIMIT


def fix_time_optimize_points(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf: float,
    n_control_pts: int,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    box: Optional[Box] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    time_weight: float = 1,  # FIGURE THIS OUT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Check inputs
    n_constraints = sum(c is not None for c in [p0, pf, v0, vf, a0, af])
    if n_constraints > n_control_pts:
        raise ValueError(
            "Number of control points must be at least the number of constraints"
        )
    if tf <= t0:
        raise ValueError(f"Invalid time interval: ({t0}, {tf})")
    dim = len(p0)
    # Form the main Variable (the control points for the position curve) and get the Expressions
    # for the control points of the derivative curves
    pos_pts = cp.Variable((n_control_pts, dim))
    pos_curve = BezierCurve(pos_pts, t0, tf)
    vel_curve = pos_curve.derivative
    vel_pts = vel_curve.points
    accel_curve = vel_curve.derivative
    accel_pts = accel_curve.points
    jerk_curve = accel_curve.derivative
    # Form the constraint list depending on what was specified in the inputs
    constraints = [pos_pts[0] == p0, pos_pts[-1] == pf]
    if v0 is not None:
        constraints.append(vel_pts[0] == v0)
    if vf is not None:
        constraints.append(vel_pts[-1] == vf)
    if a0 is not None:
        constraints.append(accel_pts[0] == a0)
    if af is not None:
        constraints.append(accel_pts[-1] == af)
    if box is not None:
        lower, upper = box
        constraints.append(pos_pts >= lower)
        constraints.append(pos_pts <= upper)
    if v_max is not None:
        constraints.append(cp.norm2(vel_pts, axis=1) <= v_max)
    if a_max is not None:
        constraints.append(cp.norm2(accel_pts, axis=1) <= a_max)
    # Objective function criteria
    jerk = jerk_curve.l2_squared
    # Form the objective function based on the relative weighting between the criteria
    objective = cp.Minimize(jerk + time_weight * (tf - t0))
    # Form the problem and solve it
    # Note: Clarabel is apparently better for quadratic objectives (like our jerk criteria)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )
    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_pos_curve = BezierCurve(pos_pts.value, t0, tf)
    return solved_pos_curve, prob.value


def fix_points_optimize_time(
    init_curve: BezierCurve,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    time_weight: float = 1,  # FIGURE THIS OUT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0 = init_curve.a
    tf = cp.Variable(1)
    # tf.value = init_curve.b  # Initialize with the time from the initial solution
    if isinstance(init_curve.points, (cp.Variable, cp.Expression)):
        points = init_curve.points.value
    else:
        points = init_curve.points
    pos_curve = BezierCurve(points, t0, tf)
    vel_curve = pos_curve.derivative
    vel_pts = vel_curve.points
    accel_curve = vel_curve.derivative
    accel_pts = accel_curve.points
    jerk_curve = accel_curve.derivative
    # Form the constraint list depending on what was specified in the inputs
    # NOTE: since the initial position points are fixed, we don't need to constrain p0, pf
    constraints = []
    if v0 is not None:
        constraints.append(vel_pts[0] == v0)
    if vf is not None:
        constraints.append(vel_pts[-1] == vf)
    if a0 is not None:
        constraints.append(accel_pts[0] == a0)
    if af is not None:
        constraints.append(accel_pts[-1] == af)
    if v_max is not None:
        constraints.append(cp.norm2(vel_pts, axis=1) <= v_max)
    if a_max is not None:
        constraints.append(cp.norm2(accel_pts, axis=1) <= a_max)
    # Objective function criteria
    jerk = jerk_curve.l2_squared
    # Form the objective function based on the relative weighting between the criteria
    objective = cp.Minimize(jerk + time_weight * (tf - t0))
    # Form the problem and solve it
    # Note: Clarabel is apparently better for quadratic objectives (like our jerk criteria)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )
    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_pos_curve = BezierCurve(points, t0, tf)
    return solved_pos_curve, prob.value


def optimal_bezier(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    tf_init: float,
    n_control_pts: int,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    box: Optional[Box] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
    time_weight: float = 1,  # FIGURE THIS OUT
) -> BezierCurve:
    tf = tf_init
    max_iters = 10
    for i in range(max_iters):
        # fmt: off
        try:
            curve_1, cost_1 = fix_time_optimize_points(p0, pf, t0, tf, n_control_pts, v0, vf, a0, af, box, v_max, a_max, time_weight)
        except cp.error.SolverError as e:
            # IMPROVE THIS
            curve_1, cost_1 = fix_time_optimize_points(p0, pf, t0, tf * 5, n_control_pts, v0, vf, a0, af, box, v_max, a_max, time_weight)
        curve_2, cost_2 = fix_points_optimize_time(curve_1, v0, vf, a0, af, v_max, a_max, time_weight)
        tf = curve_2.b.value if isinstance(curve_2.b, (cp.Variable, cp.Expression)) else curve_2.b
        print(f"ITERATION {i}:\nCost from point optimization: {cost_1}. Time: {curve_1.b}\n" + 
              f"Cost from time optimization: {cost_2}. Time: {tf}")
    return curve_2
    # fmt: on


def traj_from_bezier(curve: BezierCurve, dt: float) -> Trajectory:
    t0 = curve.a
    tf = curve.b.value if isinstance(curve.b, (cp.Variable, cp.Expression)) else curve.b
    times = np.arange(t0, tf + dt, dt)
    pos = curve(times)
    vel = curve.derivative(times)
    accel = curve.derivative.derivative(times)
    return Trajectory(pos, None, vel, None, accel, None, times)


def main():
    p0 = (0, 0, 0)
    pf = (1, 2, 3)
    t0 = 0
    tf_init = 20
    n_control_pts = 8
    dt = 0.1
    v0 = (0.1, 0.2, 0.3)
    vf = (0, 0, 0)
    a0 = (0, 0, 0)
    af = (0, 0, 0)
    time_weight = 1
    print("Speed limit: ", LINEAR_SPEED_LIMIT)
    print("Accel limit: ", LINEAR_ACCEL_LIMIT)
    curve = optimal_bezier(
        p0,
        pf,
        t0,
        tf_init,
        n_control_pts,
        v0,
        vf,
        a0,
        af,
        None,
        LINEAR_SPEED_LIMIT,
        LINEAR_ACCEL_LIMIT,
        time_weight,
    )
    traj = traj_from_bezier(curve, dt)
    traj.plot()


if __name__ == "__main__":
    main()
