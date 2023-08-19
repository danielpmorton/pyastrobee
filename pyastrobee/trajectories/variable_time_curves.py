"""Preliminary test to see if we can call our planning method with a variable total duration

WORK IN PROGRESS
"""

from typing import Optional, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pybullet

from pyastrobee.trajectories.trajectory import Trajectory, plot_traj_constraints
from pyastrobee.trajectories.bezier import BezierCurve
from pyastrobee.trajectories.splines import CompositeBezierCurve
from pyastrobee.utils.boxes import Box
from pyastrobee.utils.debug_visualizer import animate_path
from pyastrobee.config.astrobee_motion import LINEAR_SPEED_LIMIT, LINEAR_ACCEL_LIMIT


# TODO can we separate out the binary search mechanic so that we can use this for the spline as well?
def bezier_with_retiming(
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
    time_weight: float = 0.1,  # FIGURE THIS OUT
    reduction_tol: float = 1e-2,
    max_iters: int = 15,
):
    curve_kwargs = dict(
        p0=p0,
        pf=pf,
        t0=t0,
        tf=tf,
        n_control_pts=n_control_pts,
        v0=v0,
        vf=vf,
        a0=a0,
        af=af,
        box=box,
        v_max=v_max,
        a_max=a_max,
        time_weight=time_weight,
    )

    infeasibility_bound = None
    best_tf = None
    best_cost = np.inf
    best_curve = None
    prev_tf = np.inf
    cur_tf = tf
    for i in range(max_iters):
        # Solve for a curve (if possible) with the current final time
        prev_tf = cur_tf
        curve_kwargs["tf"] = cur_tf
        try:
            curve, cost = fix_time_optimize_points(**curve_kwargs)
        except cp.error.SolverError:
            curve, cost = None, np.inf
        # Binary search on the lowest final time based on the cost of the curve
        print("Cost: ", cost, " for time: ", cur_tf, end="")
        if cost < best_cost:
            best_cost = cost
            best_curve = curve
            best_tf = cur_tf
            cur_tf = (
                cur_tf / 2
                if infeasibility_bound is None
                else (cur_tf + infeasibility_bound) / 2
            )
        else:  # Worse cost (assume this must be because we went so low it's infeasible)
            infeasibility_bound = (
                cur_tf
                if infeasibility_bound is None
                else max(infeasibility_bound, cur_tf)
            )
            if best_tf is not None:
                cur_tf = (best_tf + cur_tf) / 2
            else:
                cur_tf *= 2
        reduction = abs(prev_tf - cur_tf) / cur_tf
        print(". Percent time change: ", reduction)
        if reduction < reduction_tol:
            break
    if best_curve is None:
        # TODO use a different type of exception
        raise ValueError("Unable to find a solution")
    return best_curve


# TODO this is basically the same we the main bezier trajectory function
# So, merge these
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


# TODO move this to a better location
# NOTE: Don't make a circular import since this imports both Bezier and CompositeBezier
def traj_from_curve(
    curve: Union[BezierCurve, CompositeBezierCurve], dt: float
) -> Trajectory:
    """Construct a position-only trajectory from a Bezier curve or spline

    Args:
        curve (Union[BezierCurve, CompositeBezierCurve]): Curve for the position motion
        dt (float): Timestep

    Returns:
        Trajectory: Position (and derivatives) trajectory information
    """
    t0 = curve.a
    tf = curve.b.value if isinstance(curve.b, (cp.Variable, cp.Expression)) else curve.b
    # TODO see if we can refine how this time works... The spacing isn't going to be exactly dt
    times = np.linspace(t0, tf, round((tf - t0) / dt))
    pos = curve(times)
    vel = curve.derivative(times)
    accel = curve.derivative.derivative(times)
    return Trajectory(pos, None, vel, None, accel, None, times)


def main():
    p0 = (0, 0, 0)
    pf = (1, 2, 3)
    t0 = 0
    tf_init = 20
    n_control_pts = 30
    dt = 0.1
    v0 = (0.3, 0.2, 0.1)
    vf = (0, 0, 0)
    a0 = (0, 0, 0)
    af = (0, 0, 0)
    time_weight = 0.1
    print("Speed limit: ", LINEAR_SPEED_LIMIT)
    print("Accel limit: ", LINEAR_ACCEL_LIMIT)
    # curve = optimal_bezier(
    curve = bezier_with_retiming(
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
    traj = traj_from_curve(curve, dt)
    traj.plot()
    plot_traj_constraints(
        traj, None, LINEAR_SPEED_LIMIT, LINEAR_ACCEL_LIMIT, None, None
    )
    pybullet.connect(pybullet.GUI)
    traj.visualize(30)
    animate_path(traj.positions, traj.num_timesteps // 5)
    input("Animation complete, press Enter to finish")


if __name__ == "__main__":
    main()
