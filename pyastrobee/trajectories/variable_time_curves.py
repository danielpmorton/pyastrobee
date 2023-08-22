"""Preliminary test to see if we can call our planning method with a variable total duration

WORK IN PROGRESS

Notes on the search method for the duration:
- We know that the jerk function is a quadratic, and if we add an affine factor based on the total duration
  of the trajectory, it will still be a quadratic. So, a version of quadratic fit search will work well here
- Depending on the weighting of this affine time component, the cost may look very linear. Even if this is the
  case, the search method as implemented will work well, because we will continually approach the boundary
  of feasibility until we stop within some tolerance
- Speaking of this feasibility boundary, this is the main difference between the implemented method and standard
  quadratic fit search. There is some T such that the trajectory is no longer feasible, given the constraints
  on velocity/accel/BCs..., and so in general, we want to solve for a T which is small, yet feasible. So, this
  search method incorporates this knowledge of this infeasible region for small time intervals.
- Ideally, we'd just be able to plug this into CVXPY (since it should just be a quadratic program with some 
  constraints anyways). However, I tried a bunch of formulations of the constraints and it didn't seem to be
  convex or DCP (often leading to either quadratic forms of two variables, or equality constraints on convex 
  functions). Maybe there is a better formulation out there...
"""

from typing import Optional, Union, Callable, Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pybullet
import matplotlib.pyplot as plt

from pyastrobee.trajectories.trajectory import Trajectory, plot_traj_constraints
from pyastrobee.trajectories.bezier import BezierCurve
from pyastrobee.trajectories.splines import CompositeBezierCurve
from pyastrobee.trajectories.curve_utils import traj_from_curve
from pyastrobee.utils.boxes import Box
from pyastrobee.utils.debug_visualizer import animate_path
from pyastrobee.config.astrobee_motion import LINEAR_SPEED_LIMIT, LINEAR_ACCEL_LIMIT
from pyastrobee.utils.errors import OptimizationError


def left_quadratic_fit_search(
    f: Callable[[float], float | tuple[float, Any]],
    x_init: float,
    dx_tol: float,
    max_iters: int,
) -> tuple[float, float, list[Any]]:
    """A version of quadratic fit search that assumes we have an infeasible region for small x (x >= 0)

    Reference: Algorithms for Optimization (Kochenderfer), Algorithm 3.4

    Args:
        f (Callable[[float], float  |  tuple[float, Any]]): Univariate function to optimize, callable as f(x).
            The return must have the cost of the evaluation as the first output.
        x_init (float): Initial location to start the search
        dx_tol (float): Stopping tolerance on evaluation points: Terminate if the change between consecutive
            evaluation points is less than this tolerance
        max_iters (int): Maximum iterations of the algorithm (if the stopping tolerance is not achieved)

    Raises:
        OptimizationError: If no feasible solution is found in max_iters iterations

    Returns:
        Tuple of:
            float: Best evaluation point x
            float: Cost of the function evaluation at the best x value
            list[Any]: Additional outputs of the function being optimized at the best x value
    """
    # Mutable dicts to keep track of the optimization process
    best = {"x": None, "cost": np.inf, "out": None}  # init
    log = {"iters": 0, "feasibility_bound": 0}  # init

    # Create wrapper around the function to handle if it has multiple outputs
    # Return will solely be the cost of the evaluation, but we store the other outputs
    # in the dictionaries as needed
    def _f(x: float) -> float:
        fx = f(x)
        log["iters"] += 1
        if isinstance(fx, tuple):
            cost, *out = fx
            if out == []:
                out = None
        else:
            cost = fx
            out = None
        # Check to see if this is the best so far - if so, update
        if cost <= best["cost"] and cost != np.inf:
            best["x"] = x
            best["cost"] = cost
            best["out"] = out
        return cost

    # Find the quadratic fit search interval (a, b, c) given an initial search location
    # This assumes that x is a positive value and that the only infeasible values occurs
    # when x is too small
    def _find_init_interval_from_guess(x: float):
        b = x
        yb = _f(b)
        if yb == np.inf:
            while yb == np.inf and log["iters"] <= max_iters - 1:
                log["feasibility_bound"] = max(b, log["feasibility_bound"])
                b *= 2
                yb = _f(b)
        a = (log["feasibility_bound"] + b) / 2
        ya = _f(a)
        if ya == np.inf:
            while ya == np.inf and log["iters"] <= max_iters - 1:
                log["feasibility_bound"] = max(a, log["feasibility_bound"])
                a = (a + b) / 2
                ya = _f(a)
        # we know c will be valid
        c = b + (b - a)
        yc = _f(c)
        return a, b, c, ya, yb, yc

    a, b, c, ya, yb, yc = _find_init_interval_from_guess(x_init)
    x_prev = None  # init
    while log["iters"] <= max_iters - 1:
        # Quadratic fit for the next search location
        x = (
            0.5
            * (ya * (b**2 - c**2) + yb * (c**2 - a**2) + yc * (a**2 - b**2))
            / (ya * (b - c) + yb * (c - a) + yc * (a - b))
        )
        # Handle if the fit location is known to be infeasible
        if x <= log["feasibility_bound"]:
            x = (log["feasibility_bound"] + a) / 2
        yx = _f(x)
        if yx == np.inf:  # Infeasible
            log["feasibility_bound"] = max(log["feasibility_bound"], x)
        else:
            # Standard quadratic fit update, with extra cases when x is not between a and c
            if x < a:
                if yx < ya:
                    a, ya = x, yx
            elif a <= x <= c:
                if x > b:
                    if yx > yb:
                        c, yc = x, yx
                    else:
                        a, ya, b, yb = b, yb, x, yx
                elif x < b:
                    if yx > yb:
                        a, ya = x, yx
                    else:
                        c, yc, b, yb = b, yb, x, yx
            else:  # x > c
                if yx < yc:
                    c, yc = x, yx
        # Termination criteria: if our evaluation point update has shrunk to within some tolerance
        if x_prev is not None and abs(x - x_prev) < dx_tol:
            break
        x_prev = x

    if best["x"] is None:
        raise OptimizationError("Unable to find a feasible solution")
    return best["x"], best["cost"], best["out"]


# TODO can we separate out the bisection mechanic so that we can use this for the spline as well?
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

    costs = {}  # TODO remove the plotting code

    def _f(t):
        kwargs = curve_kwargs | {"tf": t}
        print("Evaluating time: ", t)
        try:
            cost, curve = fix_time_optimize_points(**kwargs)
        except OptimizationError:
            cost, curve = np.inf, None
        costs[t] = cost
        return cost, curve

    t, cost, out = left_quadratic_fit_search(_f, tf, 1e-1, 20)
    best_curve = out[0]
    fig = plt.figure()
    ts, cs = zip(*costs.items())
    plt.scatter(ts, cs)

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # TODO UPDATE OUTPUTS
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
    # TODO MAKE THIS EXCEPTION CAPTURE A PART OF THE MAIN BEZIER FUNCTION
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as e:
        raise OptimizationError("Cannot generate the trajectory - Solver error!") from e
    if prob.status != cp.OPTIMAL:
        raise OptimizationError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )
    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_pos_curve = BezierCurve(pos_pts.value, t0, tf)

    # TODO decide on order of outputs!!
    return prob.value, solved_pos_curve


def main():
    p0 = (0, 0, 0)
    pf = (1, 2, 3)
    t0 = 0
    tf_init = 30
    n_control_pts = 30
    dt = 0.1
    v0 = (0.3, 0.2, 0.1)
    vf = (0, 0, 0)
    a0 = (0, 0, 0)
    af = (0, 0, 0)
    time_weight = 0  # 0.01
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
