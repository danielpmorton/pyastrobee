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


def variable_time_bezier_trajectory(
    p0: npt.ArrayLike,
    pf: npt.ArrayLike,
    t0: float,
    n_control_pts: int,
    n_timesteps: int,
    v0: Optional[npt.ArrayLike] = None,
    vf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
    box: Optional[Box] = None,
    v_max: Optional[float] = None,
    a_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tf = cp.Variable(1)

    # Check inputs
    n_constraints = sum(c is not None for c in [p0, pf, v0, vf, a0, af])
    if n_constraints > n_control_pts:
        raise ValueError(
            "Number of control points must be at least the number of constraints"
        )
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
    theta = 1  # NEW: tuning parameter for the time weighting
    # Form the objective function based on the relative weighting between the criteria
    objective = cp.Minimize(jerk + theta * (tf - t0))
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
    solved_vel_curve = solved_pos_curve.derivative
    solved_accel_curve = solved_vel_curve.derivative
    times = np.linspace(0, tf.value, n_timesteps)
    return (
        solved_pos_curve(times),
        solved_vel_curve(times),
        solved_accel_curve(times),
        times,
    )


p0 = (0, 0, 0)
pf = (1, 2, 3)
t0 = 0
# tf = 20
n_control_pts = 8
n_timesteps = 50
v0 = (0, 0, 0)
vf = (0, 0, 0)
a0 = (0, 0, 0)
af = (0, 0, 0)
print("Speed limit: ", LINEAR_SPEED_LIMIT)
print("Accel limit: ", LINEAR_ACCEL_LIMIT)
pos, vel, accel, time = variable_time_bezier_trajectory(
    p0,
    pf,
    t0,
    n_control_pts,
    n_timesteps,
    v0,
    vf,
    a0,
    af,
    None,
    # LINEAR_SPEED_LIMIT,
    # LINEAR_ACCEL_LIMIT,
)
traj = Trajectory(pos, None, vel, None, accel, None, time)
print("Maximum velocity magnitude: ", np.max(np.linalg.norm(vel, axis=1)))
print("Maximum acceleration magnitude: ", np.max(np.linalg.norm(accel, axis=1)))

traj.plot()
