"""Using Bezier curves to optimize SLERPs with constrained derivatives and derivative magnitudes"""

from typing import Union, Optional

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.quaternions import (
    quaternion_slerp,
    quaternion_dist,
    random_quaternion,
    quats_to_angular_velocities,
)
from pyastrobee.trajectories.bezier import BezierCurve, plot_1d_bezier_curve


def bezier_slerp(
    q1: Union[npt.ArrayLike, Quaternion],
    q2: Union[npt.ArrayLike, Quaternion],
    n: int,
    T: float,
    w_max: Optional[float] = None,
    dw_max: Optional[float] = None,
):
    """SLERP based on a Bezier-curve discretization with 0 first and second derivative at either end,
    and limits on the maximum angular velocity/acceleration (if desired)

    The angular velocity vector has a fixed direction since this is a limitation of a single SLERP arc

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Starting quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        q2 (Union[Quaternion, npt.ArrayLike]): Ending quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        n (int): Number of points at which to evaluate the SLERP
        T (float): Time period for the SLERP. Used in conjunction with the w_max and dw_max parameters. If
            this is not of interest, set T = 1
        w_max (Optional[float]): Maximum angular velocity magnitude. Defaults to None (unconstrained).
        dw_max (Optional[float]): Maximum angular acceleration magnitude. Defaults to None (unconstrained).

    Returns:
        np.ndarray: The interpolated XYZW quaternions, shape = (n, 4)
    """
    if isinstance(q1, Quaternion):
        q1 = q1.xyzw
    if isinstance(q2, Quaternion):
        q2 = q2.xyzw

    # If we have constraints on the max velocity/acceleration, bump up the degree of the curve by a lot to make sure
    # that the convex hull of the control points will be a tighter bound on the curve (so that our constraints are not
    # overly restrictive)
    if w_max is not None or dw_max is not None:
        n_control_pts = 20
    else:
        n_control_pts = 10

    # Determine the maximum derivatives of the interpolation curve based on the quaternion distance metric
    q_dist = quaternion_dist(q1, q2)
    d_max = None if w_max is None else w_max * T / q_dist
    d2_max = None if dw_max is None else dw_max * T**2 / q_dist
    solved_curve = bezier_interpolation_curve(n_control_pts, 0, 0, 0, 0, d_max, d2_max)
    pcts = solved_curve(np.linspace(0, 1, n))
    # Clamp between 0 and 1 to account for any numerical issues, and convert to 1d array
    pcts = np.ravel(np.clip(pcts, 0, 1))
    return quaternion_slerp(q1, q2, pcts)


def bezier_interpolation_curve(
    n_control_pts: int,
    d0: Optional[float] = None,
    df: Optional[float] = None,
    d20: Optional[float] = None,
    d2f: Optional[float] = None,
    d_max: Optional[float] = None,
    d2_max: Optional[float] = None,
) -> BezierCurve:
    """Use a min-jerk Bezier curve to define discretization percentages between two interpolated points

    Interpolation percentages can be found by evaluating the curve on an array with uniform-spaced
    times between 0 and 1

    This curve will satisfy derivative boundary conditions and constraints on maximum derivatives (as specified)

    Args:
        n_control_pts (int): Number of control points of the Bezier curve. Must be at least the number of constraints
            that are specified. More control points (~20) will lead to a tighter convex hull of the curve,
            meaning constraints on maximum velocity/acceleration will be more precise
        d0 (Optional[float]): Initial derivative. Defaults to None (unconstrained).
        df (Optional[float]): Final derivative. Defaults to None (unconstrained).
        d20 (Optional[float]): Initial second derivative. Defaults to None (unconstrained).
        d2f (Optional[float]): Final second derivative. Defaults to None (unconstrained).
        d_max (Optional[float]): Maximum derivative. Defaults to None (unconstrained).
        d2_max (Optional[float]): Maximum second derivative. Defaults to None (unconstrained).

    Raises:
        cp.error.SolverError: If a solution cannot be found. If this happens, try increasing the number of control
            points, which may lead to a more feasible constraints. Otherwise, the problem may have no solution

    Returns:
        BezierCurve: Curve to use for interpolation
    """
    t0 = 0
    tf = 1  # Unit time
    pts = cp.Variable((n_control_pts, 1))
    curve = BezierCurve(pts, t0, tf)
    d_curve = curve.derivative
    d_pts = d_curve.points
    d2_curve = d_curve.derivative
    d2_pts = d2_curve.points
    d3_curve = d2_curve.derivative
    # Form the constraint list depending on what was specified in the inputs
    constraints = [pts[0] == 0, pts[-1] == 1]
    if d0 is not None:
        constraints.append(d_pts[0] == d0)
    if df is not None:
        constraints.append(d_pts[-1] == df)
    if d20 is not None:
        constraints.append(d2_pts[0] == d20)
    if d2f is not None:
        constraints.append(d2_pts[-1] == d2f)
    if d_max is not None:
        constraints.append(cp.max(cp.abs(d_pts)) <= d_max)
    if d2_max is not None:
        constraints.append(cp.max(cp.abs(d2_pts)) <= d2_max)
    # Form the problem and solve it
    jerk = d3_curve.l2_squared
    objective = cp.Minimize(jerk)
    prob = cp.Problem(objective, constraints)
    # Note: Clarabel is apparently better for quadratic objectives (like our jerk criteria)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )
    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_curve = BezierCurve(pts.value, t0, tf)
    return solved_curve


def _test_constraint_example():
    np.random.seed(0)
    q1 = np.array([0, 0, 0, 1])  # random_quaternion()
    q2 = random_quaternion()
    print("Quaternion distance: ", quaternion_dist(q1, q2))
    T = 10
    n = 500
    dt = T / n
    w_max = 0.40
    dw_max = 0.20
    print("Maximum angular velocity constraint: ", w_max)
    print("Maximum angular acceleration constraint: ", dw_max)
    quats = bezier_slerp(q1, q2, n, T, w_max, dw_max)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    print(
        "Maximum angular velocity from curve: ", np.max(np.linalg.norm(omega, axis=1))
    )
    print(
        "Maximum angular acceleration from curve: ",
        np.max(np.linalg.norm(alpha, axis=1)),
    )
    traj = Trajectory(None, quats, None, omega, None, alpha)
    traj.plot()


def _test_1d_interpolation():
    curve = bezier_interpolation_curve(20, 0, 0, 0, 0)
    plt.figure()
    plot_1d_bezier_curve(curve, show=False)
    plt.title("Position")
    plt.figure()
    plot_1d_bezier_curve(curve.derivative, show=False)
    plt.title("Velocity")
    plt.figure()
    plot_1d_bezier_curve(curve.derivative.derivative, show=False)
    plt.title("Acceleration")
    plt.show()


if __name__ == "__main__":
    _test_constraint_example()
    _test_1d_interpolation()
