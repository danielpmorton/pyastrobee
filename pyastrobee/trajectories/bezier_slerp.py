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
    w_max=None,
    dw_max=None,
):
    """

    TODO UPDATE DOCSTRING

    SLERP based on a third-order polynomial discretization

    This will interpolate quaternions based on a polynomial spacing rather than a linear spacing. The resulting
    angular velocity vector has a constant direction, but will be quadratic, starting and ending at 0

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Starting quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        q2 (Union[Quaternion, npt.ArrayLike]): Ending quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        n (int): Number of points at which to evaluate the polynomial-based SLERP

    Returns:
    UPDATE
        np.ndarray: The interpolated XYZW quaternions, shape = (n, 4)
    """
    # Generate our evaluation points for SLERP so that:
    # - We evaluate over a domain of [0, 1] with n steps
    # - We want to start at 0 and end at 1 with 0 derivative at either end
    # pcts = third_order_poly(0, 1, 0, 1, 0, 0, n)[0]

    if isinstance(q1, Quaternion):
        q1 = q1.xyzw
    if isinstance(q2, Quaternion):
        q2 = q2.xyzw

    # If we have constraints, bump up the degree of the curve by a lot to make sure that the convex hull of the
    # control points will be a tighter bound on the curve (so that our constraints are not overly restrictive)
    if w_max is not None or dw_max is not None:
        n_control_pts = 20
    else:
        n_control_pts = 10
    # Form the main Variable (the control points for the position curve) and get the Expressions
    # for the control points of the derivative curves
    pts = cp.Variable((n_control_pts, 1))
    curve = BezierCurve(pts, 0, 1)
    d_curve = curve.derivative
    d_pts = d_curve.points
    d2_curve = d_curve.derivative
    d2_pts = d2_curve.points
    d3_curve = d2_curve.derivative
    # Form the constraint list depending on what was specified in the inputs
    constraints = [
        pts[0] == 0,
        pts[-1] == 1,
        d_pts[0] == 0,
        d_pts[-1] == 0,
        d2_pts[0] == 0,
        d2_pts[-1] == 0,
    ]

    # We know that since we start and stop from rest, the maximum velocity will be at 0.5 * T
    # and the maximum accelerations will occur at ??

    q_dist = quaternion_dist(q1, q2)
    if w_max is not None:
        constraints.append((q_dist / T) * cp.max(cp.abs(d_pts)) <= w_max)
    if dw_max is not None:
        constraints.append((q_dist / T**2) * cp.max(cp.abs(d2_pts)) <= dw_max)

    # Objective function criteria
    jerk = d3_curve.l2_squared
    # Form the objective function based on the relative weighting between the criteria
    objective = cp.Minimize(jerk)
    # Form the problem and solve it
    # Note: Clarabel is apparently better for quadratic objectives (like our jerk criteria)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise cp.error.SolverError(
            f"Unable to generate the trajectory (solver status: {prob.status}).\n"
            + "Check on the feasibility of the constraints"
        )

    print("VELOCITY CONSTRAINT VALUE: ", ((q_dist / T) * cp.max(cp.abs(d_pts))).value)
    print(
        "ACCEL CONSTRAINT VALUE: ", ((q_dist / T**2) * cp.max(cp.abs(d2_pts))).value
    )

    # Construct the Bezier curves from the solved control points, and return their evaluations at each time
    solved_curve = BezierCurve(pts.value, 0, 1)
    pcts = solved_curve(np.linspace(0, 1, n))
    # Clamp between 0 and 1 to account for any numerical issues, and convert to 1d array
    pcts = np.ravel(np.clip(pcts, 0, 1))
    return quaternion_slerp(q1, q2, pcts), solved_curve


def bezier_interp(
    n_control_pts: int,
    d0: Optional[float] = None,
    df: Optional[float] = None,
    d20: Optional[float] = None,
    d2f: Optional[float] = None,
    d_max: Optional[float] = None,
    d2_max: Optional[float] = None,
) -> BezierCurve:
    pts = cp.Variable((n_control_pts, 1))
    curve = BezierCurve(pts, 0, 1)
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
    solved_curve = BezierCurve(pts.value, 0, 1)
    return solved_curve


def main():
    from pyastrobee.trajectories.trajectory import Trajectory

    # np.random.seed(0)
    # q1 = np.array([0, 0, 0, 1])
    q1 = random_quaternion()
    q2 = random_quaternion()
    print("angle between two rotations: ", quaternion_dist(q1, q2))
    T = 10
    n = 500
    dt = T / n
    w_max = 0.30
    dw_max = 1
    quats, curve = bezier_slerp(q1, q2, n, T, w_max, dw_max)
    plt.figure()
    plot_1d_bezier_curve(curve, show=False)
    plot_1d_bezier_curve(curve.derivative, show=False)
    plot_1d_bezier_curve(curve.derivative.derivative, show=False)
    plt.show()
    omega = quats_to_angular_velocities(quats, dt)
    print("Max omega: ", np.max(np.linalg.norm(omega, axis=1)))
    alpha = np.gradient(omega, dt, axis=0)
    print("Max alpha: ", np.max(np.linalg.norm(alpha, axis=1)))
    traj = Trajectory(None, quats, None, omega, None, alpha)
    traj.plot()


def main2():
    curve = bezier_interp(20, 10, 10)
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
    main()
    # main2()
