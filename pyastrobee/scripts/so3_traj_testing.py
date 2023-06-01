from typing import Optional

import numpy as np
import numpy.typing as npt
import cvxpy as cp
import cv2
import matplotlib.pyplot as plt

from pyastrobee.utils.math_utils import skew, normalize
from pyastrobee.utils.quaternions import (
    quaternion_angular_error,
    random_quaternion,
    quaternion_slerp,
    quats_to_angular_velocities,
)
from pyastrobee.utils.rotations import quat_to_rmat, rmat_to_quat
from pyastrobee.trajectories.bezier_and_bernstein import BezierCurve
from pyastrobee.trajectories.trajectory import Trajectory


def rmat_to_rvec(rmat):
    angle = np.arccos((rmat[0, 0] + rmat[1, 1] + rmat[2, 2] - 1) / 2)
    if np.abs(np.sin(angle)) < 1e-14:
        raise ZeroDivisionError("Cannot convert to axis-angle: Near singularity")
    axis = (1 / (2 * np.sin(angle))) * np.array(
        [rmat[2, 1] - rmat[1, 2], rmat[0, 2] - rmat[2, 0], rmat[1, 0] - rmat[0, 1]]
    )
    return axis * angle


def rvec_to_rmat(r):
    theta = np.linalg.norm(r)
    if theta > 1e-30:
        n = r / theta
        Sn = skew(n)
        return np.eye(3) + np.sin(theta) * Sn + (1 - np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = skew(r)
        theta2 = theta**2
        return (
            np.eye(3) + (1 - theta2 / 6.0) * Sr + (0.5 - theta2 / 24.0) * np.dot(Sr, Sr)
        )


def rmat_to_rvec_2(R):
    # using pytransform
    cos_angle = (np.trace(R) - 1.0) / 2.0
    angle = np.arccos(min(max(-1.0, cos_angle), 1.0))
    if angle == 0.0:  # R == np.eye(3)
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = np.empty(4)
    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )

    if abs(angle - np.pi) < 1e-4:  # np.trace(R) close to -1
        # The threshold 1e-4 is a result from this discussion:
        # https://github.com/dfki-ric/pytransform3d/issues/43
        # The standard formula becomes numerically unstable, however,
        # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
        # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
        # squared values of the rotation axis on the diagonal of this matrix.
        # We can still use the original formula to reconstruct the signs of
        # the rotation axis correctly.

        # In case of floating point inaccuracies:
        R_diag = np.clip(np.diag(R), -1.0, 1.0)

        eeT_diag = 0.5 * (R_diag + 1.0)
        signs = np.sign(axis_unnormalized)
        signs[signs == 0.0] = 1.0
        a[:3] = np.sqrt(eeT_diag) * signs
    else:
        a[:3] = axis_unnormalized
        # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
        # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
        # but the following is much more precise for angles close to 0 or pi:
    a[:3] /= np.linalg.norm(a[:3])

    a[3] = angle
    # return a
    return a[:3] * a[3]


def rmat_to_rvec_3(R):
    return np.ravel(cv2.Rodrigues(R)[0])


def rmat_angular_error(R, R_des):
    return rmat_to_rvec(R @ R_des.T)


def cayley(A, ref=None):
    # A should either be an orthogonal matrix or skew-symmetric
    # The function is an involution so the skew->orthogonal mapping is the same as orthogonal->skew
    n = A.shape[0]
    if ref is None:
        ref = np.eye(n)
    return (ref - A) @ np.linalg.inv(ref + A)


def rmat_derivative(R, w):
    # Assume angular velocity defined in world frame
    # If angular velocity defined in base frame, the multiplication order flips
    return R @ skew(w)


# def ang_vel_from_rmats(R0, Rf, dt):
#     return (R0.T @ Rf) / dt  # NOT CORRECT!!


# TODO see if there is a way to directly do this in rotation matrix space
def midpoint_rmat(R0, Rf):

    q_mid = quaternion_slerp(rmat_to_quat(R0), rmat_to_quat(Rf), 0.5)
    return quat_to_rmat(q_mid)


def rmat_integration(ws, R0):
    n = ws.shape[0]
    # HOW TO DEAL WITH n+1 vs n??
    Rs = np.zeros((n + 1, 3, 3))
    Rs[0] = R0
    for i, w in enumerate(ws):
        # THIS DOES NOT LEAD TO A VALID ROTATION MATRIX
        # technically, skew(w) @ R is the DERIVATIVE, not the integrated rmat
        # So, need to include a time term. But, still does not guarantee orthogonality
        Rs[i + 1] = skew(w) @ Rs[i]
    return Rs


def traj_gen(
    R0: np.ndarray,
    Rf: np.ndarray,
    t0: float,
    tf: float,
    n_control_pts: int,
    n_timesteps: int,
    w0: Optional[npt.ArrayLike] = None,
    wf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
):

    w_pts = cp.Variable((n_control_pts, 3))
    w_curve = BezierCurve(w_pts, t0, tf)
    dw_curve = w_curve.derivative
    dw_pts = dw_curve.points
    ddw_curve = dw_curve.derivative
    ddw_pts = ddw_curve.points
    # Optimize in the lie algebra
    times = np.linspace(t0, tf, n_timesteps, endpoint=True)
    dt = times[1] - times[0]
    ws = cp.Variable((n_timesteps, 3))
    dw = rmat_angular_error(R0, Rf)
    # Where does cayley come in??
    # Where does the sequential aspect come in?
    # Maybe patch together splines to rotation midpoints until we are within some tolerance?
    R_mid = midpoint_rmat(R0, Rf)

    constraints = [
        cp.sum(w_curve(times) * dt, axis=0) == dw,  # CHECK THIS
        w_pts[0] == w0,
        w_pts[-1] == wf,
        dw_pts[0] == a0,
        dw_pts[-1] == af,
    ]
    objective = cp.Minimize(ddw_curve.l2_squared)  # Does this make sense?
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    solved_w_curve = BezierCurve(w_pts.value, t0, tf)
    solved_dw_curve = solved_w_curve.derivative
    ws = solved_w_curve(times)
    dws = solved_dw_curve(times)
    Rs = rmat_integration(ws, R0)
    qs = np.array([rmat_to_quat(R) for R in Rs])
    return qs, ws, dws, times


def unskew(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# THIS MAKES NICE CURVES BUT THE PROBLEM IS THAT A DERIVATIVE OF A ROTATION VECTOR CURVE
# DOES NOT YIELD AN ANGULAR VELOCITY CURVE
def traj_gen_2(
    R0: np.ndarray,
    Rf: np.ndarray,
    t0: float,
    tf: float,
    n_control_pts: int,
    n_timesteps: int,
    w0: Optional[npt.ArrayLike] = None,
    wf: Optional[npt.ArrayLike] = None,
    a0: Optional[npt.ArrayLike] = None,
    af: Optional[npt.ArrayLike] = None,
):
    # Try something new:
    # Instead of optimizing in the space of angular velocities,
    # optimize along a curve in the lie algebra
    # Use cayley to map the R0 and Rf values to the algebra
    # Create a bezier curve in the rotation algebra space
    # Derivative of the curve will also be in the rotation algebra BUT check to see if this
    # directly translates to the angular velocity
    Rmid = midpoint_rmat(R0, Rf)
    Rref = np.eye(3)
    # Note: cayley + unskewing is probably the same as using rodrigues's formula
    # But, rodrigues doesn't lend itself well to using another (non-identity) reference point
    S0 = cayley(R0, Rref)
    Sf = cayley(Rf, Rref)
    r0 = unskew(S0)
    rf = unskew(Sf)
    r_pts = cp.Variable((n_control_pts, 3))
    r_curve = BezierCurve(r_pts, t0, tf)
    w_curve = r_curve.derivative
    w_pts = w_curve.points
    dw_curve = w_curve.derivative
    dw_pts = dw_curve.points
    ddw_curve = dw_curve.derivative
    ddw_pts = ddw_curve.points
    constraints = [r_pts[0] == r0, r_pts[-1] == rf]
    if w0 is not None:
        constraints.append(w_pts[0] == w0)
    if wf is not None:
        constraints.append(w_pts[-1] == wf)
    if a0 is not None:
        constraints.append(dw_pts[0] == a0)
    if af is not None:
        constraints.append(dw_pts[-1] == af)

    times = np.linspace(t0, tf, n_timesteps, endpoint=True)
    objective = cp.Minimize(ddw_curve.l2_squared)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    solved_r_curve = BezierCurve(r_pts.value, t0, tf)
    solved_w_curve = solved_r_curve.derivative
    solved_dw_curve = solved_w_curve.derivative
    rs = solved_r_curve(times)
    ws = solved_w_curve(times)
    dws = solved_dw_curve(times)
    Rs = np.array([cayley(skew(r)) for r in rs])
    qs = np.array([rmat_to_quat(R) for R in Rs])
    return qs, ws, dws, times


def plot_orn_traj(qs, ws, dws, times):
    traj = Trajectory(None, qs, None, ws, None, dws, times)
    traj.plot()


def test_traj_gen():
    q0 = random_quaternion()
    qf = normalize(q0 + 0.2 * np.random.rand(4))
    # qf = random_quaternion()
    R0 = quat_to_rmat(q0)
    Rf = quat_to_rmat(qf)
    t0 = 0
    tf = 5
    n_control_pts = 8
    n_timesteps = 50
    w0 = [0.1, 0.2, 0.3]
    wf = [0.2, 0.3, 0.4]
    a0 = [0.1, 0.2, 0.3]
    af = [0.2, 0.3, 0.4]
    # qs, ws, dws, times = traj_gen_2(
    #     R0, Rf, t0, tf, n_control_pts, n_timesteps, w0, wf, a0, af
    # )
    qs, ws, dws, times = traj_gen(
        R0, Rf, t0, tf, n_control_pts, n_timesteps, w0, wf, a0, af
    )
    # debugging the plot
    dt = times[1] - times[0]
    ws2 = quats_to_angular_velocities(qs, dt)
    dws2 = np.gradient(ws2, dt, axis=0)
    plt.subplot(1, 2, 1)
    plt.plot(range(ws.shape[0]), ws, "--", label="curve")
    plt.plot(range(ws.shape[0]), ws2, label="new")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(dws.shape[0]), dws, "--", label="curve")
    plt.plot(range(dws.shape[0]), dws2, label="new")
    plt.legend()
    plt.show()

    print(f"Original q0: {q0}")
    print(f"Original qf: {qf}")
    print(f"Traj q0: {qs[0]}")
    print(f"Traj qf: {qs[-1]}")
    print(f"Error between q0s: {quaternion_angular_error(qs[0], q0)}")
    print(f"Error between qfs: {quaternion_angular_error(qs[-1], qf)}")
    plot_orn_traj(qs, ws, dws, times)


if __name__ == "__main__":

    # for i in range(1000):
    #     q1 = random_quaternion()
    #     q2 = random_quaternion()
    #     R1 = quat_to_rmat(q1)
    #     R2 = quat_to_rmat(q2)
    #     q_err = quaternion_angular_error(q1, q2)
    #     r_err = np.ravel(cv2.Rodrigues(R1 @ R2.T)[0])
    #     np.testing.assert_array_almost_equal(q_err, r_err, decimal=1)
    #     print(i)

    # np.random.seed(0)

    # q = random_quaternion()
    # q_des = q + 0.5 * np.random.rand(4)
    # R = quat_to_rmat(q)
    # R_des = quat_to_rmat(q_des)
    # R_delta = R @ R_des.T
    # q_err = quaternion_angular_error(q, q_des)
    # r_err = np.ravel(cv2.Rodrigues(R_delta)[0])
    # print(q_err)
    # print(r_err)
    # print(rmat_to_rvec(R_delta))
    # print(rmat_to_rvec_2(R_delta))

    # print(R_delta - rvec_to_rmat(r_err))

    test_traj_gen()
