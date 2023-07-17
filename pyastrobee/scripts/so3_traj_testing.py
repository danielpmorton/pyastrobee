"""
NOTES:
- Stephen's idea of using a difference reference matrix for the Cayley transformation doesn't seem to be invertible?
  - If we pass in a skew matrix with some rotation matrix as a reference, the output WILL NOT be S.O.
  - If we pass in a S.O. matrix with a S.O. matrix as a reference, the output WILL be skew
  - (this could be an issue when we are mapping back from the tangent space to rotations)
- Originally I thought we could map the rotation to the tangent (skew) space and then create 3D bezier curves in
  the "unskewed" vector space, but ....... TODO

  maybe some confusion is happening because I assumed that the tangent space vector components directly relate
  to the rvec parmaeters? But it might be pretty different

It seems like I should probably just be working in the tangent space of quaternions because the derivatives might be a
lot simpler (linear with respect to the angular velocity vector) -- the angular velocity relationship with derivatives
of rvecs is complicated and includes some weird (nonconvex) cross products and division between terms

I also need to actually figure out the relationship between rodrigues rotation vectors and the skew symmetric matrices
-- right now it just seems like entries are "relatively close" but sometimes have a sign flip or a 2x scaling issue
"""


from typing import Optional

import numpy as np
import numpy.typing as npt
import cvxpy as cp
import cv2
import matplotlib.pyplot as plt

from pyastrobee.utils.math_utils import skew, normalize, is_skew, is_special_orthogonal
from pyastrobee.utils.quaternions import (
    quaternion_angular_error,
    random_quaternion,
    quaternion_slerp,
    quats_to_angular_velocities,
    quaternion_integration,
)
from pyastrobee.utils.rotations import quat_to_rmat, rmat_to_quat
from pyastrobee.trajectories.bezier import BezierCurve
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


def cayley(A: npt.ArrayLike, center: Optional[npt.ArrayLike] = None) -> np.ndarray:
    """Cayley transform: Mapping between skew-symmetric matrices and SO(n)

    - This function is an involution so the skew->SO(n) mapping function is the same as SO(n)->skew
    - TODO check that the "reference matrix" concept actually works - Stephen Boyd seems to think so
      ^^ It seems that it "works" only moving from SO(n)->skew?

    Updated version of the Cayley transform is from:
    http://imar.ro/journals/Revue_Mathematique/pdfs/2018/2/5.pdf
    Assume that we are working with the real case, so we don't need to worry about conjugates

    Args:
        A (npt.ArrayLike): Either a skew-symmetrix matrix or a special orthogonal matrix, shape (n, n)
        ref (Optional[npt.ArrayLike]): Reference matrix for the transformation. Defaults to None, in which case
            we use the standard reference of the (n x n) identity

    Returns:
        np.ndarray: Skew-symmetric matrix or special orthogonal matrix, depending on the input. Shape (n, n)
    """
    # TODO should we check that a matrix is actually skew or in SO(3)?
    # TODO need to figure out if there should be a factor of 2 here
    # TODO should we check that the reference matrix is special orthogonal?
    A = np.asarray(A)
    m, n = A.shape
    if m != n:
        raise ValueError(f"Input must be a square matrix! Got shape: {(m, n)}")
    In = np.eye(n)
    if center is None:
        center = In
    else:
        center = np.asarray(center)
        if center.shape != A.shape:
            raise ValueError("Reference must have the same shape as the input!")
    return (In - center.T @ A) @ np.linalg.inv(center + A)


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


def traj_gen_3(
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
    # NEW IDEA
    # since derivatives of the rvec curve do not represent angular velocities,
    # numerically integrate the rotations with the known initial ang vel / ang accel
    # and use this for derivative information

    # TODO there is way too much conversion happening here between quaternions and such
    # Need to create equivalent functions for rotation matrices

    times = np.linspace(t0, tf, n_timesteps, endpoint=True)
    dt = times[1] - times[0]  # Do this differently

    # Rmid = midpoint_rmat(R0, Rf)
    Rref = np.eye(3)
    # Note: cayley + unskewing is probably the same as using rodrigues's formula
    # But, rodrigues doesn't lend itself well to using another (non-identity) reference point
    S0 = cayley(R0, Rref)
    Sf = cayley(Rf, Rref)
    r0 = unskew(S0)
    rf = unskew(Sf)

    # if only w is given: integrate one point
    # if w and a are given: integrate two points
    # TODO clean up these assertions and conversions
    assert w0 is not None
    assert wf is not None
    assert a0 is not None
    assert af is not None
    w0 = np.asarray(w0)
    wf = np.asarray(wf)
    a0 = np.asarray(a0)
    af = np.asarray(af)

    q0 = rmat_to_quat(R0)
    qf = rmat_to_quat(Rf)
    q1 = quaternion_integration(q0, w0, dt)
    w1 = w0 + a0 * dt
    q2 = quaternion_integration(q1, w1, dt)
    R1 = quat_to_rmat(q1)
    R2 = quat_to_rmat(q2)
    r1 = rmat_to_rvec(R1)
    r2 = rmat_to_rvec(R2)
    qfminus1 = quaternion_integration(qf, -wf, dt)
    wfminus1 = wf - af * dt
    qfminus2 = quaternion_integration(qfminus1, -wfminus1, dt)
    Rfminus1 = quat_to_rmat(qfminus1)
    Rfminus2 = quat_to_rmat(qfminus2)
    rfminus1 = rmat_to_rvec(Rfminus1)
    rfminus2 = rmat_to_rvec(Rfminus2)
    # find the derivatives in rvec space based on these integrated values
    dr0 = (r1 - r0) / dt
    dr1 = (r2 - r1) / dt
    ddr0 = (dr1 - dr0) / dt
    drf = (rf - rfminus1) / dt
    drfminus1 = (rfminus1 - rfminus2) / dt
    ddrf = (drf - drfminus1) / dt

    r_pts = cp.Variable((n_control_pts, 3))
    r_curve = BezierCurve(r_pts, t0, tf)
    dr_curve = r_curve.derivative
    dr_pts = dr_curve.points
    d2r_curve = dr_curve.derivative
    d2r_pts = d2r_curve.points
    d3r_curve = d2r_curve.derivative
    d3r_pts = d3r_curve.points
    constraints = [
        r_pts[0] == r0,
        r_pts[-1] == rf,
        dr_pts[0] == dr0,
        dr_pts[-1] == drf,
        d2r_pts[0] == ddr0,
        d2r_pts[-1] == ddrf,
    ]

    # TODO: see if a min-jerk formulation of the third derivative on the tangent space is representative
    # of a min-jerk trajectory on the manifold

    # TODO also see if the rotation vector is equivalent to the tangent space!!!

    objective = cp.Minimize(d3r_curve.l2_squared)
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


# Assorted testing functions below
# TODO turn these into proper test cases in the tests/ folder


def test_rmat_rvec_conversion():
    # Use quaternions as a reference conversion point
    q = random_quaternion()
    R = quat_to_rmat(q)
    r = rmat_to_rvec(R)
    R_test = rvec_to_rmat(r)
    np.testing.assert_array_almost_equal(R, R_test)
    np.testing.assert_array_almost_equal(q, rmat_to_quat(R_test))
    print("Test passed")


def test_rvec_tangent_space_comparison():
    # Test with an arbitray rotation which may not be "small"
    np.random.seed(1)
    q1 = random_quaternion()
    R1 = quat_to_rmat(q1)
    r1 = rmat_to_rvec(R1)
    S1 = cayley(R1)
    s1 = unskew(S1)
    print(f"Rodrigues vec: {r1}")
    print(f"Cayley tangent space vec: {s1}")
    # Generate a small rotation near the origin to see if cayley and rodrigues agree
    q2 = normalize(np.array([0, 0, 0, 1]) + 0.001 * np.random.rand(4))
    R2 = quat_to_rmat(q2)
    r2 = rmat_to_rvec(R2)
    S2 = cayley(R2)
    s2 = unskew(S2)
    print("Small rotation experiment:")
    print(f"Rodrigues: {r1}")
    print(f"Cayley: {s1}")


def test_cayley():
    # Test transformation starting with an arbitrary matrix in SO(3)
    q = random_quaternion()
    R = quat_to_rmat(q)
    S = cayley(R)
    assert is_skew(S)
    # Test invertibility of the transformation
    np.testing.assert_array_almost_equal(R, cayley(cayley(R)))
    np.testing.assert_array_almost_equal(S, cayley(cayley(S)))
    # Test transformation starting from an arbitrary skew-symmetric matrix
    v = np.random.rand(3)
    S2 = skew(v)
    R2 = cayley(S2)
    assert is_special_orthogonal(R2)
    # Test invertibility with a different center
    # Note: using the transpose for the inverse mapping
    np.testing.assert_array_almost_equal(R, cayley(cayley(R, R2), R2.T))
    np.testing.assert_array_almost_equal(S, cayley(cayley(S, R2), R2.T))
    # Try out the cayley transform from a different reference matrix, and check invertibility
    # Note: in general, cayley(some_skew_matrix, some_center) will NOT produce a special orthogonal matrix
    # for any center other than the identity, BUT we can still invert a transform even though this is the case
    # e.g. in general, R, cayley(R, center) is NOT skew and cayley(S, center) is NOT S.O.
    # BUT the output of these functions can be passed back into cayley with the center transposed and return the original value
    # unless the skew matric was constructed using that center

    R3 = cayley(S2, center=R2)
    assert is_special_orthogonal(R3)
    S3 = cayley(R3, center=R2)
    assert is_skew(S3)
    assert np.allclose(S2, S3)
    print("Test passed")


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
    # qs, ws, dws, times = traj_gen(
    #     R0, Rf, t0, tf, n_control_pts, n_timesteps, w0, wf, a0, af
    # )
    qs, ws, dws, times = traj_gen_3(
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

    # test_traj_gen()
    # test_rvec_tangent_space_comparison()
    # test_rmat_rvec_conversion()
    test_cayley()
