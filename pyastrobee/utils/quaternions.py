"""Quaternion utilities

We will always default to using XYZW convention
"""

from typing import Union

import numpy as np
import numpy.typing as npt
import pytransform3d.rotations as rt
import pytransform3d.batch_rotations as brt

from pyastrobee.utils.math_utils import normalize
from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.rotations import (
    axis_angle_between_two_vectors,
    axis_angle_to_quat,
    quat_to_rmat,
    quat_to_axis_angle,
    compact_axis_angle,
)


def check_quaternion(quat: npt.ArrayLike) -> np.ndarray:
    """Checks that a quaternion is of the correct shape and returns a normalized quat

    Args:
        quat (npt.ArrayLike): Quaternion (XYZW or WXYZ), shape (4,)

    Raises:
        ValueError: If the input is not a valid quaternion

    Returns:
        np.ndarray: Normalized quaternion, shape (4,)
    """
    quat = np.ravel(quat)
    if len(quat) != 4:
        raise ValueError(f"Invalid quaternion ({quat}):\nNot of length 4!")
    return normalize(quat)


def random_quaternion() -> np.ndarray:
    """Generate a random, normalized quaternion

    Returns:
        np.ndarray: XYZW quaternion, shape (4,)
    """
    q = np.random.rand(4)
    return q / np.linalg.norm(q)


def conjugate(quat: npt.ArrayLike) -> np.ndarray:
    """Conjugate of an XYZW quaternion (same scalar part but flipped imaginary components)

    Args:
        quat (npt.ArrayLike): XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: Conjugate XYZW quaternion, shape (4,)
    """
    x, y, z, w = quat
    return np.array([-x, -y, -z, w])


def quats_to_angular_velocities(
    quats: np.ndarray, dt: Union[float, npt.ArrayLike]
) -> np.ndarray:
    """Determines the angular velocities of a sequence of quaternions, for a given sampling time

    Based on AHRS (Attitude and Heading Reference Systems): See ahrs/common/quaternion.py

    Args:
        quats (np.ndarray): Sequence of XYZW quaternions, shape (n, 4)
        dt (Union[float, np.ndarray]): Sampling time(s). If passing in an array of sampling times,
            this must be of length n

    Returns:
        np.ndarray: Angular velocities (wx, wy, wz), shape (n, 3)
    """
    xs = quats[:, 0]
    ys = quats[:, 1]
    zs = quats[:, 2]
    ws = quats[:, 3]
    n = quats.shape[0]  # Number of quaternions

    # This uses a new central differencing method to improve handling at start/end points
    dw = np.zeros((n, 3))
    # Handle the start
    dw[0, :] = np.array(
        [
            ws[0] * xs[1] - xs[0] * ws[1] - ys[0] * zs[1] + zs[0] * ys[1],
            ws[0] * ys[1] + xs[0] * zs[1] - ys[0] * ws[1] - zs[0] * xs[1],
            ws[0] * zs[1] - xs[0] * ys[1] + ys[0] * xs[1] - zs[0] * ws[1],
        ]
    )
    # Handle the end
    dw[-1, :] = np.array(
        [
            ws[-2] * xs[-1] - xs[-2] * ws[-1] - ys[-2] * zs[-1] + zs[-2] * ys[-1],
            ws[-2] * ys[-1] + xs[-2] * zs[-1] - ys[-2] * ws[-1] - zs[-2] * xs[-1],
            ws[-2] * zs[-1] - xs[-2] * ys[-1] + ys[-2] * xs[-1] - zs[-2] * ws[-1],
        ]
    )
    # Handle the middle range of quaternions
    # Multiply by a factor of 1/2 since the central difference covers 2 timesteps
    dw[1:-1, :] = (1 / 2) * np.column_stack(
        [
            ws[:-2] * xs[2:] - xs[:-2] * ws[2:] - ys[:-2] * zs[2:] + zs[:-2] * ys[2:],
            ws[:-2] * ys[2:] + xs[:-2] * zs[2:] - ys[:-2] * ws[2:] - zs[:-2] * xs[2:],
            ws[:-2] * zs[2:] - xs[:-2] * ys[2:] + ys[:-2] * xs[2:] - zs[:-2] * ws[2:],
        ]
    )
    # If dt is scalar, broadcasting is simple. If dt is an array of time deltas, adjust shape for broadcasting
    if np.ndim(dt) == 0:
        return 2.0 * dw / dt
    else:
        if len(dt) != n:
            raise ValueError(
                f"Invalid dt array length: {len(dt)}. Must be of length {n}"
            )
        return 2.0 / (np.reshape(dt, (-1, 1)) * dw)


def xyzw_to_wxyz(quats: npt.ArrayLike) -> np.ndarray:
    """Converts a XYZW quaternion or array of quaternions to WXYZ

    Args:
        quats (npt.ArrayLike): XYZW quaternion(s), shape (4,) or (n, 4)

    Returns:
        np.ndarray: WXYZ quaternions, shape (4,) or (n, 4) (same shape as input)
    """
    quats = np.asarray(quats)
    if quats.shape[-1] != 4:
        raise ValueError("Invalid quaternion array: Must be of shape (4,) or (n, 4)")
    idx = np.array([3, 0, 1, 2])
    if np.ndim(quats) == 1:
        return quats[idx]
    else:
        return quats[:, idx]


def wxyz_to_xyzw(quats: npt.ArrayLike) -> np.ndarray:
    """Converts a WXYZ quaternion or array of quaternions to XYZW

    Args:
        quats (npt.ArrayLike): WXYZ quaternion(s), shape (4,) or (n, 4)

    Returns:
        np.ndarray: XYZW quaternions, shape (4,) or (n, 4) (same shape as input)
    """
    quats = np.asarray(quats)
    if quats.shape[-1] != 4:
        raise ValueError("Invalid quaternion array: Must be of shape (4,) or (n, 4)")
    idx = np.array([1, 2, 3, 0])
    if np.ndim(quats) == 1:
        return quats[idx]
    else:
        return quats[:, idx]


def quaternion_derivative(
    q: Union[Quaternion, npt.ArrayLike], w: npt.ArrayLike
) -> np.ndarray:
    """Quaternion derivative for a given angular velocity

    Based on "A Survey of Attitude Representations", Shuster, eqn. 306

    Args:
        q (Union[Quaternion, npt.ArrayLike]): XYZW quaternion, shape (4,)
        w (npt.ArrayLike): Angular velocity (wx, wy, wz), shape (3,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    if isinstance(q, Quaternion):
        q = q.xyzw
    else:
        q = check_quaternion(q)
    qx, qy, qz, qw = q
    Z = np.array([[qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw], [-qx, -qy, -qz]])
    return (1 / 2) * Z @ w


def quaternion_integration(
    q: Union[Quaternion, npt.ArrayLike], w: npt.ArrayLike, dt: float
) -> np.ndarray:
    """Propagate a quaternion forward one timestep based on the current angular velocity

    Args:
        q (Union[Quaternion, npt.ArrayLike]): Initial XYZW quaternion, shape (4,)
        w (npt.ArrayLike): Angular velocity (wx, wy, wz), shape (3,)
        dt (float): Timestep duration (seconds)

    Returns:
        np.ndarray: Next XYZW quaternion, q(t + dt), shape (4,)
    """
    if isinstance(q, Quaternion):
        q = q.xyzw
    else:
        q = check_quaternion(q)
    return normalize(q + dt * quaternion_derivative(q, w))


def combine_quaternions(
    q1: Union[Quaternion, npt.ArrayLike], q2: Union[Quaternion, npt.ArrayLike]
) -> np.ndarray:
    """Combines the angular representation of two quaternions

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): First XYZW quaternion, shape (4,) if passing in an array
        q2 (Union[Quaternion, npt.ArrayLike]): Second XYZW quaternion, shape (4,) if passing in an array

    Returns:
        np.ndarray: Combined XYZW quaternion, shape (4,)
    """
    if not isinstance(q1, Quaternion):
        q1 = Quaternion(xyzw=q1)
    if not isinstance(q2, Quaternion):
        q2 = Quaternion(xyzw=q2)
    q = Quaternion()
    q.wxyz = rt.concatenate_quaternions(q1.wxyz, q2.wxyz)
    return q.xyzw


def quaternion_between_two_vectors(v1: npt.ArrayLike, v2: npt.ArrayLike) -> np.ndarray:
    """Gives the quaternion rotation that would rotate vector v1 to align with v2 (magnitude-independent)

    Args:
        v1 (npt.ArrayLike): (3,) Starting vector/direction
        v2 (npt.ArrayLike): (3,) Ending vector/direction

    Returns:
        np.ndarray: (4,) XYZW quaternion
    """
    axis, angle = axis_angle_between_two_vectors(v1, v2)
    return axis_angle_to_quat(axis, angle)


def get_closest_heading_quat(q0: npt.ArrayLike, heading: npt.ArrayLike) -> np.ndarray:
    """Gives the quaternion closest to q0 that has its x-axis aligned with the heading

    Args:
        q0 (npt.ArrayLike): Initial (reference) XYZW quaternion, shape (4,)
        heading (npt.ArrayLike): Desired XYZ vector parallel to the new frame's x-axis, shape (3,)

    Returns:
        np.ndarray: XYZW quaternion, shape (4,)
    """
    # We want the x axis to point along the heading axis
    # So, a rotation between these two axes can be defined by an axis-angle rotation
    # We can then apply this rotation transformation via quaternion concatenation
    rmat1 = quat_to_rmat(q0)
    orig_x_axis = rmat1[:, 0]
    rotation_quat = quaternion_between_two_vectors(orig_x_axis, heading)
    return combine_quaternions(rotation_quat, q0)


def quaternion_slerp(
    q1: Union[Quaternion, npt.ArrayLike],
    q2: Union[Quaternion, npt.ArrayLike],
    pct: Union[float, npt.ArrayLike],
) -> np.ndarray:
    """Interpolates between two quaternions via SLERP (spherical linear interpolation)

    To interpolate at multiple points, pass in pcts as an array of interpolation percentages

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Starting quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        q2 (Union[Quaternion, npt.ArrayLike]): Ending quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        pct (Union[float, npt.ArrayLike]): Percent(s) between start -> end, expressed as float(s) in [0, 1]

    Returns:
        np.ndarray: The interpolated XYZW quaternion(s), shape = (4,) or (n, 4) if interpolating at multiple points
    """
    pct = np.atleast_1d(pct)
    n = len(pct)  # Number of interpolation points
    if not (np.all(pct >= 0) and np.all(pct <= 1)):
        raise ValueError(
            f"Interpolation percentage(s) must be between 0 and 1.\nGot: {pct}"
        )
    if not isinstance(q1, Quaternion):
        q1 = Quaternion(xyzw=q1)
    if not isinstance(q2, Quaternion):
        q2 = Quaternion(xyzw=q2)
    # The shortest path parameter does not add too much extra computation and should handle quaternion ambiguity well
    shortest_path = True
    # Simple conversion for one interpolation point, otherwise use batched process
    if n == 1:
        q_interp = Quaternion(
            wxyz=rt.quaternion_slerp(q1.wxyz, q2.wxyz, pct[0], shortest_path)
        )
        return q_interp.xyzw
    else:
        wxyz_quats = brt.quaternion_slerp_batch(q1.wxyz, q2.wxyz, pct, shortest_path)
        xyzw_quats = np.zeros_like(wxyz_quats)
        xyzw_quats[:, :3] = wxyz_quats[:, 1:]  # qx, qy, qz
        xyzw_quats[:, -1] = wxyz_quats[:, 0]  # qw
        return xyzw_quats  # (n, 4)


def quaternion_dist(
    q1: Union[Quaternion, npt.ArrayLike], q2: Union[Quaternion, npt.ArrayLike]
) -> float:
    """Computes the distance between two quaternions

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Either a Quaternion object or
            an array of the XYZW quaternions (length = 4)
        q2 (Union[Quaternion, npt.ArrayLike]): A second quaterion to compare against

    Returns:
        float: Distance between the two quaternions
    """
    if not isinstance(q1, Quaternion):
        q1 = Quaternion(xyzw=q1)
    if not isinstance(q2, Quaternion):
        q2 = Quaternion(xyzw=q2)
    return rt.quaternion_dist(q1.wxyz, q2.wxyz)


def quaternion_diff(
    q1: Union[Quaternion, npt.ArrayLike], q2: Union[Quaternion, npt.ArrayLike]
) -> np.ndarray:
    """Gives the quaternion representing the rotation from q1 -> q2

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Starting XYZW quaternion, shape (4,)
        q2 (Union[Quaternion, npt.ArrayLike]): Ending XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: XYZW quaternion, shape (4,)
    """
    return combine_quaternions(q2, conjugate(q1))


def quaternion_angular_diff(
    q1: Union[Quaternion, npt.ArrayLike], q2: Union[Quaternion, npt.ArrayLike]
) -> np.ndarray:
    """Gives a world-frame compact-axis-angle form of the angular error between two quaternions (q1 -> q2)

    - This is similar (but not the same) as a difference between fixed-XYZ conventions
      (for small angles, these are very close).
    - Differences between Euler/Fixed angle sets often do not represent true rotations

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Starting XYZW quaternion, shape (4,)
        q2 (Union[Quaternion, npt.ArrayLike]): Ending XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: XYZW quaternion, shape (4,)
    """
    if isinstance(q1, Quaternion):
        q1 = q1.xyzw
    else:
        q1 = check_quaternion(q1)
    if isinstance(q2, Quaternion):
        q2 = q2.xyzw
    else:
        q2 = check_quaternion(q2)
    # These lines are based on the math from pytransform3d's quaternion_gradient(),
    # but simplified to work with just 2 quats and no time info
    q3 = quaternion_diff(q1, q2)
    axis, angle = quat_to_axis_angle(q3)
    return compact_axis_angle(axis, angle)
