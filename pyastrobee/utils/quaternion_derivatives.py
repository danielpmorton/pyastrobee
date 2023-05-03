"""Relationships between quaternions and angular velocity

Our quaternion / angular velocity conventions (currently) are:
- XYZW quaternions
- World-frame angular velocities

The matrices used by other sources differ because they might be using WXYZ quaternions or
they might define angular velocity in the body-fixed frame. For instance, Shuster uses XYZW
quaternions but defines angular velocity in body frame, and Khatib uses WXYZ quaternions
with angular velocities in world frame

In case we decide to use body-frame angular velocities in the future, the relevant equations
are included below (with the global frame equations as well for reference)
"""

import numpy as np
import numpy.typing as npt


def world_frame_quat_deriv(q: npt.ArrayLike, omega_world: npt.ArrayLike) -> np.ndarray:
    """Quaternion derivative for a rotating body with a known WORLD-FRAME angular velocity

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        omega_world (npt.ArrayLike): World-frame angular velocity (wx, wy, wz), shape (3,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    x, y, z, w = q
    GT = np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])
    return (1 / 2) * GT @ omega_world


def body_frame_quat_deriv(q: npt.ArrayLike, omega_body: npt.ArrayLike) -> np.ndarray:
    """Quaternion derivative for a rotating body with a known BODY-FRAME angular velocity

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        omega_body (npt.ArrayLike): Body-frame angular velocity (w1, w2, w3), shape (3,)

    Returns:
        np.ndarray: Quaternion derivative, shape (4,)
    """
    x, y, z, w = q
    LT = np.array([[w, -z, y], [z, w, -x], [-y, x, w], [-x, -y, -z]])
    return (1 / 2) * LT @ omega_body


def world_frame_angular_error(q: npt.ArrayLike, q_des: npt.ArrayLike) -> np.ndarray:
    """Angular error vector between two orientations, defined in WORLD frame

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        q_des (npt.ArrayLike): Desired XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: Angular error, shape (3,)
    """
    x, y, z, w = q
    return 2 * np.array([[-w, z, -y, x], [-z, -w, x, y], [y, -x, -w, z]]) @ q_des


def body_frame_angular_error(q: npt.ArrayLike, q_des: npt.ArrayLike) -> np.ndarray:
    """Angular error vector between two orientations, defined in BODY frame

    Args:
        q (npt.ArrayLike): Current XYZW quaternion, shape (4,)
        q_des (npt.ArrayLike): Desired XYZW quaternion, shape (4,)

    Returns:
        np.ndarray: Angular error, shape (3,)
    """
    xd, yd, zd, wd = q_des
    return (
        2 * np.array([[wd, zd, -yd, -xd], [-zd, wd, xd, -yd], [yd, -xd, wd, -zd]]) @ q
    )
