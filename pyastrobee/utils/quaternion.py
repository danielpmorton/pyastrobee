"""Class for managing quaternion conventions

Usage examples:
q = Quaternion() # This will initialize it as empty
q.xyzw = [0.1, 0.2, 0.3, 0.4] # This will assign the values after initialization
q = Quaternion(xyzw=[0.1, 0.2, 0.3, 0.4]) # This will assign values at initialization
some_pytransform3d_function(q.wxyz) # Pass the wxyz data into modules that use this convention
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from pyastrobee.utils.math_utils import normalize


class Quaternion:
    """Quaternion class to handle the XYZW/WXYZ conventions with less confusion

    We will always default to using XYZW convention

    Args:
        xyzw (npt.ArrayLike, optional): Quaternions, in XYZW order. Defaults to None, in which
            case wxyz should be provided, or else the quaternion will be empty
        wxyz (npt.ArrayLike, optional): Quaternions, in WXYZ order. Defaults to None, in which
            case xyzw should be provided, or else the quaternion will be empty

    Raises:
        ValueError: if both XYZW and WXYZ inputs are provided at instantiation
    """

    def __init__(
        self, xyzw: Optional[npt.ArrayLike] = None, wxyz: Optional[npt.ArrayLike] = None
    ):
        if xyzw is not None and wxyz is not None:
            raise ValueError("Specify one of XYZW/WXYZ, not both")
        elif xyzw is not None:
            self.xyzw = xyzw
        elif wxyz is not None:
            self.wxyz = wxyz
        else:
            self._initialize_as_empty()

    def _check_if_loaded(self):
        vals = [self.x, self.y, self.z, self.w]
        if any(val is None for val in vals):
            raise ValueError(
                f"Quaternion has been initialized, but not set (value is {vals})"
            )

    def _initialize_as_empty(self):
        self.x, self.y, self.z, self.w = [None, None, None, None]

    # TODO need to decide if this is needed
    def normalize(self):
        self.xyzw = normalize(self.xyzw)

    @property
    def xyzw(self):
        """Quaternion expressed in XYZW format. Shape = (4,)"""
        self._check_if_loaded()
        return np.array([self.x, self.y, self.z, self.w])

    @property
    def wxyz(self):
        """Quaternion expressed in WXYZ format. Shape = (4,)"""
        self._check_if_loaded()
        return np.array([self.w, self.x, self.y, self.z])

    @xyzw.setter
    def xyzw(self, xyzw: npt.ArrayLike):
        """Sets the quaternion based on an array in XYZW form"""
        xyzw = check_quaternion(xyzw)
        self.x, self.y, self.z, self.w = xyzw

    @wxyz.setter
    def wxyz(self, wxyz: npt.ArrayLike):
        """Sets the quaternion based on an array in WXYZ form"""
        wxyz = check_quaternion(wxyz)
        self.w, self.x, self.y, self.z = wxyz

    @property
    def conjugate(self) -> np.ndarray:
        """Conjugate of the XYZW quaternion"""
        return conjugate(self.xyzw)


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
            this must be of length (n-1)

    Returns:
        np.ndarray: Angular velocities (wx, wy, wz), shape (n-1, 3)
    """
    xs = quats[:, 0]
    ys = quats[:, 1]
    zs = quats[:, 2]
    ws = quats[:, 3]
    dw = np.column_stack(
        [
            ws[:-1] * xs[1:] - xs[:-1] * ws[1:] - ys[:-1] * zs[1:] + zs[:-1] * ys[1:],
            ws[:-1] * ys[1:] + xs[:-1] * zs[1:] - ys[:-1] * ws[1:] - zs[:-1] * xs[1:],
            ws[:-1] * zs[1:] - xs[:-1] * ys[1:] + ys[:-1] * xs[1:] - zs[:-1] * ws[1:],
        ]
    )
    # If dt is scalar, broadcasting is simple. If dt is an array of time deltas,
    # need to adjust the shape for broadcasting
    if np.ndim(dt) == 0:
        return 2.0 * dw / dt
    else:
        if len(dt) != dw.shape[0]:
            raise ValueError(
                "Invalid dt array size. This needs to be of length (n-1), if we are working with n quaternions"
            )
        return 2.0 / np.reshape(dt, (-1, 1)) * dw
