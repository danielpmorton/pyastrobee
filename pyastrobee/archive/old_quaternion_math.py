"""Deprecated quaternion math, but still may be useful as a reference"""

from typing import Union

import numpy as np
import numpy.typing as npt


# This function was deprecated and replaced by the wrapper around pytransform3d's version
# Pytransform uses central differencing so it's symmetric, and shape (n, 3) instead of (n-1, 3)
# This way, we don't need to do any extra handling at the first timestep or last
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
