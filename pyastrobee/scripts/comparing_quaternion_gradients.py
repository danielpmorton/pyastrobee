"""Comparing quaternion methods against a known dataset to make sure the math is correct

NOTE: The IMUs record angular velocity in BODY frame (this makes sense, the IMU is installed
on the body of whatever object is being tracked). However, it means that we can't use our original
quaternion-to-angular-velocities function because those angular velocities are defined in WORLD. 

Based on https://mariogc.com/post/angular-velocity-quaternions/
RepoIMU: https://github.com/agnieszkaszczesna/RepoIMU/tree/main
"""

from typing import Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.utils.quaternions import wxyz_to_xyzw


# I copied this over from where I defined it in the quaternions.py file
# We aren't using it (right now) since we use the world frame ang vel
# But, this reference data uses body-frame, so we can use it here
def quats_to_body_frame_ang_vels(
    quats: np.ndarray, dt: Union[float, npt.ArrayLike]
) -> np.ndarray:
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


if __name__ == "__main__":
    # This data file seemed to have a nice set of values with good variation and signal/noise ratio
    # This path will need to be modified if this script is replicated on another machine
    data_file = "/home/dan/software/RepoIMU/TStick/TStick_Test07_Trial1.csv"
    # Load the data from the file
    array = np.loadtxt(data_file, dtype=float, delimiter=";", skiprows=2)
    times = array[:, 0]
    quaternions = array[:, 1:5]
    gyroscopes = array[:, 8:11]

    dt = 0.01  # Constant, observed based on the reference data
    angvels = quats_to_body_frame_ang_vels(wxyz_to_xyzw(quaternions), dt)

    # Plotting each method against the ground truth info from the dataset
    colors = ["red", "green", "blue"]
    components = ["wx", "wy", "wz"]
    for i in range(3):
        plt.plot(
            times,
            gyroscopes[:, i],
            label=f"{components[i]}: Truth",
            color=colors[i],
        )
        plt.plot(
            times,
            angvels[:, i],
            label=f"{components[i]}: Mine",
            color=colors[i],
            linestyle="dotted",
        )
    # These limits made it easier to compare the data in a "good" area of the plot
    # Can always zoom out and view more areas
    plt.xlim(9, 13)
    plt.ylim(-5, 4)
    plt.legend()
    plt.show()
