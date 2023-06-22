"""Comparing quaternion methods against a known dataset to make sure the math is correct

NOTE: The IMUs record angular velocity in BODY frame (this makes sense, the IMU is installed
on the body of whatever object is being tracked). However, it means that we can't use our original
quaternion-to-angular-velocities function because those angular velocities are defined in WORLD. 

Based on https://mariogc.com/post/angular-velocity-quaternions/
RepoIMU: https://github.com/agnieszkaszczesna/RepoIMU/tree/main
"""

import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.utils.quaternions import wxyz_to_xyzw
from pyastrobee.utils.quaternion_derivatives import body_frame_angular_velocities

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
    angvels = body_frame_angular_velocities(wxyz_to_xyzw(quaternions), dt)

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
