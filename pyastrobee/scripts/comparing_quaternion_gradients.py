"""Comparing quaternion methods against a known dataset to make sure the math is correct

Based on https://mariogc.com/post/angular-velocity-quaternions/
RepoIMU: https://github.com/agnieszkaszczesna/RepoIMU/tree/main
"""

import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as rt


# Method from the cited website
# I also tested this against AHRS and it seems to match up
def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array(
        [
            q1[0] * q2[1] - q1[1] * q2[0] - q1[2] * q2[3] + q1[3] * q2[2],
            q1[0] * q2[2] + q1[1] * q2[3] - q1[2] * q2[0] - q1[3] * q2[1],
            q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1] - q1[3] * q2[0],
        ]
    )


# This data file seemed to have a nice set of values with good variation and signal/noise ratio
# This path will need to be modified if this script is replicated on another machine
data_file = "/home/dan/software/RepoIMU/TStick/TStick_Test07_Trial1.csv"

# This is the method from the cited page
array = np.loadtxt(data_file, dtype=float, delimiter=";", skiprows=2)
times = array[:, 0]
quaternions = array[:, 1:5]
gyroscopes = array[:, 8:11]
angvel = np.zeros_like(gyroscopes)
for i in range(1, len(angvel)):
    dt = times[i] - times[i - 1]
    angvel[i] = angular_velocities(quaternions[i - 1], quaternions[i], dt)

# Testing with pytransforms
dt = 0.01  # Constant, observed based on the reference data
angvel_2 = rt.quaternion_gradient(quaternions, 0.01)

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
        angvel[:, i],
        label=f"{components[i]}: Theirs",
        color=colors[i],
        linestyle="dashed",
    )
    plt.plot(
        times,
        angvel_2[:, i],
        label=f"{components[i]}: Pytransform",
        color=colors[i],
        linestyle="dotted",
    )
# These limits made it easier to compare the data in a "good" area of the plot
# Can always zoom out and view more areas
plt.xlim(9, 13)
plt.ylim(-5, 4)
plt.legend()
plt.show()
