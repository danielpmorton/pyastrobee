# TEST SCRIPT
# mainly want to see if we can relax a constraint on a 5th order polynomial and then use the nullspace of the matrix
# to try to improve the overall path of the trajectory


from typing import Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.control.trajectory import Trajectory
from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.quaternions import (
    random_quaternion,
    quaternion_slerp,
    quats_to_angular_velocities,
)
from pyastrobee.control.quaternion_bc_planning import quaternion_interpolation_with_bcs


def main():
    t0 = 0
    tf = 5
    x0 = 1
    xf = 2
    v0 = 0.1
    vf = 0.2
    a0 = -0.2
    # af = -0.1 # LEAVE THIS UNCONSTRAINED
    n = 50

    A = np.array(
        [
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
            [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
            [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
            # [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
        ]
    )
    b = np.array([x0, xf, v0, vf, a0])  # , af])

    # Have Ax=b, but A is R5x6, x is R6, b = R5
    # So we have one degree of freedom

    A_bar = A.T @ np.linalg.inv(A @ A.T)  # Right inverse
    N = np.eye(6) - A_bar @ A

    # Form a linear discretization in time
    times = np.linspace(t0, tf, n, endpoint=True)

    # Set up figure
    fig = plt.figure()

    # Generate a bunch of vectors in the nullspace to modify the parameters of the polynomial
    n_param_sets = 20
    # Set up matrices for storing the data
    params = np.zeros((n_param_sets, 6))
    delta_params = np.zeros((n_param_sets, 6))
    fs = np.zeros((n_param_sets, len(times)))
    dfs = np.zeros((n_param_sets, len(times)))
    d2fs = np.zeros((n_param_sets, len(times)))

    # Can this be fully vectorized???
    for i in range(n_param_sets):
        delta_params[i] = N @ np.random.rand(6)
        # Solve the system using our generalized inverse
        params[i] = A_bar @ b + delta_params[i]
        # Use this to calculate the polynomial based on the coefficients
        f = params[i].T @ np.row_stack(
            [np.ones(n), times, times**2, times**3, times**4, times**5]
        )
        # Also calculate the derivatives of the function (for instance, velocity/acceleration)
        df = params[i].T @ np.row_stack(
            [
                np.zeros(n),
                np.ones(n),
                2 * times,
                3 * times**2,
                4 * times**3,
                5 * times**4,
            ]
        )
        d2f = params[i].T @ np.row_stack(
            [
                np.zeros((2, n)),
                2 * np.ones(n),
                6 * times,
                12 * times**2,
                20 * times**3,
            ]
        )

        plt.subplot(1, 3, 1)
        plt.plot(times, f)
        plt.subplot(1, 3, 2)
        plt.plot(times, df)
        plt.subplot(1, 3, 3)
        plt.plot(times, d2f)

    plt.show()


if __name__ == "__main__":
    main()
