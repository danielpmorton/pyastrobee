"""Test script to evaluate 5th order polynomials with one free parameter

This is mainly to evaluate how much the trajectory can change when we leave a parameter unconstrained
and don't just evaluate the nominal solution we might get at first
"""
# TODO: figure out how to quantify which one of these is "best" in a convex way so we can reliably optimize trajs

import numpy as np
import matplotlib.pyplot as plt


def main():
    t0 = 0
    tf = 5
    x0 = 1
    xf = 2
    v0 = 0.1
    vf = 0.2
    a0 = -0.2
    # af = -0.1 # LEAVE THIS UNCONSTRAINED
    n_pts = 50

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
    m, n = A.shape  # (5, 6) - 6 parameters for 5th order poly, 5 constraints

    # Have Ax=b, but A is R5x6, x is R6, b = R5
    # So we have one degree of freedom

    A_bar = A.T @ np.linalg.inv(A @ A.T)  # Right inverse
    N = np.eye(n) - A_bar @ A  # Nullspace projection matrix

    # Form a linear discretization in time
    times = np.linspace(t0, tf, n_pts, endpoint=True)

    # Generate a bunch of vectors in the nullspace to modify the parameters of the polynomial
    n_param_sets = 20
    # Set up matrices for storing the data
    params = np.zeros((n_param_sets, n))
    delta_params = np.zeros((n_param_sets, n))
    fs = np.zeros((n_param_sets, len(times)))
    dfs = np.zeros((n_param_sets, len(times)))
    d2fs = np.zeros((n_param_sets, len(times)))

    # Calculate matrices corresponding to our time evolution, relating to the polynomial coefficients and derivatives
    tau = np.row_stack(
        [np.ones(n_pts), times, times**2, times**3, times**4, times**5]
    )
    dtau = np.row_stack(
        [
            np.zeros(n_pts),
            np.ones(n_pts),
            2 * times,
            3 * times**2,
            4 * times**3,
            5 * times**4,
        ]
    )
    d2tau = np.row_stack(
        [
            np.zeros((2, n_pts)),
            2 * np.ones(n_pts),
            6 * times,
            12 * times**2,
            20 * times**3,
        ]
    )
    # Generate a bunch of vectors in the nullspace of our polynomial constraints
    delta_params = (N @ np.random.rand(6, n_param_sets)).T
    # Add these vectors to the nominal polynomial coefficients
    params = A_bar @ b + delta_params
    # Determine the function and derivative evaluations over the duration of the traj
    fs = params @ tau
    dfs = params @ dtau
    d2fs = params @ d2tau

    # Print out the coefficients for debugging
    for i in range(n_param_sets):
        print(f"{i}: {params[i]}")

    # Plot each trajectory for comparison
    # Set up figure
    fig = plt.figure()
    labels = [str(i) for i in range(n_param_sets)]
    plt.subplot(1, 3, 1)
    plt.plot(times, fs.T, label=labels)
    plt.title("Position")
    plt.subplot(1, 3, 2)
    plt.plot(times, dfs.T, label=labels)
    plt.title("Velocity")
    plt.subplot(1, 3, 3)
    plt.plot(times, d2fs.T, label=labels)
    plt.title("Acceleration")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
