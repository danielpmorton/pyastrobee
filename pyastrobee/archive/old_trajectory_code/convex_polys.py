# Note: if this works, should try to get this method extended to the orientation trajectory
# since the same fundamentals of acceleration continuity and reduced overshoot still apply

# Currently the objective is NONCONVEX, so need to figure out a better way to deal with this

import numpy as np
import cvxpy as cp


def position_trajectory_solver(
    t0: float,
    tf: float,
    x0: float,
    xf: float,
    v0: float,
    vf: float,
    a0: float,
    # af: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def tau(t):
        """5th order polynomial without coefficients, at time t"""
        return np.array([1, t, t**2, t**3, t**4, t**5])

    def d_tau(t):
        """Derivative of 5th order polynomial without coefficients, at time t"""
        return np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])

    def d2_tau(t):
        """Second derivative of 5th order polynomial without coefficients, at time t"""
        return np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3])

    def d3_tau(t):
        """Third derivative of 5th order polynomial without coefficients, at time t"""
        return np.array([0, 0, 0, 6, 24 * t, 60 * t**2])

    a = cp.Variable(6)
    t = cp.Variable(1)
    objective = cp.Minimize(cp.Maximize(a @ d3_tau(t)))  # NONCONVEX
    # Need to figure out what the best objective here would be
    # Minimax on jerk seems to not be convex / DCP
    constraints = [
        a @ tau(t0) == x0,
        a @ tau(tf) == xf,
        a @ d_tau(t0) == v0,
        a @ d_tau(tf) == vf,
        a @ d2_tau(t0) == a0,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Form linear system of equations: we have six polynomial coefficients for a fifth-order poly
    # and have six constraints on the endpoints (initial/final position/velocity/acceleration)
    A = np.array(
        [
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
            [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
            [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
            [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
        ]
    )
    b = np.array([x0, xf, v0, vf, a0])  # , af])
    coeffs = np.linalg.solve(A, b)
    # Form a linear discretization in time
    times = np.linspace(t0, tf, n, endpoint=True)
    # Use this to calculate the polynomial based on the coefficients
    f = coeffs.T @ np.row_stack(
        [np.ones(n), times, times**2, times**3, times**4, times**5]
    )
    # Also calculate the derivatives of the function (for instance, velocity/acceleration)
    df = coeffs.T @ np.row_stack(
        [
            np.zeros(n),
            np.ones(n),
            2 * times,
            3 * times**2,
            4 * times**3,
            5 * times**4,
        ]
    )
    d2f = coeffs.T @ np.row_stack(
        [np.zeros((2, n)), 2 * np.ones(n), 6 * times, 12 * times**2, 20 * times**3]
    )
    return f, df, d2f, times
