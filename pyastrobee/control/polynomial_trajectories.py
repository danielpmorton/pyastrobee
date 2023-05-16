"""Generating trajectories via cubic polynomials

TODO
- Add a "glide" section to the trajectory? e.g. reach the max velocity and then glide until we need to decelerate
- Enforce actuator limits - make sure that the trajectory doesn't exceed the max velocity values.
  If so, increase the duration
- Merge this file with planner.py?
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from pyastrobee.control.trajectory import Trajectory
from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.quaternions import (
    random_quaternion,
    quaternion_slerp,
    quats_to_angular_velocities,
)
from pyastrobee.control.quaternion_bc_planning import quaternion_interpolation_with_bcs


def polynomial_slerp(
    q1: Union[npt.ArrayLike, Quaternion],
    q2: Union[npt.ArrayLike, Quaternion],
    n: int,
) -> np.ndarray:
    """SLERP based on a third-order polynomial discretization

    This will sample interpolated quaternions based on the polynomial rather than a linear spacing

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): Starting quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        q2 (Union[Quaternion, npt.ArrayLike]): Ending quaternion. If passing in a np array,
            must be in XYZW order (length = 4)
        n (int): Number of points at which to evaluate the polynomial-based SLERP

    Returns:
        np.ndarray: The interpolated XYZW quaternions, shape = (n, 4)
    """
    # Generate our evaluation points for SLERP so that:
    # - We evaluate over a domain of [0, 1] with n steps
    # - We want to start at 0 and end at 1 with 0 derivative at either end
    pcts = third_order_poly(0, 1, 0, 1, 0, 0, n)[0]
    return quaternion_slerp(q1, q2, pcts)


def third_order_poly(
    t0: float, tf: float, x0: float, xf: float, v0: float, vf: float, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a third-order polynomial over a time domain based on boundary conditions

    We will refer to the variables in terms of position and velocity, but in general, this
    can be applied to any variable and its derivative

    Args:
        t0 (float): Start time
        tf (float): End time
        x0 (float): Initial position
        xf (float): Final position
        v0 (float): Initial velocity
        vf (float): Final velocity
        n (int): Number of discretizations in time

    Returns:
        Tuple of:
            np.ndarray: The third-order polynomial function evaluated from t0->tf over n timesteps, shape (n,)
            np.ndarray: The first derivative of the polynomial, shape (n,)
            np.ndarray: The second derivative of the polynomial, shape (n,)
            np.ndarray: The times associated with the polynomial, shape (n,)
    """
    # Form linear system of equations: we have four polynomial coefficients for a third-order poly
    # and have four constraints on the endpoints (initial/final position/velocity)
    A = np.array(
        [
            [1, t0, t0**2, t0**3],
            [1, tf, tf**2, tf**3],
            [0, 1, 1 * t0, 3 * t0**2],
            [0, 1, 2 * tf, 3 * tf**2],
        ]
    )
    b = np.array([x0, xf, v0, vf])
    coeffs = np.linalg.solve(A, b)
    # Form a linear discretization in time
    times = np.linspace(t0, tf, n, endpoint=True)
    # Use this to calculate the polynomial based on the coefficients
    f = coeffs.T @ np.row_stack([np.ones(n), times, times**2, times**3])
    # Also calculate the derivatives of the function (for instance, velocity/acceleration)
    df = coeffs.T @ np.row_stack([np.zeros(n), np.ones(n), 2 * times, 3 * times**2])
    d2f = coeffs.T @ np.row_stack([np.zeros((2, n)), 2 * np.ones(n), 6 * times])
    return f, df, d2f, times


def polynomial_trajectory(
    pose_1: npt.ArrayLike, pose_2: npt.ArrayLike, duration: float, dt: float
) -> Trajectory:
    """Generate a third-order polynomial trajectory between two poses

    Args:
        pose_1 (npt.ArrayLike): Starting position + XYZW quaternion pose, shape (7,)
        pose_2 (npt.ArrayLike): Ending position + XYZW quaternion pose, shape (7,)
        duration (float): Trajectory duration (seconds)
        dt (float): Sampling period (seconds)

    Returns:
        Trajectory: Trajectory with position, orientation, lin/ang velocity, lin/ang acceleration, and time info
    """
    times = np.arange(0, duration + dt, dt)
    n = len(times)
    x0, y0, z0 = pose_1[:3]
    q0 = pose_1[3:]
    xf, yf, zf = pose_2[:3]
    qf = pose_2[3:]
    t0 = times[0]
    tf = times[-1]
    v0 = 0
    vf = 0
    x, vx, ax, _ = third_order_poly(t0, tf, x0, xf, v0, vf, n)
    y, vy, ay, _ = third_order_poly(t0, tf, y0, yf, v0, vf, n)
    z, vz, az, _ = third_order_poly(t0, tf, z0, zf, v0, vf, n)
    q = polynomial_slerp(q0, qf, n)
    omega = quats_to_angular_velocities(q, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=np.column_stack([x, y, z]),
        quats=q,
        lin_vels=np.column_stack([vx, vy, vz]),
        ang_vels=omega,
        lin_accels=np.column_stack([ax, ay, az]),
        ang_accels=alpha,
        times=times,
    )


def polynomial_traj_with_velocity_bcs(
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    v0: npt.ArrayLike,
    w0: npt.ArrayLike,
    pf: npt.ArrayLike,
    qf: npt.ArrayLike,
    vf: npt.ArrayLike,
    wf: npt.ArrayLike,
    duration: float,
    dt: float,
) -> Trajectory:
    """Generate a polynomial trajectory between two poses, with velocity boundary conditions on either end

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        v0 (npt.ArrayLike): Initial linear velocity, shape (3,)
        w0 (npt.ArrayLike): Initial angular velocity, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)
        vf (npt.ArrayLike): Final linear velocity, shape (3,)
        wf (npt.ArrayLike): Final angular velocity, shape (3,)
        duration (float): Trajectory duration (seconds)
        dt (float): Sampling period (seconds)

    Returns:
        Trajectory: Trajectory with position, orientation, lin/ang velocity, lin/ang acceleration, and time info
    """
    times = np.arange(0, duration + dt, dt)
    n = len(times)
    x0, y0, z0 = p0
    xf, yf, zf = pf
    t0 = times[0]
    tf = times[-1]
    vx0, vy0, vz0 = v0
    vxf, vyf, vzf = vf
    x, vx, ax, _ = third_order_poly(t0, tf, x0, xf, vx0, vxf, n)
    y, vy, ay, _ = third_order_poly(t0, tf, y0, yf, vy0, vyf, n)
    z, vz, az, _ = third_order_poly(t0, tf, z0, zf, vz0, vzf, n)
    q = quaternion_interpolation_with_bcs(q0, qf, w0, wf, duration, n)
    omega = quats_to_angular_velocities(q, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=np.column_stack([x, y, z]),
        quats=q,
        lin_vels=np.column_stack([vx, vy, vz]),
        ang_vels=omega,
        lin_accels=np.column_stack([ax, ay, az]),
        ang_accels=alpha,
        times=times,
    )


def main():
    # TODO add the astrobee following the trajectory??

    # Update these values depending on what examples you want to run
    RUN_NO_BC_EXAMPLE = False
    RUN_BC_EXAMPLE = True

    # Example with no boundary conditions
    if RUN_NO_BC_EXAMPLE:
        np.random.seed(0)
        pose1 = np.array([0, 0, 0, 0, 0, 0, 1])
        pose2 = np.array([1, 1, 1, *random_quaternion()])
        traj = polynomial_trajectory(pose1, pose2, 5, 0.1)
        print("Plotting trajectory information")
        traj.plot()
        input("Press Enter to visualize the trajectory in pybullet, when ready")
        traj.visualize()

    # Example with boundary conditions on the initial/final velocities
    if RUN_BC_EXAMPLE:
        p0 = np.array([1, 2, 3])
        q0 = np.array([1, 2, 3, 4]) / np.linalg.norm([1, 2, 3, 4])
        v0 = np.array([1, 2, 3])
        w0 = np.array([0.1, 0.2, 0.3])
        pf = np.array([2, 3, 4])
        qf = np.array([2, 3, 4, 5]) / np.linalg.norm([2, 3, 4, 5])
        vf = np.array([2, 3, 4])
        wf = np.array([0.2, 0.3, 0.4])
        duration = 10
        dt = 0.1  # Arbitrary
        traj = polynomial_traj_with_velocity_bcs(
            p0, q0, v0, w0, pf, qf, vf, wf, duration, dt
        )
        traj.plot()
        traj.visualize()


if __name__ == "__main__":
    main()
