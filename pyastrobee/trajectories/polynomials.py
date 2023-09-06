"""Generating trajectories via polynomials

Assorted ideas
- Add a "glide" section to the trajectory? e.g. reach the max velocity and then glide until we need to decelerate
- Enforce actuator limits - make sure that the trajectory doesn't exceed the max velocity values.
  If so, increase the duration
"""

from typing import Union

import numpy as np
import numpy.typing as npt
from numpy.polynomial.polynomial import Polynomial

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.quaternions import (
    random_quaternion,
    quaternion_slerp,
    quats_to_angular_velocities,
)
from pyastrobee.trajectories.quaternion_interpolation import (
    quaternion_interpolation_with_bcs,
)


def polynomial_slerp(
    q1: Union[npt.ArrayLike, Quaternion],
    q2: Union[npt.ArrayLike, Quaternion],
    n: int,
) -> np.ndarray:
    """SLERP based on a third-order polynomial discretization

    This will interpolate quaternions based on a polynomial spacing rather than a linear spacing. The resulting
    angular velocity vector has a constant direction, but will be quadratic, starting and ending at 0

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
    pcts = third_order_poly(0, 1, 0, 1, 0, 0)(np.linspace(0, 1, n, endpoint=True))
    return quaternion_slerp(q1, q2, pcts)


def third_order_poly(
    t0: float,
    tf: float,
    x0: float,
    xf: float,
    v0: float,
    vf: float,
) -> Polynomial:
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

    Returns:
        Polynomial: The polynomial which satisfies the boundary conditions. Evaluate the polynomial by
            calling it with the evaluation times. e.g. xs = solved_polynomial(times)
    """
    # Form linear system of equations: we have four polynomial coefficients for a third-order poly
    # and have four constraints on the endpoints (initial/final position/velocity)
    A = np.array(
        [
            [1, t0, t0**2, t0**3],
            [1, tf, tf**2, tf**3],
            [0, 1, 2 * t0, 3 * t0**2],
            [0, 1, 2 * tf, 3 * tf**2],
        ]
    )
    b = np.array([x0, xf, v0, vf])
    coeffs = np.linalg.solve(A, b)
    return Polynomial(coeffs)


def fifth_order_poly(
    t0: float,
    tf: float,
    x0: float,
    xf: float,
    v0: float,
    vf: float,
    a0: float,
    af: float,
) -> Polynomial:
    """Generate a fifth-order polynomial over a time domain based on boundary conditions

    We will refer to the variables in terms of position and velocity, but in general, this
    can be applied to any variable and its derivative

    Args:
        t0 (float): Start time
        tf (float): End time
        x0 (float): Initial position
        xf (float): Final position
        v0 (float): Initial velocity
        vf (float): Final velocity
        a0 (float): Initial acceleration
        af (float): Final acceleration

    Returns:
        Polynomial: The polynomial which satisfies the boundary conditions. Evaluate the polynomial by
            calling it with the evaluation times. e.g. xs = solved_polynomial(times)
    """
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
    b = np.array([x0, xf, v0, vf, a0, af])
    coeffs = np.linalg.solve(A, b)
    return Polynomial(coeffs)


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
    f_x = third_order_poly(t0, tf, x0, xf, v0, vf)
    f_y = third_order_poly(t0, tf, y0, yf, v0, vf)
    f_z = third_order_poly(t0, tf, z0, zf, v0, vf)
    f_vx: Polynomial = f_x.deriv()
    f_vy: Polynomial = f_y.deriv()
    f_vz: Polynomial = f_z.deriv()
    f_ax: Polynomial = f_vx.deriv()
    f_ay: Polynomial = f_vy.deriv()
    f_az: Polynomial = f_vz.deriv()
    q = polynomial_slerp(q0, qf, n)
    omega = quats_to_angular_velocities(q, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=np.column_stack([f_x(times), f_y(times), f_z(times)]),
        quats=q,
        lin_vels=np.column_stack([f_vx(times), f_vy(times), f_vz(times)]),
        ang_vels=omega,
        lin_accels=np.column_stack([f_ax(times), f_ay(times), f_az(times)]),
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
    f_x = third_order_poly(t0, tf, x0, xf, vx0, vxf)
    f_y = third_order_poly(t0, tf, y0, yf, vy0, vyf)
    f_z = third_order_poly(t0, tf, z0, zf, vz0, vzf)
    f_vx: Polynomial = f_x.deriv()
    f_vy: Polynomial = f_y.deriv()
    f_vz: Polynomial = f_z.deriv()
    f_ax: Polynomial = f_vx.deriv()
    f_ay: Polynomial = f_vy.deriv()
    f_az: Polynomial = f_vz.deriv()
    # TODO make these an input. It's a bit weird that we don't have the acceleration constraint though
    # since we're using the third-order poly on the position traj... Figure this out
    dw0 = np.zeros(3)
    dwf = np.zeros(3)
    q = quaternion_interpolation_with_bcs(q0, qf, w0, wf, dw0, dwf, duration, n)
    omega = quats_to_angular_velocities(q, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=np.column_stack([f_x(times), f_y(times), f_z(times)]),
        quats=q,
        lin_vels=np.column_stack([f_vx(times), f_vy(times), f_vz(times)]),
        ang_vels=omega,
        lin_accels=np.column_stack([f_ax(times), f_ay(times), f_az(times)]),
        ang_accels=alpha,
        times=times,
    )


def fifth_order_polynomial_traj_with_velocity_bcs(
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    v0: npt.ArrayLike,
    w0: npt.ArrayLike,
    a0: npt.ArrayLike,
    dw0: npt.ArrayLike,
    pf: npt.ArrayLike,
    qf: npt.ArrayLike,
    vf: npt.ArrayLike,
    wf: npt.ArrayLike,
    af: npt.ArrayLike,
    dwf: npt.ArrayLike,
    duration: float,
    dt: float,
) -> Trajectory:
    """Generate a polynomial trajectory between two poses, with velocity boundary conditions on either end

    Args:
        p0 (npt.ArrayLike): Initial position, shape (3,)
        q0 (npt.ArrayLike): Initial XYZW quaternion, shape (4,)
        v0 (npt.ArrayLike): Initial linear velocity, shape (3,)
        w0 (npt.ArrayLike): Initial angular velocity, shape (3,)
        a0 (npt.ArrayLike): Initial linear acceleration, shape (3,)
        dw0 (npt.ArrayLike): Initial angular acceleration, shape (3,)
        pf (npt.ArrayLike): Final position, shape (3,)
        qf (npt.ArrayLike): Final XYZW quaternion, shape (4,)
        vf (npt.ArrayLike): Final linear velocity, shape (3,)
        wf (npt.ArrayLike): Final angular velocity, shape (3,)
        af (npt.ArrayLike): Final linear acceleration, shape (3,)
        dwf (npt.ArrayLike): Final angular acceleration, shape (3,)
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
    ax0, ay0, az0 = a0
    axf, ayf, azf = af
    f_x = fifth_order_poly(t0, tf, x0, xf, vx0, vxf, ax0, axf)
    f_y = fifth_order_poly(t0, tf, y0, yf, vy0, vyf, ay0, ayf)
    f_z = fifth_order_poly(t0, tf, z0, zf, vz0, vzf, az0, azf)
    f_vx: Polynomial = f_x.deriv()
    f_vy: Polynomial = f_y.deriv()
    f_vz: Polynomial = f_z.deriv()
    f_ax: Polynomial = f_vx.deriv()
    f_ay: Polynomial = f_vy.deriv()
    f_az: Polynomial = f_vz.deriv()
    q = quaternion_interpolation_with_bcs(q0, qf, w0, wf, dw0, dwf, duration, n)
    omega = quats_to_angular_velocities(q, dt)
    alpha = np.gradient(omega, dt, axis=0)
    return Trajectory(
        positions=np.column_stack([f_x(times), f_y(times), f_z(times)]),
        quats=q,
        lin_vels=np.column_stack([f_vx(times), f_vy(times), f_vz(times)]),
        ang_vels=omega,
        lin_accels=np.column_stack([f_ax(times), f_ay(times), f_az(times)]),
        ang_accels=alpha,
        times=times,
    )


def _main():
    # TODO add the astrobee following the trajectory??

    # Update these values depending on what examples you want to run
    RUN_NO_BC_EXAMPLE = True
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
    _main()
