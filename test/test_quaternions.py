"""Test cases for quaternions

TODO
- Quaternion distance
- Quaternion interpolation
"""

import unittest
import numpy as np
import ahrs

from pyastrobee.utils import rotations as rts
from pyastrobee.utils.math_utils import normalize
from pyastrobee.utils.quaternion import (
    Quaternion,
    quaternion_derivative,
    quats_to_angular_velocities,
)


class QuaternionTest(unittest.TestCase):
    """Contains test cases for the Quaternion class and all quaternion-associated functions"""

    def test_quaternion_class(self):
        q0 = normalize([0.1, 0.2, 0.3, 0.4])
        q = Quaternion(wxyz=q0)
        np.testing.assert_array_almost_equal(q0, q.wxyz, decimal=14)
        np.testing.assert_array_almost_equal([*q0[1:], q0[0]], q.xyzw, decimal=14)

    def test_quaternion_conversions(self):
        q = normalize([0.1, 0.2, 0.3, 0.4])
        axis, angle = rts.quat_to_axis_angle(q)
        np.testing.assert_array_almost_equal(rts.axis_angle_to_quat(axis, angle), q)
        rmat = rts.quat_to_rmat(q)
        np.testing.assert_array_almost_equal(rts.rmat_to_quat(rmat), q)
        euler_xyz = rts.quat_to_euler_xyz(q)
        np.testing.assert_array_almost_equal(rts.euler_xyz_to_quat(euler_xyz), q)
        fixed_xyz = rts.quat_to_fixed_xyz(q)
        np.testing.assert_array_almost_equal(rts.fixed_xyz_to_quat(fixed_xyz), q)

    def test_quaternion_combination(self):
        euler_1 = np.array([0, 0, 0])
        euler_2 = np.array([0.1, 0.2, 0.3])
        q1 = rts.euler_xyz_to_quat(euler_1)
        q2 = rts.euler_xyz_to_quat(euler_2)
        q3 = rts.combine_quaternions(q1, q2)
        euler_3 = rts.quat_to_euler_xyz(q3)
        np.testing.assert_array_almost_equal(euler_3, euler_1 + euler_2)

    def test_quaternion_deriv(self):
        # Use a known angular velocity to propagate a quaternion
        # Then, see if we get back roughly the same angular velocity
        # Test against AHRS to make sure it works correctly
        w = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        # AHRS
        ahrs_q1 = ahrs.Quaternion([1, 0, 0, 0])
        ahrs_deriv = ahrs_q1.ode(w)  # Quaternion derivative
        ahrs_q2 = ahrs_q1 + ahrs_deriv * dt
        ahrs_arr = ahrs.QuaternionArray(np.row_stack([ahrs_q1, ahrs_q2]))
        ahrs_ang_vel = AHRS_angular_velocities(ahrs_arr, dt).flatten()
        # My version
        my_q1 = np.array([0, 0, 0, 1])
        my_deriv = quaternion_derivative(my_q1, w)
        my_q2 = normalize(my_q1 + my_deriv * dt)
        my_arr = np.row_stack([my_q1, my_q2])
        my_ang_vel = quats_to_angular_velocities(my_arr, dt).flatten()
        np.testing.assert_array_almost_equal(ahrs_ang_vel, my_ang_vel, decimal=6)
        # The following check is very low-precision but this seems to be a general drawback
        # of this method that we're using
        np.testing.assert_almost_equal(my_ang_vel, w, decimal=2)

    def test_quaternion_interpolation(self):
        # TODO. Need to think of a good way to evaluate this test case
        pass

    def test_quaternion_distance(self):
        # TODO. Need to think of a good way to evaluate this test case
        pass


# This is for testing quaternion derivatives against AHRS
# This is the AHRS implementation, but this function is not available through PIP yet
# Note that AHRS uses WXYZ quaternions rather than XYZW
def AHRS_angular_velocities(q: ahrs.QuaternionArray, dt: float) -> np.ndarray:
    """
        Compute the angular velocity between N Quaternions.
        It assumes a constant sampling rate of ``dt`` seconds, and returns the
        angular velocity around the X-, Y- and Z-axis (roll-pitch-yaw angles),
        in radians per second.
        The angular velocities :math:`\\omega_x`, :math:`\\omega_y`, and
        :math:`\\omega_z` are computed from quaternions :math:`\\mathbf{q}_t=\\Big(q_w(t), q_x(t), q_y(t), q_z(t)\\Big)`
        and :math:`\\mathbf{q}_{t+\\Delta t}=\\Big(q_w(t+\\Delta t), q_x(t+\\Delta t), q_y(t+\\Delta t), q_z(t+\\Delta t)\\Big)`
        as:
        .. math::
            \\begin{array}{rcl}
            \\omega_x &=& \\frac{2}{\\Delta t}\\Big(q_w(t) q_x(t+\\Delta t) - q_x(t) q_w(t+\\Delta t) - q_y(t) q_z(t+\\Delta t) + q_z(t) q_y(t+\\Delta t)\\Big) \\\\ \\\\
            \\omega_y &=& \\frac{2}{\\Delta t}\\Big(q_w(t) q_y(t+\\Delta t) + q_x(t) q_z(t+\\Delta t) - q_y(t) q_w(t+\\Delta t) - q_z(t) q_x(t+\\Delta t)\\Big) \\\\ \\\\
            \\omega_z &=& \\frac{2}{\\Delta t}\\Big(q_w(t) q_z(t+\\Delta t) - q_x(t) q_y(t+\\Delta t) + q_y(t) q_x(t+\\Delta t) - q_z(t) q_w(t+\\Delta t)\\Big)
            \\end{array}
        where :math:`\\Delta t` is the time step between consecutive
        quaternions [MarioGC1]_.
        Parameters
        ----------
        dt : float
            Time step, in seconds, between consecutive Quaternions.
        Returns
        -------
        w : numpy.ndarray
            (N-1)-by-3 array with angular velocities in rad/s.
        """
    if not isinstance(dt, float):
        raise TypeError(f"dt must be a float. Got {type(dt)}.")
    if dt <= 0:
        raise ValueError(f"dt must be greater than zero. Got {dt}.")
    w = np.c_[
        q.w[:-1] * q.x[1:]
        - q.x[:-1] * q.w[1:]
        - q.y[:-1] * q.z[1:]
        + q.z[:-1] * q.y[1:],
        q.w[:-1] * q.y[1:]
        + q.x[:-1] * q.z[1:]
        - q.y[:-1] * q.w[1:]
        - q.z[:-1] * q.x[1:],
        q.w[:-1] * q.z[1:]
        - q.x[:-1] * q.y[1:]
        + q.y[:-1] * q.x[1:]
        - q.z[:-1] * q.w[1:],
    ]
    return 2.0 * w / dt


if __name__ == "__main__":
    unittest.main()
