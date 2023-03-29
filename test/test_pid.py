"""Test cases for the PID controller"""

import unittest

import numpy as np

from pyastrobee.control.pid import PID


class PIDTest(unittest.TestCase):
    """Contains test cases to ensure that the PID controller is working as expected"""

    def test_scalar_inputs(self):
        kp = 1
        ki = 2
        kd = 3
        i_min = -1
        i_max = 1
        controller = PID(kp, ki, kd, i_min, i_max)
        self.assertTrue(controller.p_gains == kp)
        self.assertTrue(controller.i_gains == ki)
        self.assertTrue(controller.d_gains == kd)
        self.assertTrue(controller.i_mins == i_min)
        self.assertTrue(controller.i_maxes == i_max)
        # We have only initialized this, so there should be no command
        self.assertTrue(controller.cmd == 0)
        dt = 1
        err = 1
        controller.update(err, dt)
        self.assertTrue(controller.p_errors == err)
        self.assertTrue(controller.cmd != 0)

    def test_array_inputs(self):
        kp = [1, 2]
        ki = [2, 3]
        kd = [3, 4]
        i_min = [-1, -2]
        i_max = [1, 2]
        controller = PID(kp, ki, kd, i_min, i_max)
        np.testing.assert_array_equal(controller.p_gains, kp)
        np.testing.assert_array_equal(controller.i_gains, ki)
        np.testing.assert_array_equal(controller.d_gains, kd)
        np.testing.assert_array_equal(controller.i_mins, i_min)
        np.testing.assert_array_equal(controller.i_maxes, i_max)
        # We have only initialized this, so there should be no command
        np.testing.assert_array_equal(controller.cmd, [0, 0])
        dt = 1
        err = [1, 2]
        controller.update(err, dt)
        np.testing.assert_array_equal(controller.p_errors, err)
        self.assertTrue(controller.cmd[0] != 0)
        self.assertTrue(controller.cmd[1] != 0)

    def test_decoupled_matrix_inputs(self):
        kp = np.diag([1, 2])
        ki = np.diag([2, 3])
        kd = np.diag([3, 4])
        i_min = [-1, -2]
        i_max = [1, 2]
        controller = PID(kp, ki, kd, i_min, i_max)
        np.testing.assert_array_equal(controller.p_gains, kp)
        np.testing.assert_array_equal(controller.i_gains, ki)
        np.testing.assert_array_equal(controller.d_gains, kd)
        np.testing.assert_array_equal(controller.i_mins, i_min)
        np.testing.assert_array_equal(controller.i_maxes, i_max)
        # We have only initialized this, so there should be no command
        np.testing.assert_array_equal(controller.cmd, [0, 0])
        dt = 1
        err = [1, 2]
        controller.update(err, dt)
        np.testing.assert_array_equal(controller.p_errors, err)
        self.assertTrue(controller.cmd[0] != 0)
        self.assertTrue(controller.cmd[1] != 0)

    def test_coupled_matrix_inputs(self):
        kp = np.array([[1, 2], [3, 4]])
        ki = np.diag([2, 3])  # Can't couple this
        kd = np.array([[3, 4], [5, 6]])
        i_min = [-1, -2]
        i_max = [1, 2]
        controller = PID(kp, ki, kd, i_min, i_max)
        np.testing.assert_array_equal(controller.p_gains, kp)
        np.testing.assert_array_equal(controller.i_gains, ki)
        np.testing.assert_array_equal(controller.d_gains, kd)
        np.testing.assert_array_equal(controller.i_mins, i_min)
        np.testing.assert_array_equal(controller.i_maxes, i_max)
        # We have only initialized this, so there should be no command
        np.testing.assert_array_equal(controller.cmd, [0, 0])
        dt = 1
        err = [1, 2]
        controller.update(err, dt)
        np.testing.assert_array_equal(controller.p_errors, err)
        self.assertTrue(controller.cmd[0] != 0)
        self.assertTrue(controller.cmd[1] != 0)

    def test_mismatched_dimensions(self):
        kp = [1, 2]
        ki = 2
        kd = [3, 4]
        i_min = [-1, -2]
        i_max = [1, 2]
        with self.assertRaises(ValueError):
            controller = PID(kp, ki, kd, i_min, i_max)
        # Fix the issue, and then test to see if an error with the wrong shape will also error
        ki = [2, 3]
        controller = PID(kp, ki, kd, i_min, i_max)
        with self.assertRaises(ValueError):
            controller.update(1, 1)

    def test_coupled_integral_gain(self):
        # This should specifically raise an error because I haven't figured out
        # how to deal with this yet
        kp = np.array([[1, 2], [3, 4]])
        ki = np.array([[2, 3], [4, 5]])  # Can't couple this
        kd = np.array([[3, 4], [5, 6]])
        i_min = [-1, -2]
        i_max = [1, 2]
        with self.assertRaises(ValueError):
            controller = PID(kp, ki, kd, i_min, i_max)

    def test_gain_reset(self):
        controller = PID(1, 2, 3, 4, 5)
        controller.set_gains(0, 0, 0, 0, 0)
        self.assertTrue(controller.p_gains == 0)
        self.assertTrue(controller.i_gains == 0)
        self.assertTrue(controller.d_gains == 0)
        self.assertTrue(controller.i_mins == 0)
        self.assertTrue(controller.i_maxes == 0)

    def test_array_vs_decoupled_matrix(self):
        # These should have precisely the same command values
        kp1 = [1, 2]
        kp2 = np.diag(kp1)
        ki1 = [2, 3]
        ki2 = np.diag(ki1)
        kd1 = [3, 4]
        kd2 = np.diag(kd1)
        i_min = [-1, -2]
        i_max = [1, 2]
        controller1 = PID(kp1, ki1, kd1, i_min, i_max)
        controller2 = PID(kp2, ki2, kd2, i_min, i_max)
        err = [1, 2]
        dt = 1
        controller1.update(err, dt)
        controller2.update(err, dt)
        np.testing.assert_array_equal(controller1.cmd, controller2.cmd)
        err = [0.5, 1]
        controller1.update(err, dt)
        controller2.update(err, dt)
        np.testing.assert_array_equal(controller1.cmd, controller2.cmd)


if __name__ == "__main__":
    unittest.main()
