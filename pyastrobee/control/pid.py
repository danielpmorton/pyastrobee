"""An extended version of a PID controller compatible with multivariate systems

TODO decide if negative gains could actually be possible in the coupled case
TODO determine how to handle coupled gains in the integral case

Based on pid.cpp (c) 2008, Willow Garage, Inc. (BSD License)
as well as Nathan Sprague's Python translation
"""

from typing import Optional, Union

import time
import math

import numpy as np
import numpy.typing as npt

from pyastrobee.utils.math_utils import is_diagonal


class PID:
    """An extended version of a PID controller compatible with multivariate systems

    This class implements the standard pid equation:

    $command = p_{term} + i_{term} + d_{term} $

    where:

    $ p_{term} = p_{gain} * p_{error} $
    $ i_{term} = i_{gain} * i_{error} $
    $ d_{term} = d_{gain} * d_{error} $
    $ i_{error} = i_{error} + p_{error} * dt $
    $ d_{error} = (p_{error} - p_{error last}) / dt $

    given:

    $ p_{error} = p_{target} - p_{state} $.

    - For a SISO system, pass in the gains as scalars
    - For a decoupled MIMO system, pass in the gains as lists, 1d arrays, or diagonal matrices
    - For a general MIMO system, pass in the gains as matrices
    - Integral limits are scalar for SISO, 1d array for MIMO.

    Args:
        p_gains (Union[npt.ArrayLike, float]): Proportional gain(s).
        i_gains (Union[npt.ArrayLike, float]): Integral gain(s).
        d_gains (Union[npt.ArrayLike, float]): Derivative gain(s).
        i_mins (Union[npt.ArrayLike, float, None]): Minimum value(s) for the integral term(s).
            Defaults to None, in which case the integral term(s) will be unbounded below
        i_maxes (Union[npt.ArrayLike, float, None]): Maximum value(s) for the integral term(s).
            Defaults to None, in which case the integral term(s) will be unbounded above
    """

    def __init__(
        self,
        p_gains: Union[npt.ArrayLike, float],
        i_gains: Union[npt.ArrayLike, float],
        d_gains: Union[npt.ArrayLike, float],
        i_mins: Union[npt.ArrayLike, float, None] = None,
        i_maxes: Union[npt.ArrayLike, float, None] = None,
    ):
        # Use the proportional gains as our baseline for comparison
        # Calculate some shapes/sizes to help distinguish between scalar/array/matrix methods
        reference_gains = np.atleast_1d(p_gains)
        self._gain_shape = reference_gains.shape
        self.n = self._gain_shape[0]
        self.using_gain_matrices = reference_gains.ndim > 1
        self.set_gains(p_gains, i_gains, d_gains, i_mins, i_maxes)
        self.reset()

    def reset(self):
        """Reset the state of the PID controller"""
        # Save position state for derivative state calculation
        self._p_errors_last = np.zeros(self.n)
        self._p_errors = np.zeros(self.n)  # Proportional error.
        self._d_errors = np.zeros(self.n)  # Derivative error.
        self._i_errors = np.zeros(self.n)  # Integator error.
        self._cmd = 0 if self.n == 1 else np.zeros(self.n)  # Command to send.
        self._last_time = None  # Used for automatic calculation of dt.

    def set_gains(
        self,
        p_gains: Union[npt.ArrayLike, float],
        i_gains: Union[npt.ArrayLike, float],
        d_gains: Union[npt.ArrayLike, float],
        i_mins: Union[npt.ArrayLike, float, None] = None,
        i_maxes: Union[npt.ArrayLike, float, None] = None,
    ):
        """Sets the PID gains for the controller.

        Gains are scalar for SISO, 1d array for decoupled MIMO, matrix for general MIMO
        Integral limits are scalar for SISO, 1d array for MIMO.

        Args:
            p_gains (Union[npt.ArrayLike, float]): Proportional gain(s).
            i_gains (Union[npt.ArrayLike, float]): Integral gain(s).
            d_gains (Union[npt.ArrayLike, float]): Derivative gain(s).
            i_mins (Union[npt.ArrayLike, float, None]): Minimum value(s) for the integral term(s).
                Defaults to None, in which case the integral term(s) will be unbounded below
            i_maxes (Union[npt.ArrayLike, float, None]): Maximum value(s) for the integral term(s).
                Defaults to None, in which case the integral term(s) will be unbounded above
        """
        # Call the setter functions, which will auto-validate the inputs
        self.p_gains = p_gains
        self.i_gains = i_gains
        self.d_gains = d_gains
        self.i_mins = i_mins
        self.i_maxes = i_maxes

    @property
    def p_gains(self) -> np.ndarray:
        """Proportional gains"""
        return self._p_gains

    @property
    def i_gains(self) -> np.ndarray:
        """Integral gains"""
        return self._i_gains

    @property
    def d_gains(self) -> np.ndarray:
        """Derivative gains"""
        return self._d_gains

    @property
    def i_maxes(self) -> np.ndarray:
        """Maximum value(s) of the integral term(s)"""
        return self._i_maxes

    @property
    def i_mins(self) -> np.ndarray:
        """Minimum value(s) of the integral term(s)"""
        return self._i_mins

    def _validate_gains(self, gains: Union[npt.ArrayLike, float]) -> np.ndarray:
        """Ensures that the gains are the correct shape and are nonnegative (raises an error otherwise)

        Args:
            gains (Union[npt.ArrayLike, float]): P, I, or D gain(s)

        Returns:
            np.ndarray: Validated gains
        """
        gains = np.atleast_1d(gains)
        if gains.shape != self._gain_shape:
            raise ValueError(
                f"Shape mismatch: Expected gains to have shape {self._gain_shape}, got {gains.shape}"
            )
        if np.any(gains < 0):
            raise ValueError(f"Negative gains will cause instability.\nGot: {gains}")
        return gains

    def _validate_integral_limits(
        self, lims: Union[npt.ArrayLike, float]
    ) -> np.ndarray:
        """Ensures that the integral term limits are the correct shape (raises an error otherwise)

        Args:
            lims (Union[npt.ArrayLike, float]): Limits (minimum or maximum) on the integral term(s)

        Returns:
            np.ndarray: Validated limits
        """
        lims = np.atleast_1d(lims)
        if len(lims) != self.n:
            raise ValueError(
                f"Invalid integral limits.\nExpected an array of length {self.n}, got {lims}"
            )
        return lims

    @p_gains.setter
    def p_gains(self, gains: Union[npt.ArrayLike, float]):
        gains = self._validate_gains(gains)
        self._p_gains = gains

    @i_gains.setter
    def i_gains(self, gains: Union[npt.ArrayLike, float]):
        gains = self._validate_gains(gains)
        if self.using_gain_matrices:
            # Extra logic to ensure that the gain matrix is diagonal (decoupled) because
            # determining the integral saturation for this case requires some more work
            if not is_diagonal(gains):
                raise ValueError(
                    "Coupling between the integral gains is not currently supported"
                )
        self._i_gains = gains

    @d_gains.setter
    def d_gains(self, gains: Union[npt.ArrayLike, float]):
        gains = self._validate_gains(gains)
        self._d_gains = gains

    @i_maxes.setter
    def i_maxes(self, lims: Union[npt.ArrayLike, float, None]):
        if lims is None:
            lims = float("inf") * np.ones(self.n)
        else:
            lims = self._validate_integral_limits(lims)
        self._i_maxes = lims

    @i_mins.setter
    def i_mins(self, lims: Union[npt.ArrayLike, float, None]):
        if lims is None:
            lims = float("-inf") * np.ones(self.n)
        else:
            lims = self._validate_integral_limits(lims)
        self._i_mins = lims

    @property
    def p_errors(self) -> np.ndarray:
        """Proportional errors (Read-only)"""
        return self._p_errors

    @property
    def i_errors(self) -> np.ndarray:
        """Integral errors (Read-only)"""
        return self._i_errors

    @property
    def d_errors(self) -> np.ndarray:
        """Derivative errors (Read-only)"""
        return self._d_errors

    @property
    def cmd(self) -> np.ndarray:
        """Read-only access to the latest command."""
        return self._cmd

    def __str__(self):
        """String representation of the current state of the controller."""
        result = ""
        result += "p_gain:  " + str(self.p_gains) + "\n"
        result += "i_gain:  " + str(self.i_gains) + "\n"
        result += "d_gain:  " + str(self.d_gains) + "\n"
        result += "i_min:   " + str(self.i_mins) + "\n"
        result += "i_max:   " + str(self.i_maxes) + "\n"
        result += "p_error: " + str(self.p_errors) + "\n"
        result += "i_error: " + str(self.i_errors) + "\n"
        result += "d_error: " + str(self.d_errors) + "\n"
        result += "cmd:     " + str(self.cmd) + "\n"
        return result

    def update(
        self, p_errors: Union[npt.ArrayLike, float], dt: Optional[float] = None
    ) -> Union[np.ndarray, float]:
        """Update the PID controller state

        Args:
            p_errors (Union[npt.ArrayLike, float]): Error since last iteration (target - state)
            dt (Optional[float]): Change in time since the last iteration, in seconds. Defaults to None,
                in which case the system clock will be used to determine the delta

        Returns:
            Union[np.ndarray, float]: PID controller output. Array if we have a MIMO system, float if SISO
        """
        # Validate error input
        p_errors = np.atleast_1d(p_errors)
        if len(p_errors) != self.n:
            raise ValueError(
                f"Invalid error term.\nExpected an array of length {self.n}, got {p_errors}"
            )
        self._p_errors = p_errors

        # Validate dt input
        if dt is None:
            cur_time = time.time()
            if self._last_time is None:
                self._last_time = cur_time
                # TODO decide why they put this line here (below)
                self._p_errors_last = p_errors
            dt = cur_time - self._last_time
            self._last_time = cur_time
        assert not math.isnan(dt) and not math.isinf(dt)

        # Proportional component
        if self.using_gain_matrices:
            p_term = self._p_gains @ self._p_errors
        else:
            p_term = self._p_gains * self._p_errors

        # Integral component
        self._i_errors += dt * self._p_errors

        if self.using_gain_matrices:
            i_term = self._i_gains @ self._i_errors
            i_term = np.clip(i_term, self._i_mins, self._i_maxes)
            i_gains_diag = np.diag(self._i_gains)
            self._i_errors = np.where(
                i_gains_diag != 0, i_term / i_gains_diag, self._i_errors
            )
        else:
            i_term = self._i_gains * self._i_errors
            # Saturate the integral term to prevent it from getting too large
            i_term = np.clip(i_term, self._i_mins, self._i_maxes)
            self._i_errors = np.where(
                self._i_gains != 0, i_term / self._i_gains, self._i_errors
            )

        # Derivative component
        # If the timestep is 0, we can't calculate the derivative term, so set it to 0
        if abs(dt) < 1e-8:
            self._d_errors = np.zeros(self.n)
        else:
            self._d_errors = (self._p_errors - self._p_errors_last) / dt

        if self.using_gain_matrices:
            d_term = self._d_gains @ self._d_errors
        else:
            d_term = self._d_gains * self._d_errors

        # Store the proportional errors for calculating the derivative term in the next iteration
        self._p_errors_last = self._p_errors

        # Calculate the PID control command
        self._cmd = p_term + i_term + d_term

        # Unpack the np array if we are dealing with a singleton dimension
        # TODO decide if this is the right choice here
        if self.n == 1:
            self._cmd = self._cmd[0]
        return self._cmd


if __name__ == "__main__":
    # Quick check that it's working
    # See test/test_pid for proper test cases
    print("Scalar example with auto-timestep")
    controller = PID(1.0, 2.0, 3.0, -1.0, 1.0)
    print(controller)
    controller.update(-1)
    print(controller)
    controller.update(-0.5)
    print(controller)
    print("Array example with fixed timestep")
    controller = PID([1, 2], [0.3, 0.4], [0.5, 0.6], [-10, -20], [10, 20])
    print(controller)
    controller.update([1, -2], 1)
    print(controller)
    controller.update([0.5, -1], 1)
    print(controller)
