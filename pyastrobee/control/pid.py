"""Extending PID to a decoupled MIMO system where we control n different parameters

TODO update all docstrings
TODO add test cases

Based on pid.cpp (c) 2008, Willow Garage, Inc. (BSD License)
as well as Nathan Sprague's Python translation
"""

from typing import Optional, Union

import time
import math

import numpy as np
import numpy.typing as npt


class ExtendedPID:
    """A basic pid class.

    This class implements a generic structure that can be used to
    create a wide range of pid controllers. It can function
    independently or be subclassed to provide more specific controls
    based on a particular control loop.

    In particular, this class implements the standard pid equation:

    $command = p_{term} + i_{term} + d_{term} $

    where:

    $ p_{term} = p_{gain} * p_{error} $
    $ i_{term} = i_{gain} * i_{error} $
    $ d_{term} = d_{gain} * d_{error} $
    $ i_{error} = i_{error} + p_{error} * dt $
    $ d_{error} = (p_{error} - p_{error last}) / dt $

    given:

    $ p_{error} = p_{target} - p_{state} $.
    """

    def __init__(
        self,
        p_gains: npt.ArrayLike,
        i_gains: npt.ArrayLike,
        d_gains: npt.ArrayLike,
        i_mins: Optional[npt.ArrayLike] = None,
        i_maxes: Optional[npt.ArrayLike] = None,
    ):
        """Constructor, zeros out Pid values when created and
        initialize Pid-gains and integral term limits.

        Parameters:
          p_gain     The proportional gain.
          i_gain     The integral gain.
          d_gain     The derivative gain.
          i_min      The integral lower limit.
          i_max      The integral upper limit.
        """
        self.n = len(np.atleast_1d(p_gains))
        self.set_gains(p_gains, i_gains, d_gains, i_mins, i_maxes)
        self.reset()

    def reset(self):
        """Reset the state of this PID controller"""
        # Save position state for derivative state calculation
        self._p_errors_last = np.zeros(self.n)
        self._p_errors = np.zeros(self.n)  # Proportional error.
        self._d_errors = np.zeros(self.n)  # Derivative error.
        self._i_errors = np.zeros(self.n)  # Integator error.
        self._cmd = 0 if self.n == 1 else np.zeros(self.n)  # Command to send.
        self._last_time = None  # Used for automatic calculation of dt.

    def set_gains(
        self,
        p_gains: npt.ArrayLike,
        i_gains: npt.ArrayLike,
        d_gains: npt.ArrayLike,
        i_mins: Optional[npt.ArrayLike] = None,
        i_maxes: Optional[npt.ArrayLike] = None,
    ):
        """Set PID gains for the controller.

        Parameters:
         p_gain     The proportional gain.
         i_gain     The integral gain.
         d_gain     The derivative gain.
         i_min      The integral lower limit.
         i_max      The integral upper limit.
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

    def _validate_gains(self, gains: npt.ArrayLike) -> np.ndarray:
        gains = np.atleast_1d(gains)
        if len(gains) != self.n:
            raise ValueError(
                f"Invalid gains.\nExpected an array of length {self.n}, got {gains}"
            )
        if np.any(gains < 0):
            raise ValueError(f"Negative gains will cause instability.\nGot: {gains}")
        return gains

    def _validate_integral_limits(self, lims: npt.ArrayLike) -> np.ndarray:
        lims = np.atleast_1d(lims)
        if len(lims) != self.n:
            raise ValueError(
                f"Invalid integral limits.\nExpected an array of length {self.n}, got {lims}"
            )
        return lims

    @p_gains.setter
    def p_gains(self, gains: npt.ArrayLike):
        gains = self._validate_gains(gains)
        self._p_gains = gains

    @i_gains.setter
    def i_gains(self, gains: npt.ArrayLike):
        gains = self._validate_gains(gains)
        self._i_gains = gains

    @d_gains.setter
    def d_gains(self, gains: npt.ArrayLike):
        gains = self._validate_gains(gains)
        self._d_gains = gains

    @i_maxes.setter
    def i_maxes(self, lims: Optional[npt.ArrayLike]):
        if lims is None:
            lims = float("inf") * np.ones(self.n)
        else:
            lims = self._validate_integral_limits(lims)
        self._i_maxes = lims

    @i_mins.setter
    def i_mins(self, lims: Optional[npt.ArrayLike]):
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

    def update_PID(
        self, p_errors: np.ndarray, dt: Optional[float] = None
    ) -> Union[np.ndarray, float]:
        """Update the Pid loop with nonuniform time step size.

        Parameters:
          p_error  Error since last call (target - state)
          dt       Change in time since last call, in seconds, or None.
                   If dt is None, then the system clock will be used to
                   calculate the time since the last update.
        Returns:
          pid controller output
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
        p_term = self._p_gains * self._p_errors

        # Integral component
        self._i_errors += dt * self._p_errors
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
    controller = ExtendedPID(1.0, 2.0, 3.0, -1.0, 1.0)
    print(controller)
    controller.update_PID(-1)
    print(controller)
    controller.update_PID(-0.5)
    print(controller)
    controller = ExtendedPID([1, 2], [0.3, 0.4], [0.5, 0.6], [-10, -20], [10, 20])
    print(controller)
    controller.update_PID([1, -2])
    print(controller)
    controller.update_PID([0.5, -1])
    print(controller)
