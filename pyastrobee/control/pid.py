"""An extended version of a PID controller compatible with multivariate systems

Based on pid.cpp (c) 2008, Willow Garage, Inc. (BSD License)
as well as Nathan Sprague's Python translation
"""
# TODO decide if negative gains could actually be possible in the coupled case
# TODO determine how to handle coupled gains in the integral case

from typing import Optional, Union

import time
import math

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.utils.math_utils import is_diagonal, safe_divide
from pyastrobee.utils.plotting import num_subplots_to_shape


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
        p_gain (npt.ArrayLike): Proportional gain(s).
        i_gain (npt.ArrayLike): Integral gain(s).
        d_gain (npt.ArrayLike): Derivative gain(s).
        i_min (Optional[npt.ArrayLike]): Minimum value(s) for the integral term(s).
            Defaults to None, in which case the integral term(s) will be unbounded below
        i_max (Optional[npt.ArrayLike]): Maximum value(s) for the integral term(s).
            Defaults to None, in which case the integral term(s) will be unbounded above
    """

    def __init__(
        self,
        p_gain: npt.ArrayLike,
        i_gain: npt.ArrayLike,
        d_gain: npt.ArrayLike,
        i_min: Optional[npt.ArrayLike] = None,
        i_max: Optional[npt.ArrayLike] = None,
    ):
        # Use the proportional gain as our baseline for comparison
        # Calculate some shapes/sizes to help distinguish between scalar/array/matrix methods
        reference_gain = np.atleast_1d(p_gain)
        self._gain_shape = reference_gain.shape
        self.n = self._gain_shape[0]
        self.using_gain_matrices = reference_gain.ndim > 1
        self.set_gains(p_gain, i_gain, d_gain, i_min, i_max)
        self.reset()

    def reset(self):
        """Reset the state of the PID controller"""
        # Save position state for derivative state calculation
        self._p_error_last = np.zeros(self.n)
        self._p_error = np.zeros(self.n)  # Proportional error.
        self._d_error = np.zeros(self.n)  # Derivative error.
        self._i_error = np.zeros(self.n)  # Integator error.
        self._cmd = 0 if self.n == 1 else np.zeros(self.n)  # Command to send.
        self._last_time = None  # Used for automatic calculation of dt.

    def set_gains(
        self,
        p_gain: npt.ArrayLike,
        i_gain: npt.ArrayLike,
        d_gain: npt.ArrayLike,
        i_min: Optional[npt.ArrayLike] = None,
        i_max: Optional[npt.ArrayLike] = None,
    ):
        """Sets the PID gains for the controller.

        Gains are scalar for SISO, 1d array for decoupled MIMO, matrix for general MIMO
        Integral limits are scalar for SISO, 1d array for MIMO.

        Args:
            p_gain (npt.ArrayLike): Proportional gain(s).
            i_gain (npt.ArrayLike): Integral gain(s).
            d_gain (npt.ArrayLike): Derivative gain(s).
            i_min (Optional[npt.ArrayLike]): Minimum value(s) for the integral term(s).
                Defaults to None, in which case the integral term(s) will be unbounded below
            i_max (Optional[npt.ArrayLike]): Maximum value(s) for the integral term(s).
                Defaults to None, in which case the integral term(s) will be unbounded above
        """
        # Call the setter functions, which will auto-validate the inputs
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.i_min = i_min
        self.i_max = i_max

    @property
    def p_gain(self) -> np.ndarray:
        """Proportional gain(s)"""
        return self._p_gain

    @property
    def i_gain(self) -> np.ndarray:
        """Integral gain(s)"""
        return self._i_gain

    @property
    def d_gain(self) -> np.ndarray:
        """Derivative gain(s)"""
        return self._d_gain

    @property
    def i_max(self) -> np.ndarray:
        """Maximum value(s) of the integral term(s)"""
        return self._i_max

    @property
    def i_min(self) -> np.ndarray:
        """Minimum value(s) of the integral term(s)"""
        return self._i_min

    def _validate_gain(self, gain: npt.ArrayLike) -> np.ndarray:
        """Ensures that the gain(s) are the correct shape and are nonnegative (raises an error otherwise)

        Args:
            gain (npt.ArrayLike): P, I, or D gain(s)

        Returns:
            np.ndarray: Validated gain(s)
        """
        gain = np.atleast_1d(gain)
        if gain.shape != self._gain_shape:
            raise ValueError(
                f"Shape mismatch: Expected gain to have shape {self._gain_shape}, got {gain.shape}"
            )
        if np.any(gain < 0):
            raise ValueError(f"Negative gains will cause instability.\nGot: {gain}")
        return gain

    def _validate_integral_limits(self, lim: npt.ArrayLike) -> np.ndarray:
        """Ensures that the integral term limit(s) are the correct shape (raises an error otherwise)

        Args:
            lim (npt.ArrayLike): Limit(s) (minimum or maximum) on the integral term(s)

        Returns:
            np.ndarray: Validated limit(s)
        """
        lim = np.atleast_1d(lim)
        if len(lim) != self.n:
            raise ValueError(
                f"Invalid integral limit.\nExpected an array of length {self.n}, got {lim}"
            )
        return lim

    @p_gain.setter
    def p_gain(self, gain: npt.ArrayLike):
        gain = self._validate_gain(gain)
        self._p_gain = gain

    @i_gain.setter
    def i_gain(self, gain: npt.ArrayLike):
        gain = self._validate_gain(gain)
        if self.using_gain_matrices:
            # Extra logic to ensure that the gain matrix is diagonal (decoupled) because
            # determining the integral saturation for this case requires some more work
            if not is_diagonal(gain):
                raise ValueError(
                    "Coupling between the integral gains is not currently supported"
                )
        self._i_gain = gain

    @d_gain.setter
    def d_gain(self, gain: npt.ArrayLike):
        gain = self._validate_gain(gain)
        self._d_gain = gain

    @i_max.setter
    def i_max(self, lim: Optional[npt.ArrayLike]):
        if lim is None:
            lim = float("inf") * np.ones(self.n)
        else:
            lim = self._validate_integral_limits(lim)
        self._i_max = lim

    @i_min.setter
    def i_min(self, lim: Optional[npt.ArrayLike]):
        if lim is None:
            lim = float("-inf") * np.ones(self.n)
        else:
            lim = self._validate_integral_limits(lim)
        self._i_min = lim

    @property
    def p_error(self) -> np.ndarray:
        """Proportional error(s) (Read-only)"""
        return self._p_error

    @property
    def i_error(self) -> np.ndarray:
        """Integral error(s) (Read-only)"""
        return self._i_error

    @property
    def d_error(self) -> np.ndarray:
        """Derivative error(s) (Read-only)"""
        return self._d_error

    @property
    def cmd(self) -> np.ndarray:
        """Read-only access to the latest command."""
        return self._cmd

    def __str__(self):
        """String representation of the current state of the controller."""
        result = ""
        result += "p_gain:  " + str(self.p_gain) + "\n"
        result += "i_gain:  " + str(self.i_gain) + "\n"
        result += "d_gain:  " + str(self.d_gain) + "\n"
        result += "i_min:   " + str(self.i_min) + "\n"
        result += "i_max:   " + str(self.i_max) + "\n"
        result += "p_error: " + str(self.p_error) + "\n"
        result += "i_error: " + str(self.i_error) + "\n"
        result += "d_error: " + str(self.d_error) + "\n"
        result += "cmd:     " + str(self.cmd) + "\n"
        return result

    def update(
        self, p_error: npt.ArrayLike, dt: Optional[float] = None
    ) -> npt.ArrayLike:
        """Update the PID controller state

        Args:
            p_error (npt.ArrayLike): Error since last iteration (target - state)
            dt (Optional[float]): Change in time since the last iteration, in seconds. Defaults to None,
                in which case the system clock will be used to determine the delta

        Returns:
            npt.ArrayLike: PID controller output. Array if we have a MIMO system, float if SISO
        """
        # Validate error input
        p_error = np.atleast_1d(p_error)
        if len(p_error) != self.n:
            raise ValueError(
                f"Invalid error term.\nExpected an array of length {self.n}, got {p_error}"
            )
        self._p_error = p_error

        # Validate dt input
        if dt is None:
            cur_time = time.time()
            if self._last_time is None:
                self._last_time = cur_time
                # TODO decide why they put this line here (below)
                self._p_error_last = p_error
            dt = cur_time - self._last_time
            self._last_time = cur_time
        assert not math.isnan(dt) and not math.isinf(dt)

        # Proportional component
        if self.using_gain_matrices:
            p_term = self._p_gain @ self._p_error
        else:
            p_term = self._p_gain * self._p_error

        # Integral component
        self._i_error += dt * self._p_error

        if self.using_gain_matrices:
            i_term = self._i_gain @ self._i_error
            i_term = np.clip(i_term, self._i_min, self._i_max)
            i_gain_diag = np.diag(self._i_gain)
            self._i_error = safe_divide(i_term, i_gain_diag, fill="original")
        else:
            i_term = self._i_gain * self._i_error
            # Saturate the integral term to prevent it from getting too large
            i_term = np.clip(i_term, self._i_min, self._i_max)
            # Update the integral error based on the saturated value, using safe division where gain is nonzero
            self._i_error = safe_divide(i_term, self._i_gain, fill="original")

        # Derivative component
        # If the timestep is 0, we can't calculate the derivative term, so set it to 0
        if abs(dt) < 1e-8:
            self._d_error = np.zeros(self.n)
        else:
            self._d_error = (self._p_error - self._p_error_last) / dt

        if self.using_gain_matrices:
            d_term = self._d_gain @ self._d_error
        else:
            d_term = self._d_gain * self._d_error

        # Store the proportional errors for calculating the derivative term in the next iteration
        self._p_error_last = self._p_error

        # Calculate the PID control command
        self._cmd = p_term + i_term + d_term

        # Unpack the np array if we are dealing with a singleton dimension
        # TODO decide if this is the right choice here
        if self.n == 1:
            self._cmd = self._cmd[0]
        return self._cmd


def plot_controller_history(
    history: np.ndarray,
    target: Union[npt.ArrayLike, float],
    p_gain: Union[npt.ArrayLike, float, None] = None,
    i_gain: Union[npt.ArrayLike, float, None] = None,
    d_gain: Union[npt.ArrayLike, float, None] = None,
):
    """Plots the state history of variables controlled by a PID controller

    Args:
        history (np.ndarray): History of the variables through the duration of the control process,
            in an array of shape (num_variables, num_timesteps)
        target (Union[npt.ArrayLike, float]): Desired value(s) for the controlled variables
        p_gain (Union[npt.ArrayLike, float, None], optional): Proportional gain(s).
            Include this to print the gain value(s) onto the subplots. Defaults to None.
        i_gain (Union[npt.ArrayLike, float, None], optional): Integral gain(s). Defaults to None.
            Include this to print the gain value(s) onto the subplots. Defaults to None.
        d_gain (Union[npt.ArrayLike, float, None], optional): Derivative gain(s). Defaults to None.
            Include this to print the gain value(s) onto the subplots. Defaults to None.
    """
    # Ensure the dimensions of the arrays so that indexing works properly
    history = np.atleast_2d(history)
    target = np.atleast_1d(target)
    if p_gain is not None:
        p_gain = np.atleast_1d(p_gain)
    if i_gain is not None:
        i_gain = np.atleast_1d(i_gain)
    if d_gain is not None:
        d_gain = np.atleast_1d(d_gain)
    n, nt = history.shape
    fig_shape = num_subplots_to_shape(n)
    plt.figure()
    # Plot the history of each variable on its own subplot
    for i in range(n):
        ax = plt.subplot(*fig_shape, i + 1)
        timesteps = range(nt)
        plt.plot(timesteps, target[i] * np.ones(nt), "k--")
        plt.plot(timesteps, history[i, :], "b")
        title = f"Var. {i+1}"
        # If the P/I/D gains are provided, include info about the gains for that
        # control variable in the bottom right corner of the respective subplot
        pid_info = ""
        if p_gain is not None:
            pid_info += f"P: {round(p_gain[i], 3)}\n"
        if i_gain is not None:
            pid_info += f"I: {round(i_gain[i], 3)}\n"
        if d_gain is not None:
            pid_info += f"D: {round(d_gain[i], 3)}"
        plt.text(
            0.98,
            0.02,
            pid_info,
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )
        plt.title(title)
    plt.show()


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
