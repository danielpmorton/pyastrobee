"""Functions for plotting various data or debugging info

TODO look into pytransform3d.plot_utils
"""

from typing import Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# from pytransform3d.transformations import plot_transform
# from pytransform3d.plot_utils import make_3d_axis


def plot_astrobee_frames():
    raise NotImplementedError


def plot_trajectory():
    raise NotImplementedError


def plot_controller_history(
    history: np.ndarray,
    target: Union[npt.ArrayLike, float],
    p_gain: Union[npt.ArrayLike, float, None] = None,
    i_gain: Union[npt.ArrayLike, float, None] = None,
    d_gain: Union[npt.ArrayLike, float, None] = None,
):
    """Plots the state history of variables controlled by a PID controller

    TODO: add names of the variables? Actual times rather than timesteps?

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


def num_subplots_to_shape(n: int) -> tuple[int, int]:
    """Determines the best layout of a number of subplots within a larger figure

    Args:
        n (int): Number of subplots

    Returns:
        tuple[int, int]: Number of rows and columns for the subplot divisions
    """
    n_rows = int(np.sqrt(n))
    n_cols = n // n_rows + (n % n_rows > 0)
    assert n_rows * n_cols >= n
    return (n_rows, n_cols)
