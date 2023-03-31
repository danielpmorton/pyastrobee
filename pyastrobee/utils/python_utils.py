"""Utility functions broadly related to python as a whole

TODO determine if this should be separated into more specific files - e.g. debug.py
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def print_red(message: Any):
    """Helper function for printing in red text

    Args:
        message (Any): The message to print out in red
    """
    print(f"\033[31m{message}\033[0m")


def print_green(message: Any):
    """Helper function for printing in green text

    Args:
        message (Any): The message to print out in green
    """
    print(f"\033[32m{message}\033[0m")


def set_small_vals_to_zero(arr: npt.ArrayLike, tol: float = 1e-10) -> np.ndarray:
    """Array formatting helper function, improves legibility of printed array with floating "zeros"

    Args:
        arr (npt.ArrayLike): Array that would normally get printed in scientific notation with very small values
        tol (float, optional): Tolerance on determining when a float should = 0. Defaults to 1e-10.

    Returns:
        np.ndarray: Array with small floating values set to exactly 0
    """
    arr = np.array(arr)
    arr[np.abs(arr) < tol] = 0
    return arr
