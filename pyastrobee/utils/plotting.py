"""Functions for plotting various data or debugging info"""

# TODO look into pytransform3d.plot_utils

import numpy as np


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
