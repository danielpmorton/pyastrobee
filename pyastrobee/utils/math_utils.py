"""Assorted helper functions related to math operations / linear algebra"""


import numpy as np
import numpy.typing as npt


def normalize(vec: npt.ArrayLike) -> np.ndarray:
    """Normalizes a vector to have magnitude 1

    Args:
        vec (npt.ArrayLike): Input vector

    Returns:
        np.ndarray: A unit vector in the same direction as the input
    """
    return np.array(vec) / np.linalg.norm(vec)


def is_diagonal(array: npt.ArrayLike) -> bool:
    """Checks if an array is diagonal or not

    Args:
        array (npt.ArrayLike): The array to check

    Returns:
        bool: True if the array is diagonal, False otherwise
    """
    return np.array_equal(array, np.diag(np.diag(array)))
