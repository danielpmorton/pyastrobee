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
