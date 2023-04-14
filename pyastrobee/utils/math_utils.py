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
    norm = np.linalg.norm(vec)
    if abs(norm) < 1e-12:
        raise ZeroDivisionError("Cannot normalize the vector, it has norm 0")
    return np.array(vec) / norm


def is_diagonal(array: npt.ArrayLike) -> bool:
    """Checks if an array is diagonal or not

    Args:
        array (npt.ArrayLike): The array to check

    Returns:
        bool: True if the array is diagonal, False otherwise
    """
    return np.array_equal(array, np.diag(np.diag(array)))


def safe_divide(
    a: npt.ArrayLike, b: npt.ArrayLike, fill: str = "original", tol: float = 1e-8
) -> npt.ArrayLike:
    """Calculates a/b while handling division by zero

    Args:
        a (npt.ArrayLike): Dividend
        b (npt.ArrayLike): Divisor (which may contain zeros)
        fill (str, optional): How to fill the result where the divisor = 0. One of "original" (a/0 = a),
            "zero" (a/0 = 0), "nan" (a/0 = nan), "inf" (a/0 = inf). Defaults to "original".
        tol (float, optional): Tolerance for determining float equality with 0. Defaults to 1e-8.

    Returns:
        npt.ArrayLike: Division result
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    fill = fill.lower()
    if fill == "original":
        out = a
    elif fill == "zero":
        out = np.zeros_like(a)
    elif fill == "nan":
        out = np.full(a.shape, np.nan)
    elif fill == "inf":
        # Account for sign of dividend for positive or negative inf
        out = np.array((np.sign(a) + (a == 0)) * np.inf)
    else:
        raise ValueError(f"Invalid fill type: {fill}")
    return np.divide(a, b, out=out, where=np.abs(b) >= tol)


def skew(v: npt.ArrayLike) -> np.ndarray:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (npt.ArrayLike): Vector to convert, shape (3,)

    Returns:
        np.ndarray: (3, 3) skew-symmetric matrix
    """
    v = np.asarray(v)
    if len(v) != 3:
        raise ValueError(f"Vector needs to be of length 3.\nGot: {v}")
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
