"""Assorted helper functions related to math operations / linear algebra"""

from typing import Union

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


def skew(v: npt.ArrayLike) -> np.ndarray:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (npt.ArrayLike): Vector to convert, shape (3,)

    Returns:
        np.ndarray: (3, 3) skew-symmetric matrix
    """
    v = np.ravel(v)
    if len(v) != 3:
        raise ValueError(f"Vector needs to be of length 3.\nGot: {v}")
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def unskew(S: npt.ArrayLike) -> np.ndarray:
    """Gives the vector associated with a skew-symmetric matrix S such that skew(vec) = S

    Args:
        S (npt.ArrayLike): Skew-symmetric 3x3 matrix

    Returns:
        np.ndarray: "Unskewed" vector, shape (3,)
    """
    S = np.asarray(S)
    if S.shape != (3, 3):
        raise ValueError(f"S must be a 3x3 matrix. Got shape: {S.shape}")
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def is_skew(S: np.ndarray) -> bool:
    """Checks that a square matrix is skew-symmetric

    Args:
        S (np.ndarray): Matrix to check

    Returns:
        bool: True if skew-symmetric, False otherwise
    """
    return np.allclose(S, -S.T)


def is_special_orthogonal(R: np.ndarray) -> bool:
    """Checks that a square matrix is in the special orthogonal group

    Args:
        R (np.ndarray): Matrix to check

    Returns:
        bool: True if special orthogonal (such as a rotation matrix), False otherwise
    """
    m, n = R.shape
    if m != n:
        return False
    return np.allclose(R @ R.T, np.eye(n)) and np.isclose(np.linalg.det(R), 1)


def spherical_vonmises_sampling(
    mu: npt.ArrayLike, kappa: Union[float, npt.ArrayLike], n_pts: int
) -> np.ndarray:
    """Samples points on a (generalized) sphere based on the von Mises distribution

    This is slightly more relevant to sampling on a sphere than a Gaussian because the
    distribution is circular in nature. Technically, the von Mises-Fisher distribution is for
    a sphere but Numpy doesn't seem to distinguish between these.

    Args:
        mu (npt.ArrayLike): Mean. Length defines the dimension of the sphere. Should be normalized
        kappa (Union[float, npt.ArrayLike]): Concentration parameter. This can be thought of as the
            "inverse of variance". Large kappa results in a distribution that approaches a gaussian;
            small kappa approaches a uniform distribution. As a quick not-at-all precise measure,
            kappa = 0: Uniform distribution around the sphere
            kappa = 5: Angular dispersion of ~90 degrees from the mean
            kappa = 10: Angular dispersion of ~60 degrees from the mean
            kappa = 50: Angular dispersion of ~15 degrees from the mean
        n_pts (int): Number of points to sample

    Returns:
        np.ndarray: Sampled points, shape (n_pts, dimension)
    """
    mu = np.atleast_1d(mu)
    kappa = np.atleast_1d(kappa)
    sampled = np.random.vonmises(mu, kappa, size=(n_pts, len(mu)))
    return sampled / np.linalg.norm(sampled, axis=1).reshape(-1, 1)
