"""Helper functions associated with transformations (rotations + translations)

Notation for transformation matrices:
A
  T
B
describes the frame B with respect to frame A. AKA "B in A", or "B to A"
In code, this can be described as T_B_in_A or T_B2A

Composing transformations:
A       A     B     C
  T  =    T     T     T
D       B     C     D
e.g. T_D2A = T_B2A @ T_C2B @ T_D2C

All angles are in radians
"""

from typing import Union

import numpy as np

from pyastrobee.utils.rotations import check_rotation_mat


def make_transform_mat(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Creates a transformation matrix from a rotation matrix and translation vector

    TODO add checks that the rotation matrix is valid, and that the translation
    is the correct size

    Args:
        rot (np.ndarray): (3,3) rotation matrix
        trans (np.ndarray): (3,) translation vector

    Returns:
        np.ndarray: Transformation matrix, shape (4, 4)
    """
    T = np.eye(4, 4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def check_transform_mat(T: np.ndarray) -> bool:
    """Checks to see if a transformation matrix is valid

    Args:
        T (np.ndarray): (4, 4) transformation matrix

    Returns:
        bool: Whether or not the transformation matrix is valid
    """
    if not T.shape == (4, 4):
        return False
    rot = T[:3, :3]
    last_row = T[3, :]
    return check_rotation_mat(rot) and np.array_equal(
        last_row, np.array([0.0, 0.0, 0.0, 1.0])
    )


def invert_transform_mat(T: np.ndarray) -> np.ndarray:
    """Inverts a transformation matrix

    Example: T_B2A = invert_transform_mat(T_A2B)

    Args:
        T (np.ndarray): (4, 4) transformation matrix

    Returns:
        np.ndarray: (4, 4) matrix representing the inverse transform of T
    """
    rot = T[:3, :3]
    trans = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = np.transpose(rot)
    T_inv[:3, 3] = -np.transpose(rot) @ trans
    return T_inv


def transform_point(
    tmat: np.ndarray, point: Union[list[float], np.ndarray]
) -> np.ndarray:
    """Applies a transformation to a point

    As a mapping: Changes the description of the point between frames
        Example: point_in_B = transform_point(T_A2B, point_in_A)
    As an operator: Moves a point within the same frame
        Example: new_point = transform_point(transform, orig_point)

    TODO: validate inputs?

    Args:
        tmat (np.ndarray): (4, 4) transformation matrix
        point (Union[list[float], np.ndarray]): (3,) point to transform

    Returns:
        np.ndarray: (3,) transformed point
    """
    p = np.append(point, 1)  # Convert to (4,) array
    return (tmat @ p)[:3]
