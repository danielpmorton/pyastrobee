"""Boxes and associated utility functions

These are currently used as the definition of the safe sets in the trajectory optimization

Based on: https://github.com/cvxgrp/fastpathplanning/blob/main/fastpathplanning/boxes.py
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from pyastrobee.utils.bullet_utils import create_box


class Box:
    """Representation of a box defined by lower/upper limits on its coordinates

    Unpackable as (lower, upper) = box

    Args:
        lower (npt.ArrayLike): Lower limits on the box coordinates, shape (box_dim,)
        upper (npt.ArrayLike): Upper limits on the box coordinates, shape (box_dim,)
    """

    def __init__(self, lower: npt.ArrayLike, upper: npt.ArrayLike):
        self.lower = np.ravel(lower)
        self.upper = np.ravel(upper)
        self._validate()
        self.center = (self.lower + self.upper) / 2
        self.dim = len(self.lower)

    def __iter__(self):
        return iter([self.lower, self.upper])

    def _validate(self):
        if len(self.lower) != len(self.upper):
            raise ValueError("Invalid input dimensions")
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid inputs: Mismatched order of lower/upper points")


def intersect_boxes(b1: Box, b2: Box) -> Box:
    """Calculate the intersection of two boxes

    Args:
        b1 (Box): First box
        b2 (Box): Second box

    Returns:
        Box: The intersection region
    """
    return Box(np.maximum(b1.lower, b2.lower), np.minimum(b1.upper, b2.upper))


def check_box_intersection(b1: Box, b2: Box) -> bool:
    """Evaluate if two boxes intersect or not

    Args:
        b1 (Box): First box
        b2 (Box): Second box

    Returns:
        bool: True if the boxes intersect, False if not
    """
    l = np.maximum(b1.lower, b2.lower)
    u = np.minimum(b1.upper, b2.upper)
    return np.all(u >= l)


def is_in_box(point: npt.ArrayLike, box: Box) -> bool:
    """Evaluate if a point lies within a box

    Args:
        point (npt.ArrayLike): Point to evaluate, shape (box_dim,)
        box (Box): Box to test

    Returns:
        bool: True if the point is inside the bounds of the box, False otherwise
    """
    assert np.size(point) == box.dim
    return np.all(point >= box.lower) and np.all(point <= box.upper)


def find_containing_box(
    point: npt.ArrayLike, boxes: Union[list[Box], npt.ArrayLike]
) -> int | None:
    """Find the index of the first box which contains a certain point

    Args:
        point (npt.ArrayLike): Point to evaluate
        boxes (Union[list[Box], npt.ArrayLike]): Boxes to search. If an array, must be of shape (n_boxes, 2, box_dim)

    Returns:
        int | None: Index of the first box which contains the point. None if the point is not in any box
    """
    for i, box in enumerate(boxes):
        lower, upper = box
        if np.all(point >= lower) and np.all(point <= upper):
            return i
    return None


def visualize_3D_box(
    box: Union[Box, npt.ArrayLike],
    padding: Optional[npt.ArrayLike] = None,
    rgba: npt.ArrayLike = (1, 0, 0, 0.5),
) -> int:
    """Visualize a box in Pybullet

    Args:
        box (Union[Box, npt.ArrayLike]): Box to visualize. If an array, must be of shape (1, 2, box_dim)
        padding (Optional[npt.ArrayLike]): If expanding (or contracting) the boxes by a certain amount, include the
            (x, y, z) padding distances here (shape (3,)). Defaults to None.
        rgba (npt.ArrayLike): Color of the box (RGB + alpha), shape (4,). Defaults to (1, 0, 0, 0.5).

    Returns:
        int: Pybullet ID of the box
    """
    lower, upper = box
    if padding is not None:
        lower -= padding
        upper += padding
    return create_box(
        pos=(lower + (upper - lower) / 2),  # Midpoint
        orn=(0, 0, 0, 1),
        mass=0,
        sidelengths=(upper - lower),
        use_collision=False,
        rgba=rgba,
    )
