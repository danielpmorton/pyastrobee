import os
import time
from typing import Union, Optional
import pybullet as p
import pybullet_data


# TODO add a function that would check to see if a pybullet client is active?


def set_gravity(location: str = "Earth") -> None:
    """_summary_

    Args:
        location (str, optional): _description_. Defaults to "Earth".

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    if location.lower() == "earth":
        return p.setGravity(0, 0, -9.81)
    elif location.lower() == "space":
        # Pybullet defaults to 0 gravity, but this option can allow us to reset it to 0-g
        # if it had previously been changed
        return p.setGravity(0, 0, 0)
    elif location.lower() == "moon":
        return p.setGravity(0, 0, -1.62)
    else:
        raise Exception("Invalid location")


def load_ground() -> int:
    """Loads a flat ground plane into pybullet

    Returns:
        int: The body ID for the ground plane
    """
    return load_urdf("plane.urdf")


def initialize_pybullet(use_gui: bool = True) -> int:
    """_summary_

    Args:
        use_gui (bool, optional): _description_. Defaults to True.

    Returns:
        int: A Physics Client ID
    """
    if use_gui:
        client = p.connect(p.GUI)
    else:
        client = p.connect(p.DIRECT)
    # setAdditionalSearchPath is in theory optional -- TODO check this
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return client


def load_urdf(
    urdf_loc: str, pos: Optional[list] = None, orn: Optional[list] = None
) -> int:
    """_summary_

    TODO add better typing for the inputs
    TODO add other parameters - potentially GlobalScaling might be useful

    Args:
        urdf_loc (str): _description_
        pos (_type_): _description_
        orn (_type_): _description_

    Returns:
        int: A unique ID for the body which was loaded (
    """
    return p.loadURDF(urdf_loc, pos, orn)
