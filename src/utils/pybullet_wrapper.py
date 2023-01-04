"""Wrapper for pybullet to provide more control, usability, and information for their functions

TODO:
- Decide where the best location is for importing/searching the pybullet_data
"""

from typing import Union

import pybullet as p
import pybullet_data


def set_gravity(location: str = "Earth") -> None:
    """Sets the gravity of the pybullet simulation

    Args:
        location (str, optional): Location of the simulation (e.g. Earth, Space, Moon, ...). Defaults to "Earth".
    """
    if location.lower() == "earth":
        return p.setGravity(0, 0, -9.81)
    elif location.lower() in {"space", "iss"}:
        # Pybullet defaults to 0 gravity, but this option can allow us to reset it to 0-g
        # if it had previously been changed
        return p.setGravity(0, 0, 0)
    elif location.lower() == "moon":
        return p.setGravity(0, 0, -1.62)
    else:
        raise Exception("Invalid location")


def load_ground() -> int:
    """Loads a flat ground plane into pybullet. (This requires pybullet_data)

    Returns:
        int: The body ID for the ground plane
    """
    load_urdf("plane.urdf")


def initialize_pybullet(use_gui: bool = True, use_pybullet_data: bool = True) -> int:
    """Starts a pybullet client

    Args:
        use_gui (bool, optional): Whether or not to use the GUI as opposed to headless. Defaults to True
        use_pybullet_data (bool, optional): Whether or not to also search through pybullet's provided models.
            Defaults to True

    Returns:
        int: A Physics Client ID
    """
    if use_gui:
        client = p.connect(p.GUI)
    else:
        client = p.connect(p.DIRECT)
    if use_pybullet_data:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return client


def load_urdf(
    urdf_loc: str,
    pos: Union[list, tuple] = (0.0, 0.0, 0.0),
    orn: Union[list, tuple] = (0.0, 0.0, 0.0, 1.0),
    fixed: bool = False,
) -> int:
    """Loads a URDF (via a specified file location) into pybullet

    Args:
        urdf_loc (str): The location of the URDF file
        pos (Union[list, tuple]): Starting xyz position of the model
        orn (Union[list, tuple]): Quaternions to specify the starting orientation of the model
        fixed (bool): Whether or not the model should be fixed in place

    Returns:
        int: A unique ID for the body which was loaded
    """
    if fixed:
        return p.loadURDF(urdf_loc, pos, orn, useFixedBase=True)
    return p.loadURDF(urdf_loc, pos, orn)


def check_pybullet_connection():
    return p.is_connected()
