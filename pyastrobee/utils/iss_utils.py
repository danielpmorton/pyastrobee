"""Functions related to loading the ISS modules into pybullet

Notes:
- If any of the ISS meshes are updated (such as a reorientation or repositioning in Blender), the default orientation
    should be changed in the loading functions
- If the mesh directory gets changed, the hardcoded relative paths need to be updated
"""

import os
from enum import Enum

import pybullet
import numpy as np
import numpy.typing as npt

from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    run_sim,
)
from pyastrobee.utils.errors import PybulletError
from pyastrobee.utils.python_utils import print_green


class ISSModule(Enum):
    """Enumerates the different ISS modules

    - The naming of these corresponds with NASA code and mesh filenames (typically lowercase)
    """

    CUPOLA = 0
    EU_LAB = 1
    JPM = 2
    NODE_1 = 3
    NODE_2 = 4
    NODE_3 = 5
    US_LAB = 6


def load_iss(
    orn: npt.ArrayLike = [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2], debug: bool = False
) -> list[int]:
    """Loads all modules of the ISS into pybullet

    Args:
        orn (npt.ArrayLike, optional): Orientation of the ISS (XYZW quaternion). Defaults to
            [sqrt(2)/2, 0, 0, sqrt(2)/2] (a 90-degree rotation in x). This will orient the ISS so
            it lays flat in the simulator.
        debug (bool, optional): Whether or not we are in debug-collision-bodies mode. If True, this will just
            visualize the collision bodies. Defaults to False

    Returns:
        list[int]: The pybullet IDs for each of the modules' collision body
    """
    ids = []
    for module in ISSModule:
        module_id = load_iss_module(module, orn, debug)
        ids.append(module_id)
    print_green("ISS is ready")
    return ids


def load_iss_module(
    module: ISSModule,
    orn: npt.ArrayLike = [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
    debug: bool = False,
) -> int:
    """Loads a single ISS module. For example, US_LAB

    Args:
        module (ISSModule): The module to load. For example, ISSModule.CUPOLA / .EU_LAB / .JPM / ...
        orn (npt.ArrayLike, optional): Orientation of the ISS module (XYZW quaternion). Defaults to
            [sqrt(2)/2, 0, 0, sqrt(2)/2] (a 90-degree rotation in x). This will orient the module so
            it lays flat in the simulator.
        debug (bool, optional): Whether or not we are in debug-collision-bodies mode. If True, this will just
            visualize the module's collision body. Defaults to False

    Raises:
        PybulletError: If pybullet fails to properly load a visual or collision object

    Returns:
        int: ID of the body corresponding to the VHACD collision object
    """
    # Locate the paths to all of the meshes for the module
    vhacd_path, part_paths = _find_mesh_files(module)

    # If we're debugging the collision info, just load the VHACD results as both the collision and visual
    if debug:
        visual_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=vhacd_path,
            visualFrameOrientation=orn,
        )
        collision_id = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=vhacd_path,
            collisionFrameOrientation=orn,
        )
        rigid_id = pybullet.createMultiBody(
            baseMass=0,  # Fixed position
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            baseInertialFrameOrientation=orn,
        )
        return rigid_id

    # Load the module:
    # Each part of the module will load the visual for the associated body
    # If we're dealing with the first part, load the VHACD file as the collision body.
    # For the remaining parts, we won't provide any collision information

    ids = []
    for i, path in enumerate(part_paths):
        # Every part will have an associated visual shape
        # When the path points to an OBJ, it will load colors via the associated MTL file in the same directory
        visual_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH, fileName=path, visualFrameOrientation=orn
        )
        if visual_id < 0:
            raise PybulletError(
                f"Could not load the visual shape for {path}", visual_id
            )
        if i == 0:
            # Load the VHACD results as the collision info
            collision_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=vhacd_path,
                collisionFrameOrientation=orn,
            )
            if collision_id < 0:
                raise PybulletError(
                    f"Could not load the collision shape for {path}", collision_id
                )
        else:
            collision_id = -1  # -1 means no collision bodies will be generated
        rigid_id = pybullet.createMultiBody(
            baseMass=0,  # Fixed position
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            baseInertialFrameOrientation=orn,
        )
        ids.append(rigid_id)
    return ids[0]  # Just the ID for the VHACD object


def _find_mesh_files(module: ISSModule) -> tuple[str, list[str]]:
    """Helper function to locate the paths to the ISS module meshes

    Args:
        module (ISSModule): The module to load. For example, ISSModule.CUPOLA / .EU_LAB / .JPM / ...

    Raises:
        ValueError: If an invalid ISS module name is provided
        NotADirectoryError: If the module's mesh directory cannot be found
        FileNotFoundError: If either the vhacd obj file or the obj2sdf objs cannot be found in the mesh directory

    Returns:
        Tuple of:
            str: The path to the VHACD collision mesh for the module
            list[str]: Paths to all of the decomposed visual meshes for the module
    """
    # Extract the name from the enum
    module_name = module.name.lower()

    # Get the paths for all files in the directory (visual and collision)
    cwd = os.getcwd()
    directory = f"{cwd}/pyastrobee/assets/meshes/iss/obj/{module_name}"
    if not os.path.exists(directory):
        raise NotADirectoryError(
            f"{directory} is not valid.\nCheck on the input, {module}, or current working directory, {cwd}"
        )
    part_paths = []
    vhacd_path = ""
    for filename in os.listdir(directory):
        if filename == "decomp.obj":
            vhacd_path = os.path.join(directory, filename)
        elif filename.startswith("part"):
            part_paths.append(os.path.join(directory, filename))
        else:
            continue
    if not vhacd_path:
        raise FileNotFoundError("Could not find the VHACD OBJ file for collision info")
    if not part_paths:
        raise FileNotFoundError("Could not find the OBJ files for the ISS visual info")
    return vhacd_path, part_paths


if __name__ == "__main__":
    initialize_pybullet()
    load_iss(debug=False)
    run_sim()
