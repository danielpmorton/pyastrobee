"""Functions related to loading Astrobee components

Notes:
- If any of the ISS meshes are updated (such as a reorientation or repositioning in Blender), the default orientation
    should be changed in the loading functions
- If the mesh directory gets changed, the hardcoded relative paths need to be updated

TODO check if there is an easier solution (no dummies) with a URDF or SDF? Multiple fixed links?
TODO check with Gazebo to see if we should be positioning the ISS in the exact same position as in NASA's sim
"""

import os
from enum import Enum

import pybullet
import numpy as np
import numpy.typing as npt

from pyastrobee.utils.bullet_utils import (
    initialize_pybullet,
    run_sim,
    load_floor,
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


def load_iss(orn: npt.ArrayLike = [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]) -> list[int]:
    """Loads all modules of the ISS into pybullet

    Args:
        orn (npt.ArrayLike, optional): Orientation of the ISS (XYZW quaternion). Defaults to
            [sqrt(2)/2, 0, 0, sqrt(2)/2] (a 90-degree rotation in x). This will orient the ISS so
            it lays flat in the simulator.

    Returns:
        list[int]: The pybullet IDs for each of the modules' collision body
    """
    ids = []
    for module in ISSModule:
        module_id = load_iss_module(module, orn=orn)
        ids.append(module_id)
    print_green("ISS is ready")
    return ids


def load_iss_module(
    module: ISSModule,
    orn: npt.ArrayLike = [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
    dummy_z_pos: float = -5,
    dummy_radius: float = 0.01,
) -> int:
    """Loads a single ISS module. For example, US_LAB

    Note: this will also load some "dummy" collision objects into the environment. This is necessary to get all of the
    visuals for the ISS to load. These objects will be invisible and outside of the ISS workspace

    Args:
        module (ISSModule): The module to load. For example, ISSModule.CUPOLA / .EU_LAB / .JPM / ...
        orn (npt.ArrayLike, optional): Orientation of the ISS module (XYZW quaternion). Defaults to
            [sqrt(2)/2, 0, 0, sqrt(2)/2] (a 90-degree rotation in x). This will orient the module so
            it lays flat in the simulator.
        dummy_z_pos (float, optional): Z coordinate to place the dummy collision objects safely under the ISS.
            Defaults to -5.
        dummy_radius (float, optional): Radius of the dummy collision objects (spheres). Defaults to 0.01.

    Raises:
        PybulletError: If pybullet fails to properly load a visual or collision object

    Returns:
        int: ID of the body corresponding to the VHACD collision object
    """
    # Locate the paths to all of the meshes for the module
    vhacd_path, part_paths = _find_mesh_files(module)

    # Load the module:
    # Each part of the module will load the visual for the associated body
    # If we're dealing with the first part, load the VHACD file as the collision body
    # For the remaining parts, load a "dummy" collision object outside of the ISS workspace
    # Since the ISS is flat, we can put these dummy objects at a fixed z-position below the ISS

    num_parts = len(part_paths)
    num_dummies = num_parts - 1  # Since the first part will load the VHACD geometry
    # Create a set of positions for the dummy objects that are not colliding with eachother
    # This will look like grid of dummy objects for each module
    # For the x values, add the Enum value to separate the dummies for each module, so they don't overlap
    n_along_edge = np.ceil(np.sqrt(num_dummies))
    spacings = 2.5 * dummy_radius * np.arange(n_along_edge)
    xgrid, ygrid = np.meshgrid(spacings, spacings)
    xs = module.value + xgrid.flatten()[:num_dummies]
    ys = ygrid.flatten()[:num_dummies]
    dummy_coords = np.column_stack((xs, ys))  # Stack for easy indexing

    ids = []
    for i, path in enumerate(part_paths):
        # Every part will have an associated visual shape
        # When the path points to an OBJ, it will load colors via the associated MTL file in the same directory
        visual_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH, fileName=path, visualFrameOrientation=orn
        )
        if i == 0:
            # Load the VHACD results as the collision info
            collision_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=vhacd_path,
                collisionFrameOrientation=orn,
            )
        else:
            # Load one of the dummies
            collision_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=dummy_radius,
                collisionFramePosition=[*dummy_coords[i - 1], dummy_z_pos],
                collisionFrameOrientation=orn,
            )
        # Check to make sure things were loaded properly before forming the multibody
        if visual_id < 0:
            raise PybulletError(
                f"Could not load the visual shape for {path}", visual_id
            )
        if collision_id < 0:
            raise PybulletError(
                f"Could not load the collision shape for {path}", collision_id
            )

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


# Keeping the logic for the debugging functions separate from the main load_iss() function for now
# If it was integrated, it would make a mess of the readability
# Just make sure that if any logical updates were made in load_iss(), they are reflected here
def _debug_iss_module(
    module: ISSModule,
    dummy_z_pos: float = -5,
    dummy_radius: float = 0.01,
    orn=[0, 0, 0, 1],
) -> int:
    """Debugging version of load_iss_module(): Same functionality, but will visualize just the collision bodies

    Args:
        module (ISSModule): The module to load. For example, ISSModule.CUPOLA / .EU_LAB / .JPM / ...
        orn (npt.ArrayLike, optional): Orientation of the ISS module (XYZW quaternion). Defaults to
            [sqrt(2)/2, 0, 0, sqrt(2)/2] (a 90-degree rotation in x). This will orient the module so
            it lays flat in the simulator.
        dummy_z_pos (float, optional): Z coordinate to place the dummy collision objects safely under the ISS.
            Defaults to -5.
        dummy_radius (float, optional): Radius of the dummy collision objects (spheres). Defaults to 0.01.

    Returns:
        int: ID of the body corresponding to the VHACD collision object
    """
    # Locate the paths to all of the meshes for the module
    vhacd_path, part_paths = _find_mesh_files(module)

    # Set up dummies (see the non-debugging function for more info)
    num_parts = len(part_paths)
    num_dummies = num_parts - 1  # Since the first part will load the VHACD geometry
    n_along_edge = np.ceil(np.sqrt(num_dummies))
    spacings = 2.5 * dummy_radius * np.arange(n_along_edge)
    xgrid, ygrid = np.meshgrid(spacings, spacings)
    xs = module.value + xgrid.flatten()[:num_dummies]
    ys = ygrid.flatten()[:num_dummies]
    dummy_coords = np.column_stack((xs, ys))  # Stack for easy indexing

    ids = []
    for i, _ in enumerate(part_paths):
        if i == 0:
            # Load the VHACD result as the collision and the visual
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
        else:
            # Load the dummies as the collision and the visual
            visual_id = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=dummy_radius,
                visualFramePosition=[*dummy_coords[i - 1], dummy_z_pos],
                visualFrameOrientation=orn,
            )
            collision_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=dummy_radius,
                collisionFramePosition=[*dummy_coords[i - 1], dummy_z_pos],
                collisionFrameOrientation=orn,
            )
        rigid_id = pybullet.createMultiBody(
            baseMass=0,  # Fixed position
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            baseInertialFrameOrientation=orn,
        )
        ids.append(rigid_id)
    return ids[0]


def _debug_iss(orn: npt.ArrayLike = [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2]):
    """Debugging version of load_iss(): Same functionality, but will visualize just the collision bodies

    Args:
        orn (npt.ArrayLike, optional): Orientation of the ISS (XYZW quaternion). Defaults to
            [sqrt(2)/2, 0, 0, sqrt(2)/2] (a 90-degree rotation in x). This will orient the ISS so
            it lays flat in the simulator.

    Returns:
        list[int]: The pybullet IDs for each of the modules' collision body
    """
    dummy_radius = 0.01
    dummy_z_pos = -5
    # Load the floor above the "dummy" collision objects we needed to make to get the textures to load
    load_floor(z_pos=dummy_z_pos + 3 * dummy_radius)
    ids = []
    for module in ISSModule:
        module_id = _debug_iss_module(module, orn=orn)
        ids.append(module_id)
    return ids


if __name__ == "__main__":
    initialize_pybullet()
    load_iss()
    # _debug_iss()
    run_sim()
