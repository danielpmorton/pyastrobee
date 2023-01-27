"""Functions related to loading Astrobee components

TODO need to confirm that:
- All of the ISS modules can be loaded properly
- When loading all of the modules at once, they're in the correct position and orientation
- No dummy objects are anywhere near the working environment (especially check this if you need to modify the
  orientation of any of the ISS parts)

TODO make a function to visualize the dummy objects
TODO get the cupola and node 1 VHACD working, then put in the right directories with the vhacd_ prefix
TODO check if there is an easier solution (no dummies) with a URDF or SDF? Multiple fixed links?
"""

import os

import pybullet
import numpy as np

from astrobee_pybullet.utils.bullet_utils import (
    initialize_pybullet,
    run_sim,
    load_floor,
)


def load_iss() -> list[int]:
    # NOT DONE
    # TODO need to confirm if these will be loaded in the correct positions or not,
    # or if there is more info needed from the URDF
    modules = ["cupola", "eu_lab", "jpm", "node_1", "node_2", "node_3", "us_lab"]

    dummy_radius = 0.01
    dummy_z_pos = -5
    load_floor(z_pos=dummy_z_pos + 3 * dummy_radius)
    ids = []
    for name in modules:
        module_id = load_iss_module(name)
        ids.append(module_id)
    return ids


def load_iss_module(
    name: str, dummy_z_pos: float = -5, dummy_radius: float = 0.01
) -> int:
    """Loads a single ISS module. For example, "us_lab"

    Note: this will also load some "dummy" collision objects into the environment. This is necessary to get all of the
    visuals for the ISS to load. These objects will be invisible and outside of the ISS workspace

    Args:
        name (str): The module to load. One of {"cupola", "eu_lab", "jpm", "node_1", "node_2", "node_3", "us_lab"}
        dummy_z_pos (float, optional): Z coordinate to place the dummy collision objects safely under the ISS.
            Defaults to -5.
        dummy_radius (float, optional): Radius of the dummy collision objects (spheres). Defaults to 0.01.

    Raises:
        ValueError: If an invalid ISS module name is provided
        NotADirectoryError: If the module's mesh directory cannot be found
        FileNotFoundError: If either the vhacd obj file or the obj2sdf objs cannot be found in the mesh directory
        Exception: If pybullet fails to properly load a visual or collision object

    Returns:
        int: ID of the body corresponding to the VHACD collision object
    """

    if name not in {"cupola", "eu_lab", "jpm", "node_1", "node_2", "node_3", "us_lab"}:
        raise ValueError(f"Invalid module name: {name}")

    # Get the paths for all files in the directory (visual and collision)
    cwd = os.getcwd()
    directory = f"{cwd}/astrobee_media/astrobee_iss/meshes/{name}"
    if not os.path.exists(directory):
        raise NotADirectoryError(
            f"{directory} is not valid.\nCheck on the input, {name}, or current working directory, {cwd}"
        )
    part_paths = []
    vhacd_path = ""
    for filename in os.listdir(directory):
        if filename.startswith("vhacd"):
            vhacd_path = os.path.join(directory, filename)
        elif filename.startswith("part"):
            part_paths.append(os.path.join(directory, filename))
        else:
            continue
    if not vhacd_path:
        raise FileNotFoundError("Could not find the VHACD OBJ file for collision info")
    if not part_paths:
        raise FileNotFoundError(
            "Could not find the OBJ part files for the ISS module visuals"
        )

    # Load the module:
    # Each part of the module will load the visual for the associated body
    # If we're dealing with the first part, load the VHACD file as the collision body
    # For the remaining parts, load a "dummy" collision object outside of the ISS workspace
    # Since the ISS is flat, we can put these dummy objects at a fixed z-position below the ISS

    num_parts = len(part_paths)
    num_dummies = num_parts - 1  # Since the first part will load the VHACD geometry
    # Create a set of positions for the dummy objects that are not colliding with eachother
    # This will look like a square grid of dummy objects (not fully filled out if the number of dummies is not square)
    n_along_edge = np.ceil(np.sqrt(num_dummies))
    spacings = 2.5 * dummy_radius * np.arange(n_along_edge)
    xgrid, ygrid = np.meshgrid(spacings, spacings)
    xs = xgrid.flatten()[:num_dummies]
    ys = ygrid.flatten()[:num_dummies]
    dummy_coords = np.column_stack((xs, ys))  # Stack for easy indexing

    ids = []
    for i, path in enumerate(part_paths):
        # Every part will have an associated visual shape
        # When the path points to an OBJ, it will load colors via the associated MTL file in the same directory
        visual_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=path,
        )
        if i == 0:
            # Load the VHACD results as the collision info
            collision_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_MESH, fileName=vhacd_path
            )
        else:
            # Load one of the dummies
            collision_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=0.1,
                collisionFramePosition=[*dummy_coords[i - 1], dummy_z_pos],
            )
        # Check to make sure things were loaded properly before forming the multibody
        if visual_id < 0:
            raise Exception(f"Pybullet could not load the visual shape for {path}")
        if collision_id < 0:
            raise Exception(f"Pybullet could not load the collision shape for {path}")

        rigid_id = pybullet.createMultiBody(
            baseMass=0,  # Fixed position
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
        )
        ids.append(rigid_id)
    return ids[0]  # Just the ID for the VHACD object


# TODO remove this once we're done with testing these functions
if __name__ == "__main__":
    initialize_pybullet(use_gui=True)
    load_iss_module("us_lab")
    load_floor(z_pos=-3)
    run_sim()
