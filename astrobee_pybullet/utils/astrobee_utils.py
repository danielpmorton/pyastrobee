"""Functions related to loading Astrobee components

TODO
- Make sure the vhacd files are in the correct dfirectories with the correct vhacd_ prefix
- Debug "Unsupported texture image format" messages
- Debug the VisualShapeArray - Unsure if this will actually be usable since it seems to have the same issue where
  you can only load one texture at a time
- If it comes down to it and you have to load all of the visual shapes as separate entities (rather than a nice multibody), 
  could just add an "empty" collision body to each one?
"""
import os

import pybullet

from astrobee_pybullet.utils.bullet_utils import (
    initialize_pybullet,
    load_deformable_object,
    load_rigid_object,
    run_sim,
)


def load_iss():
    # TODO need to confirm if these will be loaded in the correct positions or not,
    # or if there is more info needed from the URDF
    modules = ["cupola", "eu_lab", "jpm", "node_1", "node_2", "node_3", "us_lab"]
    ids = []
    for name in modules:
        module_id = load_iss_module(name)
        ids.append(module_id)
    return ids


def load_iss_module(name: str) -> int:
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
    # createVisualShapeArray() is not well documented, but the following kwargs appear to be supported:
    # shapeTypes, halfExtents, radii, lengths, fileNames, meshScales, rgbaColors, visualFramePositions,
    # visualFrameOrientations, physicsClientId
    # See "urdfEditor.py" or "createVisualShapeArray.py" in Bullet3 for more info
    shape_types = [pybullet.GEOM_MESH for _ in range(len(part_paths))]
    visual_id = pybullet.createVisualShapeArray(
        shapeTypes=shape_types, fileNames=part_paths
    )
    if visual_id < 0:
        raise Exception("Pybullet could not load the visual shape array")

    collision_id = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_MESH, fileName=vhacd_path
    )
    rigid_id = pybullet.createMultiBody(
        baseMass=0,  # Fixed position
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    # TODO need to see if there are more parameters here that need to be included
    return rigid_id


def load_iss_module_DEBUG(name: str) -> int:
    # TODO will need to manage the VHACD files too
    # Make sure that there is a "vhacd_*" file in the directory
    cwd = os.getcwd()
    directory = f"{cwd}/astrobee_media/astrobee_iss/meshes/{name}"
    if not os.path.exists(directory):
        raise NotADirectoryError(
            f"{directory} is not valid.\nCheck on the input, {name}, or current working directory, {cwd}"
        )
    part_paths = []
    vhacd_path = ""
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.startswith("vhacd"):
            vhacd_path = path
        elif filename.startswith("part"):
            part_paths.append(path)
            load_obj(path)
        else:
            continue
    return

    if not vhacd_path:
        raise FileNotFoundError("Could not find the VHACD OBJ file for collision info")
    if not part_paths:
        raise FileNotFoundError(
            "Could not find the OBJ part files for the ISS module visuals"
        )
    # createVisualShapeArray() is not well documented, but the following kwargs appear to be supported:
    # shapeTypes, halfExtents, radii, lengths, fileNames, meshScales, rgbaColors, visualFramePositions,
    # visualFrameOrientations, physicsClientId
    # See "urdfEditor.py" or "createVisualShapeArray.py" in Bullet3 for more info
    shape_types = [pybullet.GEOM_MESH for _ in range(len(part_paths))]
    visual_id = pybullet.createVisualShapeArray(
        shapeTypes=shape_types, fileNames=part_paths
    )
    if visual_id < 0:
        raise Exception("Pybullet could not load the visual shape array")

    collision_id = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_MESH, fileName=vhacd_path
    )
    rigid_id = pybullet.createMultiBody(
        baseMass=0,  # Fixed position
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    # TODO need to see if there are more parameters here that need to be included
    return rigid_id


# HACKY, TODO REMOVE THIS
# can't just load the collision like this because of needing to use the vhacd results
def load_obj(filename):
    visual_id = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=filename,
    )
    collision_id = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_MESH, fileName=filename
    )
    rigid_id = pybullet.createMultiBody(
        baseMass=0,  # mass==0 => fixed at position where it is loaded
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    return rigid_id


# TODO remove this once we're done with testing these functions
if __name__ == "__main__":
    initialize_pybullet(use_gui=True)
    # load_obj(
    #     "/home/dan/astrobee_pybullet/astrobee_media/astrobee_iss/meshes/obj_new_2/part6.obj"
    # )
    # load_iss_module_DEBUG("us_lab")
    load_iss_module("us_lab")

    run_sim()
