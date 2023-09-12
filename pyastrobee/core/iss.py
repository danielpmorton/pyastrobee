"""Functions related to loading the ISS modules into pybullet

Notes:
- If any of the ISS meshes are updated (such as a reorientation or repositioning in Blender), the default orientation
    should be changed in the loading functions
- If the mesh directory gets changed, the hardcoded relative paths need to be updated
"""

import os
from enum import Enum
from typing import Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np

from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.errors import PybulletError
from pyastrobee.utils.python_utils import print_green
from pyastrobee.config.iss_safe_boxes import FULL_SAFE_SET, ROBOT_SAFE_SET
from pyastrobee.config.iss_paths import GRAPH
from pyastrobee.utils.boxes import visualize_3D_box


class ISS:
    """The ISS, as represented in the original NASA/astrobee repo

    Args:
        debug (bool, optional): Whether or not to visualize just the collision bodies. Defaults to False
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)
    """

    class Modules(Enum):
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

    def __init__(self, debug: bool = False, client: Optional[BulletClient] = None):
        self.debug = debug
        self.client: pybullet = pybullet if client is None else client
        # The meshes have a weird orientation so we need to use this orientation to rotate them to lay flat
        self.mesh_orn = (np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2)
        self.full_safe_set = FULL_SAFE_SET
        self.robot_safe_set = ROBOT_SAFE_SET
        self.graph = GRAPH  # Precomputed
        self._debug_box_ids = []
        self.ids = []
        for module in ISS.Modules:
            self.ids.append(self._load_module(module))
        print_green("ISS is ready")

    def show_safe_set(self, for_robot: bool = False) -> None:
        """Visualizes the collision-free regions inside the ISS

        Args:
            for_robot (bool, optional): Whether to shrink the safe set to account for the collision radius of the
                Astrobee's body. Defaults to False (Show the full safe set)
        """
        boxes = self.robot_safe_set if for_robot else self.full_safe_set
        for box in boxes.values():
            self._debug_box_ids.append(visualize_3D_box(box))

    def hide_safe_set(self) -> None:
        """Removes the visualization of the collision-free regions"""

        for box_id in self._debug_box_ids:
            self.client.removeBody(box_id)
        self._debug_box_ids = []

    def _load_module(self, module: Modules) -> int:
        """Loads a single ISS module. For example, US_LAB

        Args:
            module (Modules): The module to load. For example, Modules.CUPOLA / .EU_LAB / .JPM / ...

        Returns:
            int: The Pybullet ID for the multibody associated with the VHACD collision object
        """
        # Locate the paths to all of the meshes for the module
        vhacd_path, part_paths = self._find_mesh_files(module)

        # If we're debugging the collision info, just load the VHACD results as both the collision and visual
        if self.debug:
            visual_id = self.client.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=vhacd_path,
                visualFrameOrientation=self.mesh_orn,
            )
            collision_id = self.client.createCollisionShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=vhacd_path,
                collisionFrameOrientation=self.mesh_orn,
            )
            rigid_id = self.client.createMultiBody(
                baseMass=0,  # Fixed position
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                baseInertialFrameOrientation=self.mesh_orn,
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
            visual_id = self.client.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=path,
                visualFrameOrientation=self.mesh_orn,
            )
            if visual_id < 0:
                raise PybulletError(
                    f"Could not load the visual shape for {path}", visual_id
                )
            if i == 0:
                # Load the VHACD results as the collision info
                collision_id = self.client.createCollisionShape(
                    shapeType=pybullet.GEOM_MESH,
                    fileName=vhacd_path,
                    collisionFrameOrientation=self.mesh_orn,
                )
                if collision_id < 0:
                    raise PybulletError(
                        f"Could not load the collision shape for {path}", collision_id
                    )
            else:
                collision_id = -1  # -1 means no collision bodies will be generated
            rigid_id = self.client.createMultiBody(
                baseMass=0,  # Fixed position
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                baseInertialFrameOrientation=self.mesh_orn,
            )
            ids.append(rigid_id)
        return ids[0]  # Just the ID for the VHACD object

    def _find_mesh_files(self, module: Modules) -> tuple[str, list[str]]:
        """Helper function to locate the paths to the ISS module meshes

        Args:
            module (ISSModule): The module to load. For example, ISSModule.CUPOLA / .EU_LAB / .JPM / ...

        Raises:
            ValueError: If an invalid ISS module name is provided
            NotADirectoryError: If the module's mesh directory cannot be found
            FileNotFoundError: If either the vhacd obj file or the obj2sdf objs cannot be found in the mesh directory

        Returns:
            tuple[str, list[str]]:
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
            raise FileNotFoundError(
                "Could not find the VHACD OBJ file for collision info"
            )
        if not part_paths:
            raise FileNotFoundError(
                "Could not find the OBJ files for the ISS visual info"
            )
        return vhacd_path, part_paths


def _main():
    client = initialize_pybullet()
    iss = ISS(debug=False, client=client)
    iss.show_safe_set()
    input("Press Enter to hide the safe set visualization")
    iss.hide_safe_set()
    input("Press Enter to exit")
    client.disconnect()


if __name__ == "__main__":
    _main()
