"""Script to identify (and visualize) the mesh vertices closest to the corners of the bags

Results will be something like:
[228, 331, 138, 372, 279, 223, 166, 201]  # Front
[299, 221, 111, 151, 332, 312, 186, 89]  # Side
[296, 243, 281, 99, 237, 151, 171, 262]  # Top

See the bag config file for the most up-to-date values
"""

import pybullet
import numpy as np
import numpy.typing as npt
import time
from pyastrobee.utils.bullet_utils import load_deformable_object, create_anchor
from pyastrobee.utils.mesh_utils import get_closest_mesh_vertex, get_mesh_data
from pyastrobee.utils.rotations import quat_to_rmat
from pyastrobee.utils.transformations import make_transform_mat, transform_point
from pyastrobee.config.bag_properties import BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT

NUM_CORNERS = 8

# Local-frame corners w.r.t the box's origin frame
LOCAL_CORNERS = [
    (BOX_LENGTH / 2, BOX_WIDTH / 2, BOX_HEIGHT / 2),  # Right back top
    (BOX_LENGTH / 2, BOX_WIDTH / 2, -BOX_HEIGHT / 2),  # Right back bottom
    (BOX_LENGTH / 2, -BOX_WIDTH / 2, BOX_HEIGHT / 2),  # Right front top
    (BOX_LENGTH / 2, -BOX_WIDTH / 2, -BOX_HEIGHT / 2),  # Right front bottom
    (-BOX_LENGTH / 2, BOX_WIDTH / 2, BOX_HEIGHT / 2),  # Left back top
    (-BOX_LENGTH / 2, BOX_WIDTH / 2, -BOX_HEIGHT / 2),  # Left back bottom
    (-BOX_LENGTH / 2, -BOX_WIDTH / 2, BOX_HEIGHT / 2),  # Left front top
    (-BOX_LENGTH / 2, -BOX_WIDTH / 2, -BOX_HEIGHT / 2),  # Left front bottom
]


def get_box_corners(pos: npt.ArrayLike, orn: npt.ArrayLike) -> np.ndarray:
    """Calculate the locations of the bag's bounding box corners, given its position and orientation

    Args:
        pos (npt.ArrayLike): XYZ Position of the bag
        orn (npt.ArrayLike): XYZW quaternion orientation of the bag

    Returns:
        list[np.ndarray]: Corners of the bag in world frame, shape (8, 3)
    """
    T = make_transform_mat(quat_to_rmat(orn), pos)
    return np.row_stack([transform_point(T, corner) for corner in LOCAL_CORNERS])


if __name__ == "__main__":
    mesh_dir = "pyastrobee/assets/meshes/bags/"
    bags = [
        "front_handle",
        "right_handle",
        "top_handle",
        "front_back_handle",
        "right_left_handle",
        "top_bottom_handle",
    ]
    all_files = [mesh_dir + file + ".vtk" for file in bags]
    pybullet.connect(pybullet.GUI)
    pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    # Run the identification process on each of the bags
    for bag_file in all_files:
        pos = [0, 0, 0]
        orn = [0, 0, 0, 1]
        bag_id = load_deformable_object(
            bag_file, pos=pos, orn=orn, bending_stiffness=10
        )
        n_verts, mesh = get_mesh_data(bag_id)
        box_corners = get_box_corners(pos, orn)
        verts = []
        anchors = []
        anchor_geoms = []
        # Find the closest mesh vertex to each of the corners
        for i in range(NUM_CORNERS):
            world_frame_corner = box_corners[i, :]
            v_pos, v_id = get_closest_mesh_vertex(world_frame_corner, mesh)
            verts.append(v_id)
            # Anchor the corners of the box to the world to help visualize
            anchor_id, anchor_geom = create_anchor(
                bag_id,
                v_id,
                -1,
                -1,
                parent_frame_pos=LOCAL_CORNERS[i],
                add_geom=True,
                geom_pos=world_frame_corner,
            )
            anchors.append(anchor_id)
            anchor_geoms.append(anchor_geom)
        print(f"Corner verts for {bag_file}:\n{verts}")
        # Allow for some interactive dragging to verify the corners look correct
        print("Sim is interactive for the next 10 seconds")
        init_time = time.time()
        while time.time() - init_time < 10:
            pybullet.stepSimulation()
            time.sleep(1 / 240)
        # Clear the sim before continuing to the next bag
        input("Press Enter to continue to the next bag")
        pybullet.removeBody(bag_id)
        for anchor in anchors:
            pybullet.removeConstraint(anchor)
        for geom in anchor_geoms:
            pybullet.removeBody(geom)
    input("Done. Press Enter to exit")
