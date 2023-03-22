"""Demo: Load an Astrobee holding a cargo bag in the ISS and move around

TODO
- Calibrate the GRIPPER_TO_ARM_DISTAL transformation in pyastrobee/config/astrobee_transforms.py
  (This transform will dictate our offset between the last frame on the arm, and where we're holding the bag handle)
- Find the orientation(s) to load the bag and astrobee together, initially connected
- Attach an anchor to the bag handle
"""
import os

import numpy as np
import pybullet

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.control.controller import PoseController
from pyastrobee.control.plan_control_traj import plan_control_traj
from pyastrobee.utils.iss_utils import load_iss
from pyastrobee.utils.mesh_utils import get_mesh_data
from pyastrobee.utils.bullet_utils import (
    create_anchor_geom,
    get_closest,
    initialize_pybullet,
    load_deformable_object,
    run_sim,
)


def load_bag(robot_id, side=0):
    # Load deformable bag and attach the middle of each side of the handle to
    # the middle of each of the astrobee fingers.
    pfx = "pyastrobee/assets/meshes/bags/"
    fnames = ["front_handle_bag.vtk", "side_handle_bag.vtk"]  # TODO add top_handle_bag
    poss = np.array([[-0.05, 0.00, -0.53], [-0.05, 0.00, -0.65]])  # z=-0.53  -0.48
    orns = np.array([[-np.pi / 2, 0, 0], [0, -np.pi / 2, 0]])
    print("poss[side]", poss[side], "orns[side]", orns[side])
    bag_id = load_deformable_object(
        os.path.join(pfx, fnames[side]),
        pos=poss[side],
        orn=orns[side],
        bending_stiffness=50,
        elastic_stiffness=50,
        mass=1.0,
    )
    bag_texture_id = pybullet.loadTexture(
        "pyastrobee/assets/imgs/textile_pixabay_red.jpg"
    )
    kwargs = {}
    if hasattr(pybullet, "VISUAL_SHAPE_DOUBLE_SIDED"):
        kwargs["flags"] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    pybullet.changeVisualShape(
        bag_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=bag_texture_id, **kwargs
    )
    n_vert, bag_mesh = get_mesh_data(pybullet, bag_id)
    finger1_link_id = 4
    finger2_link_id = 6
    finger1_pos = pybullet.getLinkState(robot_id, finger1_link_id)[0]
    finger2_pos = pybullet.getLinkState(robot_id, finger2_link_id)[0]
    v1_pos, v1_ids = get_closest(finger1_pos, bag_mesh)
    v2_pos, v2_ids = get_closest(finger2_pos, bag_mesh)
    v1_id, v2_id = v1_ids[0], v2_ids[0]
    pybullet.createSoftBodyAnchor(bag_id, v1_id, robot_id, finger1_link_id)
    pybullet.createSoftBodyAnchor(bag_id, v2_id, robot_id, finger2_link_id)
    anchor_kwargs = {"mass": 0.01, "radius": 0.01, "rgba": (0, 1, 0, 0.5)}
    anchor1_id = create_anchor_geom(pybullet, v1_pos, **anchor_kwargs)
    anchor2_id = create_anchor_geom(pybullet, v2_pos, **anchor_kwargs)
    pybullet.createSoftBodyAnchor(bag_id, v1_id, anchor1_id, -1)
    pybullet.createSoftBodyAnchor(bag_id, v2_id, anchor2_id, -1)


def glide_to_pose(
    target_pos,
    target_quat,
    robot_id,
    constraint_id,
    n_traj_steps=1000,
    max_force=1000,
    sim_freq=350,
):
    curr_pos, curr_quat = pybullet.getBasePositionAndOrientation(robot_id)
    target_pos_traj, target_quat_traj = plan_control_traj(
        target_pos,
        target_quat,
        num_steps=n_traj_steps,
        freq=sim_freq,
        curr_pos=curr_pos,
        curr_quat=curr_quat,
        curr_vel=[0, 0, 0],
    )
    for t in range(n_traj_steps):
        # mult = max(0.1, (t+1.0)/n_traj_steps)
        pybullet.changeConstraint(
            constraint_id,
            target_pos_traj[t],
            target_quat_traj[t],
            maxForce=1.0 * max_force,
        )
        pybullet.stepSimulation()


def demo_with_iss():
    """A simple demo of loading the astrobee in the ISS and moving it around in various ways

    TODO: the WPs might need to be refined, there seems to be a lot of weird rotating going on
    Or it could be a quaternion issue? Quaternion ambiguity?

    This could just in general be cleaned up and refined, but it works ok for now
    """
    # Hardcoded waypoints and positions found from keyboard-controlling the Astrobee
    waypts = [
        [0, 0, 0, 0, 0, 0, 1],
        [
            0.44631294,
            -1.33893871,
            0.44631287,
            0.08824572,
            0.06790329,
            -0.78759863,
            0.60604474,
        ],
        [
            0.05603137,
            -2.81145659,
            0.10060672,
            -0.06176491,
            -0.0185934,
            -0.69867597,
            0.71252457,
        ],
        [
            -0.31709299,
            0.31352898,
            0.53193288,
            -0.03191529,
            0.0062923,
            -0.83105266,
            0.55524166,
        ],
    ]
    initialize_pybullet()
    # pybullet.connect(pybullet.GUI)  # a simple version without deformables
    # Bring the camera close to the action (another just random hardcoded position I found)
    pybullet.resetDebugVisualizerCamera(1.6, 206, -26.2, [0, 0, 0])
    load_iss()
    robot = Astrobee()
    load_bag(robot.id)
    controller = PoseController(robot)
    for i in range(len(waypts)):
        glide_to_pose(
            target_pos=waypts[i][:3],
            target_quat=waypts[i][3:],
            robot_id=robot.id,
            constraint_id=controller.constraint_id,
        )
    print("Gliding done. Keep sim running...")
    run_sim()  # keep sim spinning


def demo_with_bag():
    initialize_pybullet()
    cam_args = {
        "cameraDistance": 1.6,  # use 0.7 to look at anchor attachment closely
        "cameraPitch": 200,
        "cameraYaw": 80,
        "cameraTargetPosition": np.array([0, 0, 0]),
    }
    pybullet.resetDebugVisualizerCamera(**cam_args)
    robot = Astrobee()
    load_bag(robot.id)
    controller = PoseController(robot)
    glide_to_pose(
        target_pos=[-0.5, -0.4, 0.8],
        target_quat=[0, 0, 0, 1],
        robot_id=robot.id,
        constraint_id=controller.constraint_id,
    )
    run_sim()  # keep sim spinning


if __name__ == "__main__":
    demo_with_iss()
