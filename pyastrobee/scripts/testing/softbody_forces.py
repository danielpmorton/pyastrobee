"""Test script to evaluate the forces in an object-constrained-to-deformable system"""

import time
import pybullet
import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.core.deformable_cargo_bag import DeformableCargoBag
from pyastrobee.utils.bullet_utils import create_box, initialize_pybullet
from pyastrobee.utils.bullet_utils import create_anchor
from pyastrobee.utils.mesh_utils import get_closest_mesh_vertex
from pyastrobee.utils.transformations import make_transform_mat, transform_point
from pyastrobee.config.bag_properties import TOP_HANDLE_TRANSFORM
from pyastrobee.utils.rotations import (
    axis_angle_between_two_vectors,
    axis_angle_to_quat,
    quat_to_rmat,
)


def main():
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    # We'll use our standard cargo bag for the deformable object
    bag = DeformableCargoBag("top_handle")
    # And for the rigid object, we'll just use a simple cube
    # We'll rotate the cube so that we can attach the bag handle right at the tip of the cube
    cube_rot = axis_angle_to_quat(
        *axis_angle_between_two_vectors(np.array([1, 1, 1]), np.array([0, 0, 1]))
    )
    cube_size = 0.25
    cube_center = TOP_HANDLE_TRANSFORM[:3, 3] + np.array(
        [0, 0, np.linalg.norm(cube_size / 2 * np.ones(3))]
    )
    cube_mass = 1
    cube = create_box(cube_center, cube_rot, cube_mass, cube_size * np.ones(3), True)
    cube_tmat = make_transform_mat(quat_to_rmat(cube_rot), cube_center)
    cube_attach_pt_local = -0.5 * np.ones(3) * cube_size
    cube_attach_pt_world = transform_point(cube_tmat, cube_attach_pt_local)
    vert, vert_id = get_closest_mesh_vertex(cube_attach_pt_world, bag.mesh_vertices)
    anchor_id, geom_id = create_anchor(
        bag.id,
        vert_id,
        cube,
        -1,
        cube_attach_pt_local,
        add_geom=True,
        geom_pos=cube_attach_pt_world,
    )

    # Apply a constant force every simulation loop and record the position information on our objects
    bag_log = []
    cube_log = []
    com_log = []
    force = np.array([0, 0, 1])
    try:
        print(
            "Applying a constant force to the cube.\n"
            + "Press Ctrl+C to stop the simulation and show the plots"
        )
        while True:
            bag_pos, bag_orn = pybullet.getBasePositionAndOrientation(bag.id)
            cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(cube)
            system_com = (
                np.array(cube_pos) * cube_mass + np.array(bag_pos) * bag.mass
            ) / (cube_mass + bag.mass)
            bag_log.append(bag_pos)
            cube_log.append(cube_pos)
            com_log.append(system_com)
            pybullet.applyExternalForce(cube, -1, force, cube_pos, pybullet.WORLD_FRAME)
            pybullet.stepSimulation()
            time.sleep(1 / 120)
    except KeyboardInterrupt:
        print("Terminating simulation")
        pybullet.disconnect()

    # Take derivatives of the recorded position data to determine the internal/system forces
    bag_log = np.array(bag_log)
    cube_log = np.array(cube_log)
    com_log = np.array(com_log)

    bag_vels = np.gradient(bag_log, dt, axis=0)
    cube_vels = np.gradient(cube_log, dt, axis=0)
    com_vels = np.gradient(com_log, dt, axis=0)

    bag_accels = np.gradient(bag_vels, dt, axis=0)
    cube_accels = np.gradient(cube_vels, dt, axis=0)
    com_accels = np.gradient(com_vels, dt, axis=0)

    bag_forces = bag.mass * bag_accels
    cube_forces = cube_mass * cube_accels
    com_forces = (bag.mass + cube_mass) * com_accels

    fig1, axes1 = _plot(bag_log, cube_log, com_log, "Positions")
    fig2, axes2 = _plot(bag_vels, cube_vels, com_vels, "Velocities")
    fig3, axes3 = _plot(bag_accels, cube_accels, com_accels, "Accelerations")
    fig4, axes4 = _plot(bag_forces, cube_forces, com_forces, "Forces")
    # Add the applied force to the last plot
    for i, ax in enumerate(axes4):
        ax.plot(force[i] * np.ones(bag_forces.shape[0]), "k-", label="Applied")
    plt.legend()
    plt.show(block=False)
    input("Press Enter to close the plots")
    plt.close("all")


def _plot(
    bag_data: np.ndarray, cube_data: np.ndarray, com_data: np.ndarray, title: str
):
    """Helper function to plot the dynamics history of our objects"""
    fig, axes = plt.subplots(1, 3)
    y_min, y_max = np.inf, -np.inf
    for i, ax in enumerate(axes):
        ax.plot(bag_data[:, i], "r-", label="Bag")
        ax.plot(cube_data[:, i], "g-", label="Cube")
        ax.plot(com_data[:, i], "b-", label="COM")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(["x", "y", "z"][i])
        bottom, top = ax.get_ylim()
        y_min = min(y_min, bottom)
        y_max = max(y_max, top)
    # Rescale y axes to be the same
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    plt.suptitle(title)
    plt.legend()
    return fig, axes


if __name__ == "__main__":
    main()
