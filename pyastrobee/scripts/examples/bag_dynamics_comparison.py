"""Script to visualize the dynamics of our different bag models under a initial disturbance"""

from pathlib import Path
from datetime import datetime

import numpy as np
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.core.composite_bag import CompositeCargoBag
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet


RECORD_VIDEO = False
VIDEO_LOCATION = (
    f"artifacts/{Path(__file__).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
)
# Debug visualizer camera parameters: Dist, yaw, pitch, target
CAMERA_VIEW = (1.80, 0.40, -14.60, (0, 0.2, -0.5))


# TODO clean this up and move to multi robot control
def step_controllers(
    controllers: list[ForceTorqueController],
    des_states: list[np.ndarray],
    client,
):
    assert len(des_states) == len(controllers)
    # HACKY: this assumes each state in des_states contains (in order):
    # position, velocity, acceleration, quaternion, omega, alpha
    for controller, des_state in zip(controllers, des_states):
        pos, orn, vel, omega = controller.get_current_state()
        controller.step(pos, vel, orn, omega, *des_state, step_sim=False)
    client.stepSimulation()


def main():
    np.random.seed(0)
    num_robots = 4
    positions = [(-1.5, 0, 0), (-0.5, 0, 0), (0.5, 0, 0), (1.5, 0, 0)]
    orn = (0, 0, 0, 1)
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    client.resetDebugVisualizerCamera(*CAMERA_VIEW)
    bag_vel = 0.5 * np.random.rand(3)
    bag_omega = np.random.rand(3)
    print(
        "Initial velocity applied to the bags:\n"
        + f"Velocity: {bag_vel}\n"
        + f"Angular velocity: {bag_omega}"
    )
    kp, kv, kq, kw = 20, 10, 5, 5
    robots = []
    controllers = []
    bags = []
    bag_types = [
        DeformableCargoBag,
        ConstraintCargoBag,
        CompositeCargoBag,
        RigidCargoBag,
    ]
    for i in range(num_robots):
        robots.append(Astrobee((*positions[i], *orn)))
        if bag_types[i] == DeformableCargoBag:
            bags.append(bag_types[i]("top_handle_symmetric", 10))
        else:
            bags.append(bag_types[i]("top_handle", 10))
        bags[-1].attach_to(robots[-1], "bag")
        controllers.append(
            ForceTorqueController(
                robots[-1].id,
                robots[-1].mass + bags[-1].mass,
                robots[-1].inertia,
                kp,
                kv,
                kq,
                kw,
                dt,
            )
        )
    client.stepSimulation()
    for i in range(num_robots):
        bag_pos, bag_orn = client.getBasePositionAndOrientation(bags[i].id)
        bags[i].reset_dynamics(bag_pos, bag_orn, bag_vel, bag_omega)
    des_states = [
        (positions[i], np.zeros(3), np.zeros(3), orn, np.zeros(3), np.zeros(3))
        for i in range(num_robots)
    ]

    rgbs = [[1, 0, 0]] * 5 * 2  # 5 constraints for tetrahedron, 2 bags
    constraint_pos = np.vstack(
        [bags[1].get_world_constraint_pos(0), bags[2].get_world_constraint_pos(0)]
    )
    points_uid = client.addUserDebugPoints(constraint_pos, rgbs, 10, 0)

    if RECORD_VIDEO:
        input("Ready to record video, press Enter to continue")
        log_id = client.startStateLogging(
            client.STATE_LOGGING_VIDEO_MP4, VIDEO_LOCATION
        )

    try:
        while True:
            step_controllers(controllers, des_states, client)
            # Visualize constraints for the handles
            points_uid = client.addUserDebugPoints(
                np.vstack(
                    [
                        bags[1].get_world_constraint_pos(0),
                        bags[2].get_world_constraint_pos(0),
                    ]
                ),
                rgbs,
                10,
                0,
                replaceItemUniqueId=points_uid,
            )
    finally:
        client.stopStateLogging(log_id)
        client.disconnect()


if __name__ == "__main__":
    main()
