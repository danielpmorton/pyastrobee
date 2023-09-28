"""Script to help tune controller gains and mass/inertia values via interacting with the GUI"""


import numpy as np
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.debug_visualizer import visualize_frame


def _main(load_bag: bool = True):
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    des_pos = (0, 0, 0)
    des_orn = (0, 0, 0, 1)
    des_tmat = np.eye(4)
    visualize_frame(des_tmat)
    robot = Astrobee((*des_pos, *des_orn))
    if load_bag:
        bag = ConstraintCargoBag("top_handle", 10)
        bag.attach_to(robot, "bag")
    kp, kv, kq, kw = 20, 10, 5, 5
    mass = robot.mass + bag.mass if load_bag else robot.mass
    controller = ForceTorqueController(
        robot.id,
        mass,
        robot.inertia,  # + bag inertia! TODO try this... calculate every so often
        kp,
        kv,
        kq,
        kw,
        dt,
    )
    print("Ready. Interact with the GUI to observe controller behavior")
    while True:
        pos, orn, vel, ang_vel = controller.get_current_state()
        controller.step(
            pos,
            vel,
            orn,
            ang_vel,
            des_pos,
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 0),
            (0, 0, 0),
        )


if __name__ == "__main__":
    _main(load_bag=True)
