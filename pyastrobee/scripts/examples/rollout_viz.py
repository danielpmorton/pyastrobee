"""Script to create an illustrative figure of the MPC rollouts"""

import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.config.iss_safe_boxes import FULL_SAFE_SET
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.utils.boxes import visualize_3D_box
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.trajectories.planner import local_planner
from pyastrobee.utils.rotations import Rz, rmat_to_quat
from pyastrobee.trajectories.sampling import sample_state
from pyastrobee.utils.debug_visualizer import visualize_path
from pyastrobee.control.force_torque_control import ForceTorqueController

np.random.seed(0)
client = initialize_pybullet()
dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
client.resetDebugVisualizerCamera(8, 136.40, -16.20, (4.84, -4.29, -1.93))
for box in FULL_SAFE_SET.values():
    visualize_3D_box(box, rgba=(1, 0, 0, 0.25))
start_pos = (10, 0, 0)
start_orn = rmat_to_quat(Rz(np.pi))
client.stepSimulation()
goal_pos = (5, 0, 0)
goal_vel = (0, 0, 0)
goal_accel = (0, 0, 0)
goal_orn = start_orn
goal_omega = (0, 0, 0)
goal_alpha = (0, 0, 0)
n_trajs = 10
trajs = []
for i in range(n_trajs):
    pos, orn, vel, omega, accel, alpha = sample_state(
        goal_pos,
        goal_orn,
        goal_vel,
        goal_omega,
        goal_accel,
        goal_alpha,
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    )
    trajs.append(
        local_planner(
            start_pos,
            start_orn,
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            pos,
            orn,
            vel,
            omega,
            accel,
            alpha,
            10,
            1 / 350,
        )
    )
    visualize_path(trajs[-1].positions, 20, (1, 1, 1), 2)
n_rollout_bots = 3
traj_idxs = np.random.randint(0, n_trajs, n_rollout_bots)
traj_terminal_idxs = [
    int(trajs[0].num_timesteps * (i + 1) / n_rollout_bots)
    for i in range(n_rollout_bots)
]
robot_states = []
bag_states = []


for i in range(n_rollout_bots):
    traj = trajs[traj_idxs[i]].get_segment(0, traj_terminal_idxs[i])
    robot = Astrobee((*traj.positions[0], *traj.quaternions[0]))
    bag = ConstraintCargoBag("top_handle", 10)
    bag.attach_to(robot, "bag")
    kp, kv, kq, kw = 20, 10, 5, 5
    controller = ForceTorqueController(
        robot.id,
        robot.mass + bag.mass,
        robot.inertia,
        kp,
        kv,
        kq,
        kw,
        dt,
        client=client,
    )
    controller.follow_traj(traj, False)
    robot_states.append(robot.dynamics_state)
    bag_states.append(bag.dynamics_state)
    bag.detach()
    bag.unload()
    robot.unload()

robots = []
bags = []
for i in range(n_rollout_bots):
    robots.append(Astrobee())
    client.resetBasePositionAndOrientation(
        robots[-1].id, robot_states[i][0], robot_states[i][1]
    )
    bags.append(ConstraintCargoBag("top_handle", 10))
    bags[-1].attach_to(robots[-1], "bag")
    bags[-1].reset_dynamics(bag_states[i][0], bag_states[i][1], (0, 0, 0), (0, 0, 0))
main_robot = Astrobee()
main_robot.reset_to_base_pose((*start_pos, *start_orn))
main_bag = DeformableCargoBag("top_handle_symmetric", 10)
main_bag.attach_to(main_robot, "bag")
client.stepSimulation()
input("Press Enter to exit")
client.disconnect()
