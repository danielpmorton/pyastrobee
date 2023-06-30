"""Model predictive control

NOTES
- This implementation really just considers the dynamics/control of the astrobee base
  rather than focusing on the operational space point or the bag

TODOS
- Make a new stopping_criteria function that takes the bag dynamics into account
- Add different rollout methods with simplified models. Add logic/structure to incorporate the new methods
- Allow for physics clients to start in parallel without opening the GUI
  (see the gym environments Rika was mentioning)

QUESTIONS
- Is it better to define things in terms of the bag, the gripper, or the robot base?
- Should there be two stages to the control? Navigation vs stopping? We'll need to stabilize
  both the robot and the bag, stop both, and potentially disconnect from the bag. This might
  need multiple control modes
"""

import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.core.iss import load_iss
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.trajectories.trajectory import stopping_criteria
from pyastrobee.trajectories.rewards_and_penalties import deviation_penalty
from pyastrobee.trajectories.planner import plan_trajectory
from pyastrobee.trajectories.sampling import generate_trajs
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.debug_visualizer import remove_debug_objects


def init(
    robot_pose: npt.ArrayLike, use_gui: bool = True
) -> tuple[BulletClient, Astrobee, CargoBag]:
    """Initialize the simulation environment with our assets

    Args:
        robot_pose (npt.ArrayLike): Initial pose of the Astrobee
        use_gui (bool, optional): Whether or not to launch the simulation in a GUI window. Defaults to True.

    Returns:
        Tuple of:
            BulletClient: The Pybullet client
            Astrobee: The Astrobee object
            CargoBag: The Cargo Bag object
    """
    client = initialize_pybullet(use_gui)
    load_iss(client=client)
    robot = Astrobee(robot_pose, client=client)
    bag = CargoBag("top_handle", client=client)
    bag.attach_to(robot, object_to_move="bag")
    return client, robot, bag


def mpc_main(
    start_pose: npt.ArrayLike,
    goal_pose: npt.ArrayLike,
    duration: float,
    debug: bool = False,
):
    # Assign constants (TODO decide which of these should be inputs, if any)
    # Tracking controller gains
    kp = 20
    kv = 5
    kq = 1
    kw = 0.1
    # Penalty scales for deviations from the nominal trajectory
    # TODO refine these values!! totally different scales, and different impact on performance
    pos_penalty = 1
    orn_penalty = 1
    vel_penalty = 1
    ang_vel_penalty = 1
    # Sampling standard deviations for candidate replanning trajectories
    # TODO refine these values
    pos_stdev = 0.05
    orn_stdev = 0.05
    vel_stdev = 0.05
    ang_vel_stdev = 0.05
    accel_stdev = 0.05
    alpha_stdev = 0.05
    # Timestep (based on pybullet physics)
    dt = 1 / 350
    # Number of steps to execute in a rollout
    n_rollout_steps = 20
    # Number of trajectories to consider within an MPC iteration
    n_candidate_trajs = 5
    # Tolerance on dynamics errors for determining if we've stopped the Astrobee
    # TODO figure out if these should be larger
    pos_tol = 1e-2
    vel_tol = 1e-2
    orn_tol = 1e-2
    ang_vel_tol = 5e-3

    client, robot, bag = init(start_pose, use_gui=True)
    tracking_controller = ForceTorqueController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt, client=client
    )
    nominal_traj = plan_trajectory(
        start_pose[:3],
        start_pose[3:],
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        goal_pose[:3],
        goal_pose[3:],
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        duration,
        dt,
    )
    if debug:
        line_ids = nominal_traj.visualize(20, client)
        remove_debug_objects(line_ids, client)
    # Add some buffer time to the end of the trajectory for stopping
    max_stopping_time = 3  # seconds
    max_steps = nominal_traj.num_timesteps + round(max_stopping_time / dt)
    end_idx = nominal_traj.num_timesteps - 1
    cur_idx = 0
    prev_accel = 0  # TODO improve this handling
    prev_alpha = 0
    step_count = 0
    # Each iteration of the loop will step the simulation by the number of rollout steps
    # This is because we don't need to replan every single simulation step
    while True:
        if step_count >= max_steps:
            print("MAX STEPS EXCEEDED")
            break
        state_id = client.saveState()
        pos, orn, vel, ang_vel = robot.dynamics_state
        if cur_idx == end_idx and stopping_criteria(
            pos,
            orn,
            vel,
            ang_vel,
            nominal_traj.positions[-1],
            nominal_traj.quaternions[-1],
            pos_tol,
            orn_tol,
            vel_tol,
            ang_vel_tol,
        ):
            break

        costs = []
        lookahead_idx = min(cur_idx + n_rollout_steps, end_idx)
        stop_at_end = lookahead_idx == end_idx
        # TODO need to figure out the number of steps in the generated trajectories
        # depending on if the robot should be stopping or not
        # TODO wondering if I should just have an entirely different mode for "stopping"
        # TODO MAKE SURE THAT THE STEP/CONTROL ALLOCATION WORKS WHEN STOPPING
        trajs = generate_trajs(
            pos,
            orn,
            vel,
            ang_vel,
            prev_accel,
            prev_alpha,
            nominal_traj.positions[lookahead_idx],
            nominal_traj.quaternions[lookahead_idx],
            nominal_traj.linear_velocities[lookahead_idx],
            nominal_traj.angular_velocities[lookahead_idx],
            nominal_traj.linear_accels[lookahead_idx],
            nominal_traj.angular_accels[lookahead_idx],
            pos_stdev,
            orn_stdev,
            vel_stdev,
            ang_vel_stdev,
            accel_stdev,
            alpha_stdev,
            n_candidate_trajs,
            min(n_rollout_steps, lookahead_idx - cur_idx + 1),
            dt,
        )
        for traj in trajs:
            if debug:
                line_ids = traj.visualize(10, client)
                remove_debug_objects(line_ids, client)
            # This is effectively a perfect rollout (TODO make this fact clearer)
            tracking_controller.follow_traj(traj, stop_at_end, n_rollout_steps)
            # TODO should we visualize the deviation in the trajectory?
            pos, orn, vel, ang_vel = robot.dynamics_state
            costs.append(
                deviation_penalty(
                    pos,
                    orn,
                    vel,
                    ang_vel,
                    nominal_traj.positions[lookahead_idx],
                    nominal_traj.quaternions[lookahead_idx],
                    nominal_traj.linear_velocities[lookahead_idx],
                    nominal_traj.angular_velocities[lookahead_idx],
                    pos_penalty,
                    orn_penalty,
                    vel_penalty,
                    ang_vel_penalty,
                )
            )
            client.restoreState(stateId=state_id)
        best_traj = trajs[np.argmin(costs)]
        # Execute the best trajectory
        tracking_controller.follow_traj(best_traj, stop_at_end)
        # Update loop variables
        cur_idx = lookahead_idx
        step_count += n_rollout_steps
        prev_accel = best_traj.linear_accels[-1]
        prev_alpha = best_traj.angular_accels[-1]
    # TODO decide what to do once the main loop finishes?


def _test_mpc(debug=False):
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    duration = 5
    mpc_main(start_pose, end_pose, duration, debug)


if __name__ == "__main__":
    _test_mpc(debug=False)
    # _test_init()
