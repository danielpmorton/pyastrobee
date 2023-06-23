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
import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.core.iss import load_iss
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.quaternions import quaternion_angular_error
from pyastrobee.utils.math_utils import spherical_vonmises_sampling
from pyastrobee.trajectories.trajectory import Trajectory, stopping_criteria
from pyastrobee.trajectories.planner import bezier_and_quat_poly_traj
from pyastrobee.control.force_controller_new import ForcePIDController
from pyastrobee.utils.debug_visualizer import remove_debug_objects


def init(
    robot_pose: npt.ArrayLike, use_gui: bool = True
) -> tuple[int, Astrobee, CargoBag]:
    """Initialize the simulation environment with our assets

    Args:
        robot_pose (npt.ArrayLike): Initial pose of the Astrobee
        use_gui (bool, optional): Whether or not to launch the simulation in a GUI window. Defaults to True.

    Returns:
        Tuple of:
            int: The Pybullet client ID
            Astrobee: The Astrobee object
            CargoBag: The Cargo Bag object
    """
    client = initialize_pybullet(use_gui)
    load_iss()
    robot = Astrobee(robot_pose)
    bag = CargoBag("top_handle")
    bag.attach_to(robot, object_to_move="bag")
    return client, robot, bag


def deviation_penalty(
    cur_pos: npt.ArrayLike,
    cur_orn: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    cur_ang_vel: npt.ArrayLike,
    des_pos: npt.ArrayLike,
    des_orn: npt.ArrayLike,
    des_vel: npt.ArrayLike,
    des_ang_vel: npt.ArrayLike,
    pos_penalty: float,
    orn_penalty: float,
    vel_penalty: float,
    ang_vel_penalty: float,
) -> float:
    """Evaluate a loss/penalty for deviations between the current and desired dynamics state

    Args:
        cur_pos (npt.ArrayLike): Current position, shape (3,)
        cur_orn (npt.ArrayLike): Current XYZW quaternion orientation, shape (4,)
        cur_vel (npt.ArrayLike): Current linear velocity, shape (3,)
        cur_ang_vel (npt.ArrayLike): Current angular velocity, shape (3,)
        des_pos (npt.ArrayLike): Desired position, shape (3,)
        des_orn (npt.ArrayLike): Desired XYZW quaternion orientation, shape (4,)
        des_vel (npt.ArrayLike): Desired linear velocity, shape (3,)
        des_ang_vel (npt.ArrayLike): Desired angular velocity, shape (3,)
        pos_penalty (float): Penalty scaling factor for position error
        orn_penalty (float): Penalty scaling factor for orientation/angular error
        vel_penalty (float): Penalty scaling factor for linear velocity error
        ang_vel_penalty (float): Penalty scaling factor for angular velocity error

    Returns:
        float: Loss/penalty value
    """
    pos_err = np.subtract(cur_pos, des_pos)
    orn_err = quaternion_angular_error(cur_orn, des_orn)
    vel_err = np.subtract(cur_vel, des_vel)
    ang_vel_err = np.subtract(cur_ang_vel, des_ang_vel)
    # NOTE Would an L1 norm be better here?
    return (
        pos_penalty * np.linalg.norm(pos_err)
        + orn_penalty * np.linalg.norm(orn_err)
        + vel_penalty * np.linalg.norm(vel_err)
        + ang_vel_penalty * np.linalg.norm(ang_vel_err)
    )


def generate_trajs(
    cur_pos: npt.ArrayLike,
    cur_orn: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    cur_ang_vel: npt.ArrayLike,
    cur_accel: npt.ArrayLike,  # Optional?
    cur_alpha: npt.ArrayLike,  # Optional?
    nominal_target_pos: npt.ArrayLike,
    nominal_target_orn: npt.ArrayLike,
    nominal_target_vel: npt.ArrayLike,
    nominal_target_ang_vel: npt.ArrayLike,
    nominal_target_accel: npt.ArrayLike,  # Optional?
    nominal_target_alpha: npt.ArrayLike,  # Optional?
    pos_sampling_stdev: float,
    orn_sampling_stdev: float,
    vel_sampling_stdev: float,
    ang_vel_sampling_stdev: float,
    accel_sampling_stdev: float,
    alpha_sampling_stdev: float,
    n_trajs: int,
    n_steps: int,
    dt: float,
) -> list[Trajectory]:
    """Generate a number of trajectories from the current state to a sampled state about a nominal target

    TODO
    - Decide if we should be passing in covariance matrices or arrays instead of scalars
    - Decide if the "orientation stdev" should be replaced by the von Mises kappa parameter
    - Update the trajectory generation when we improve that process
    - Add in the current acceleration parameters so we can enforce acceleration continuity

    Args:
        cur_pos (npt.ArrayLike): Current position, shape (3,)
        cur_orn (npt.ArrayLike): Current XYZW quaternion orientation, shape (4,)
        cur_vel (npt.ArrayLike): Current linear velocity, shape (3,)
        cur_ang_vel (npt.ArrayLike): Current angular velocity, shape (3,)
        nominal_target_pos (npt.ArrayLike): Nominal desired position to sample about, shape (3,)
        nominal_target_orn (npt.ArrayLike): Nominal desired XYZW quaternion to sample about, shape (4,)
        nominal_target_vel (npt.ArrayLike): Nominal desired linear velocity to sample about, shape (3,)
        nominal_target_ang_vel (npt.ArrayLike): Nominal desired angular velocity to sample about, shape (3,)
        pos_sampling_stdev (float): Standard deviation of the position sampling distribution
        orn_sampling_stdev (float): Standard deviation of the orientation sampling distribution
        vel_sampling_stdev (float): Standard deviation of the velocity sampling distribution
        ang_vel_sampling_stdev (float): Standard deviation of the angular velocity sampling distribution
        n_trajs (int): Number of trajectories to generate
        n_steps (int): Number of trajectory steps between the current state and the sampled goal state
        dt (float): Timestep

    Returns:
        list[Trajectory]: Sampled trajectories, length n_trajs
    """
    # Sample enpoints and then use these to generate trajectories
    # First trajectory will be nominal, other trajectories will be sampling-based
    n_samples = n_trajs - 1

    # Sample endpoints for the candidate trajectories about the nominal targets
    sampled_positions = np.random.multivariate_normal(
        nominal_target_pos, pos_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_quats = spherical_vonmises_sampling(
        nominal_target_orn, 1 / (orn_sampling_stdev**2), n_samples
    )
    sampled_vels = np.random.multivariate_normal(
        nominal_target_vel, vel_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_ang_vels = np.random.multivariate_normal(
        nominal_target_ang_vel, ang_vel_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_accels = np.random.multivariate_normal(
        nominal_target_accel, accel_sampling_stdev**2 * np.eye(3), n_samples
    )
    sampled_alphas = np.random.multivariate_normal(
        nominal_target_alpha, alpha_sampling_stdev**2 * np.eye(3), n_samples
    )
    trajs = [
        bezier_and_quat_poly_traj(
            cur_pos,
            cur_orn,
            cur_vel,
            cur_ang_vel,
            cur_accel,
            cur_alpha,
            nominal_target_pos,
            nominal_target_orn,
            nominal_target_vel,
            nominal_target_ang_vel,
            nominal_target_accel,
            nominal_target_alpha,
            n_steps * dt,
            dt,
        )
    ]
    for i in range(n_samples):
        trajs.append(
            bezier_and_quat_poly_traj(
                cur_pos,
                cur_orn,
                cur_vel,
                cur_ang_vel,
                cur_accel,
                cur_alpha,
                sampled_positions[i],
                sampled_quats[i],
                sampled_vels[i],
                sampled_ang_vels[i],
                sampled_accels[i],
                sampled_alphas[i],
                n_steps * dt,
                dt,
            )
        )
    return trajs


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
    tracking_controller = ForcePIDController(
        robot.id,
        robot.mass,
        robot.inertia,
        kp,
        kv,
        kq,
        kw,
        dt,
    )
    nominal_traj = bezier_and_quat_poly_traj(
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
        line_ids = nominal_traj.visualize(20)
        remove_debug_objects(line_ids)
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
        state_id = pybullet.saveState()
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
                line_ids = traj.visualize(10)
                remove_debug_objects(line_ids)
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
            pybullet.restoreState(stateId=state_id)
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
