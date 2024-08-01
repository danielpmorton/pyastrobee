"""MPC with vectorized environments running in parallel

We create three different types of environments:
1: The main simulation
2: One parallel environment which evaluates the "nominal" trajectory
3: Other parallel environments which evaluate deviations on the nominal value

When debugging, we visualize the nominal parallel environment as well, and show the trajectory rollout plan
"""
# TODO:
# - Merge this with the main MPC file?
# - Turn this into a controller class, or just leave as a script?
# - Update the observation and action spaces to match what we're using
#   (robot/bag state observations). We're using dummy action values right now but we could
#   also let the action input be a trajectory, rather than sampling in the env

from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import numpy.typing as npt
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv

from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.core.environments import AstrobeeMPCEnv, make_vec_env
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.planner import global_planner
from pyastrobee.trajectories.arm_planner import plan_arm_traj
from pyastrobee.utils.python_utils import print_red, print_green
from pyastrobee.utils.video_concatenation import concatenate_videos

# Recording parameters
RECORD_MAIN_ENV = False
RECORD_DEBUG_ENV = False
MAIN_VIDEO_DIRECTORY = (
    f"artifacts/{Path(__file__).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}/"
)
DEBUG_VIDEO_DIRECTORY = MAIN_VIDEO_DIRECTORY.rstrip("/") + "_debug/"
# Debug visualizer camera parameters: Dist, yaw, pitch, target
NODE_2_VIEW = (1.40, -69.60, -19.00, (0.55, 0.00, -0.39))
JPM_VIEW = (1.00, 64.40, -12.20, (6.44, -0.39, 0.07))
EXTERNAL_VIEW = (9.20, 49.60, -9.80, (-1.07, -1.53, -0.41))


def parallel_mpc_main(
    start_pose: npt.ArrayLike,
    goal_pose: npt.ArrayLike,
    n_vec_envs: int,
    bag_name: str = "top_handle",
    bag_mass: float = 10,
    use_deformable_primary_sim: bool = True,
    use_deformable_rollouts: bool = False,
    debug: bool = False,
    random_seed: Optional[int] = None,
):
    """Launches a series of environments in parallel and runs a model-predictive-controller to move Astrobee between
    two poses while carrying a cargo bag

    Args:
        start_pose (npt.ArrayLike): Starting pose of the Astrobee (position and XYZW quaternion), shape (7,)
        goal_pose (npt.ArrayLike): Ending pose of the Astrobee (position and XYZW quaternion), shape (7,)
        n_vec_envs (int): Number of vectorized environments to launch in parallel (>= 1)
        bag_name (str, optional): Type of cargo bag to load. Defaults to "top_handle".
        bag_mass (float): Mass of the cargo bag, in kg. Defaults to 10
        use_deformable_primary_sim (bool, optional): Whether to load the deformable bag in the main simulation env.
            Defaults to True (load the deformable version)
        use_deformable_rollouts (bool, optional): Whether to use the deformable bag for rollouts. Defaults to False
            (perform rollouts with the simplified rigid bag)
        debug (bool, optional): Whether to launch one of the vectorized environments with the GUI active, to visualize
            some of the rollouts being evaluated. Defaults to False.
        random_seed (Optional[int]): Seed for the random number generator, if desired. Defaults to None (unseeded)
    """
    if n_vec_envs < 1:
        raise ValueError("Must have at least one environment for evaluating rollouts")
    # Set up main environment
    main_env = AstrobeeMPCEnv(
        use_gui=True,
        is_primary=True,
        robot_pose=start_pose,
        bag_name=bag_name,
        bag_mass=bag_mass,
        bag_type=DeformableCargoBag
        if use_deformable_primary_sim
        else ConstraintCargoBag,
        load_full_iss=True,
    )
    # Set up vectorized environments
    env_kwargs = {
        "use_gui": False,
        "is_primary": False,
        "robot_pose": start_pose,
        "bag_name": bag_name,
        "bag_mass": bag_mass,
        "bag_type": DeformableCargoBag
        if use_deformable_rollouts
        else ConstraintCargoBag,
        # We need the full ISS loaded if using deformable rollouts so save/restore state sees the same envs
        "load_full_iss": use_deformable_rollouts,
    }
    debug_env_idx = 0
    # Enable GUI for one of the vec envs if debugging, and use this to test the nominal (non-sampled) trajs
    per_env_kwargs = {debug_env_idx: {"use_gui": debug, "nominal_rollouts": True}}
    vec_env = make_vec_env(
        AstrobeeMPCEnv,
        n_vec_envs,
        random_seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
        per_env_kwargs=per_env_kwargs,
    )
    main_env.reset(random_seed)
    vec_env.seed(random_seed)
    vec_env.reset()  # Random seed included in make_vec_env

    # Generate nominal trajectory
    dt = main_env.client.getPhysicsEngineParameters()["fixedTimeStep"]
    nominal_traj = global_planner(
        start_pose[:3],
        start_pose[3:],
        goal_pose[:3],
        goal_pose[3:],
        dt,
    )
    nominal_arm_traj = plan_arm_traj(nominal_traj)

    # Store the goal pose to determine stopping criteria
    # TODO should this instead be an input to the environment?
    # TODO it seems like this doesn't actually work properly
    main_env.goal_pose = goal_pose
    vec_env.set_attr("goal_pose", goal_pose)

    video_index = 0
    camera_moved = False
    if RECORD_MAIN_ENV or RECORD_DEBUG_ENV:
        main_env.client.resetDebugVisualizerCamera(*NODE_2_VIEW)
        vec_env.env_method(
            "send_client_command",
            "resetDebugVisualizerCamera",
            *NODE_2_VIEW,
            indices=[debug_env_idx],
        )
        # main_env.client.resetDebugVisualizerCamera(*EXTERNAL_VIEW)
        print("Ready to record video. Remember to maximize the GUI")
        main_path = Path(MAIN_VIDEO_DIRECTORY)
        debug_path = Path(DEBUG_VIDEO_DIRECTORY)
        if (main_path.exists() and RECORD_MAIN_ENV) or (
            debug_path.exists and RECORD_DEBUG_ENV
        ):
            print_red("WARNING: Recording video will overwrite existing files")
            # TODO decide if we should empty the directory/directories
        input("Press Enter to begin")
        if RECORD_MAIN_ENV:
            main_path.mkdir(parents=True, exist_ok=True)
        if RECORD_DEBUG_ENV:
            debug_path.mkdir(parents=True, exist_ok=True)

    # Time parameters (TODO make some of these inputs?)
    cur_time = 0.0
    traj_end_time = nominal_traj.times[-1]
    max_stopping_time = 30  # seconds
    max_time = traj_end_time + max_stopping_time
    rollout_duration = 5  # seconds
    execution_duration = 1  # seconds (How much of the rollout to actually execute)

    # TEMP, TESTING. IMPROVE THIS
    main_env.set_planning_duration(rollout_duration)
    vec_env.env_method("set_planning_duration", rollout_duration)

    # State machine
    mode = AstrobeeMPCEnv.FlightStates.NOMINAL

    point_ids = None
    cur_idx = 0

    # Execute the main MPC code in a try/finally block to make sure things close out / clean up when done
    try:
        while True:
            # TODO handle the stopping mode better... Should we be "stopping" if the rollout or planned execution
            # reaches the end of the trajectory? Or just when we've fully reached the end?

            # Determine the duration of the rollout and where on the trajectory we are interested
            remaining_traj_time = traj_end_time - cur_time
            remaining_total_time = max_time - cur_time
            out_of_time = remaining_total_time <= dt

            # Update our flight state machine (TODO improve logic, make separate method)
            if (
                remaining_traj_time <= rollout_duration
                and mode == AstrobeeMPCEnv.FlightStates.NOMINAL
            ):
                # Update this flag in our environments only once, when this changes
                print("Setting flight state to SLOWING")
                main_env.set_flight_state(AstrobeeMPCEnv.FlightStates.SLOWING)
                vec_env.env_method(
                    "set_flight_state", AstrobeeMPCEnv.FlightStates.SLOWING
                )
                mode = AstrobeeMPCEnv.FlightStates.SLOWING
            if (
                remaining_traj_time <= dt
                and mode != AstrobeeMPCEnv.FlightStates.STOPPING
            ):
                # Update this flag in our environments only once, when this changes
                print("Setting flight state to STOPPING")
                main_env.set_flight_state(AstrobeeMPCEnv.FlightStates.STOPPING)
                vec_env.env_method(
                    "set_flight_state", AstrobeeMPCEnv.FlightStates.STOPPING
                )
                mode = AstrobeeMPCEnv.FlightStates.STOPPING
            if out_of_time:
                print_red("Terminating due to time limit")
                break
            if mode in {
                AstrobeeMPCEnv.FlightStates.SLOWING,
                AstrobeeMPCEnv.FlightStates.STOPPING,
            }:
                lookahead_idx = -1
            else:
                lookahead_idx = np.searchsorted(
                    nominal_traj.times, cur_time + rollout_duration
                )
            # Clear any previously visualized trajectories before viewing the new plan
            if debug:
                vec_env.env_method("unshow_traj_plan", indices=[debug_env_idx])
                if point_ids is not None:
                    for pid in point_ids:
                        vec_env.env_method(
                            "send_client_command", "removeUserDebugItem", pid
                        )

            # THIS IS WEIRD
            # The thinking here is that when we are slowing down we might have a plan that we will get to our goal in
            # like 2 seconds even if the rollout duration is 5 seconds
            target_duration = min(max(0, remaining_traj_time), rollout_duration)
            # Set the desired state of the robot at the lookahead point
            target_state = [
                nominal_traj.positions[lookahead_idx],
                nominal_traj.quaternions[lookahead_idx],
                nominal_traj.linear_velocities[lookahead_idx],
                nominal_traj.angular_velocities[lookahead_idx],
                nominal_traj.linear_accels[lookahead_idx],
                nominal_traj.angular_accels[lookahead_idx],
                target_duration,
            ]
            main_env.set_target_state(*target_state)
            vec_env.env_method("set_target_state", *target_state)
            # Generate sampled trajectories within each vec env
            vec_env.env_method("sample_trajectory")

            # HACK
            n = vec_env.get_attr("traj_plan", [0])[0].num_timesteps
            # Handle arm traj
            arm_traj_plan = nominal_arm_traj.get_segment(cur_idx, cur_idx + n)
            # main_env.set_arm_traj(arm_traj_plan)
            vec_env.env_method("set_arm_traj", arm_traj_plan)

            if debug:
                env_state_samples = vec_env.get_attr("sampled_end_state")
                env_poss = [s[0] for s in env_state_samples]
                point_ids = vec_env.env_method(
                    "send_client_command",
                    "addUserDebugPoints",
                    env_poss,
                    [[1, 1, 1]] * len(env_poss),
                    10,
                    0,
                    indices=[debug_env_idx],
                )
                vec_env.env_method("show_traj_plan", 10, indices=[debug_env_idx])
            # Stepping in the vec env will follow the sampled trajectory
            # Action input in step(actions) is a dummy parameter for now, just for Gym compatibility
            if RECORD_DEBUG_ENV:
                debug_log_id = vec_env.env_method(
                    "send_client_command",
                    "startStateLogging",
                    main_env.client.STATE_LOGGING_VIDEO_MP4,
                    f"{DEBUG_VIDEO_DIRECTORY}{video_index}.mp4",
                    indices=[debug_env_idx],
                )[0]
            env_obs, env_rewards, env_dones, env_infos = vec_env.step(
                np.zeros(n_vec_envs)
            )
            if RECORD_DEBUG_ENV:
                vec_env.env_method(
                    "send_client_command",
                    "stopStateLogging",
                    debug_log_id,
                    indices=[debug_env_idx],
                )
                video_index += 1
            best_traj: Trajectory = vec_env.get_attr(
                "traj_plan", [int(np.argmax(env_rewards))]
            )[0]
            # Follow the best rollout in the main environment. (Use dummy action value in step call)
            n_execution_timesteps = int(
                best_traj.num_timesteps * (execution_duration / rollout_duration)
            )
            # print("N EXECUTION TIMESTEPS: ", n_execution_timesteps)
            # print("N ROLLOUT TIMESTEPS: ", lookahead_idx - cur_idx)
            # print("CUR IDX: ", cur_idx)
            main_env.traj_plan = best_traj.get_segment(0, n_execution_timesteps)
            main_env.set_arm_traj(
                nominal_arm_traj.get_segment(cur_idx, cur_idx + n_execution_timesteps)
            )
            if RECORD_MAIN_ENV:
                main_log_id = main_env.client.startStateLogging(
                    main_env.client.STATE_LOGGING_VIDEO_MP4,
                    f"{MAIN_VIDEO_DIRECTORY}{video_index}.mp4",
                )
            (
                main_obs,
                main_reward,
                main_terminated,
                main_truncated,
                main_info,
            ) = main_env.step(0)
            if RECORD_MAIN_ENV:
                main_env.client.stopStateLogging(main_log_id)
                video_index += 1

            robot_state, bag_state = main_obs

            # Update our knowledge of the last acceleration commands
            main_env.last_accel_cmd = best_traj.linear_accels[-1]
            main_env.last_alpha_cmd = best_traj.angular_accels[-1]
            vec_env.set_attr("last_accel_cmd", best_traj.linear_accels[-1])
            vec_env.set_attr("last_alpha_cmd", best_traj.angular_accels[-1])

            # Update our time information
            cur_time += execution_duration
            # TODO should the cur_time value actually be times[cur_idx]????
            cur_idx += n_execution_timesteps

            # Update the camera if we're taking video. These are hardcoded for the JPM motion
            # Switch cameras when the robot base passes x = 2.5

            if (
                (RECORD_MAIN_ENV or RECORD_DEBUG_ENV)
                and not camera_moved
                and robot_state[0][0] >= 2.5
            ):
                main_env.client.resetDebugVisualizerCamera(*JPM_VIEW)
                vec_env.env_method(
                    "send_client_command", "resetDebugVisualizerCamera", *JPM_VIEW
                )
                camera_moved = True

            # Check if we've successfully completed the trajectory
            if main_terminated:
                print_green("Success! Stabilized at end of trajectory within tolerance")
                break
            # We are not done, so reset the environments back to the same point as the main env
            # Ensure that the vec envs start from the same point as the main simulation
            if use_deformable_rollouts:
                # If we are using the deformable bag for rollouts, we have to fully save the state to disk (slow)
                # because there is no other way to restore the deformable
                saved_file = main_env.save_state()
                vec_env.env_method("restore_state", saved_file)
            else:
                # If we're using the simple rigid bag for rollouts, we can just do a very simple reset mechanic
                vec_env.env_method("reset_robot_state", robot_state)
                vec_env.env_method("reset_bag_state", bag_state)

        input("Complete. Press Enter to exit")
    finally:
        if RECORD_MAIN_ENV:
            concatenate_videos(MAIN_VIDEO_DIRECTORY, cleanup=True)
        if RECORD_DEBUG_ENV:
            concatenate_videos(DEBUG_VIDEO_DIRECTORY, cleanup=True)
        print("Closing environments")
        main_env.close()
        vec_env.close()


def _test_node_2_to_jpm():
    """Quick function to test that the parallel MPC is working as expected"""
    random_seed = 0
    np.random.seed(random_seed)
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    bag_name = "top_handle_symmetric"
    bag_mass = 10
    n_vec_envs = 10
    debug = True
    use_deformable_main_sim = True
    use_deformable_rollouts = False
    parallel_mpc_main(
        start_pose,
        end_pose,
        n_vec_envs,
        bag_name,
        bag_mass,
        use_deformable_main_sim,
        use_deformable_rollouts,
        debug,
        random_seed,
    )


def _test_jpm_to_us_lab():
    """Quick function to test that the parallel MPC is working as expected"""

    random_seed = 0
    np.random.seed(random_seed)
    start_pose = [6, 0, 0.2, 0, 0, 1, 0]  # JPM
    end_pose = [-0.063, -8.5355, 0, 0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2]  # US
    bag_name = "top_handle_symmetric"
    bag_mass = 10
    n_vec_envs = 10
    debug = False
    use_deformable_main_sim = True
    use_deformable_rollouts = False
    parallel_mpc_main(
        start_pose,
        end_pose,
        n_vec_envs,
        bag_name,
        bag_mass,
        use_deformable_main_sim,
        use_deformable_rollouts,
        debug,
        random_seed,
    )


def _test_node_2_to_us_lab():
    """Quick function to test that the parallel MPC is working as expected"""

    random_seed = 0
    np.random.seed(random_seed)
    start_pose = [0, 0, 0, 0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2]  # N2
    end_pose = [-0.063, -8.5355, 0, 0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2]  # US
    bag_name = "top_handle_symmetric"
    bag_mass = 10
    n_vec_envs = 10
    debug = False
    use_deformable_main_sim = True
    use_deformable_rollouts = False
    parallel_mpc_main(
        start_pose,
        end_pose,
        n_vec_envs,
        bag_name,
        bag_mass,
        use_deformable_main_sim,
        use_deformable_rollouts,
        debug,
        random_seed,
    )


if __name__ == "__main__":
    _test_node_2_to_jpm()
    # _test_jpm_to_us_lab()
    # _test_node_2_to_us_lab()
