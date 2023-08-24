"""MPC with vectorized environments running in parallel

We create three different types of environments:
1: The main simulation
2: One parallel environment which evaluates the "nominal" trajectory
3: Other parallel environments which evaluate deviations on the nominal value

When debugging, we visualize the nominal parallel environment as well, and show the trajectory rollout plan

TODO:
- Stopping criteria at end of trajectory (use the terminated/truncated parameters somehow)
- Merge this with the main MPC file?
- Turn this into a controller class, or just leave as a script?
"""

import numpy as np
import numpy.typing as npt
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv

from pyastrobee.core.environments import AstrobeeMPCEnv, make_vec_env
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.planner import global_planner


def parallel_mpc_main(
    start_pose: npt.ArrayLike,
    goal_pose: npt.ArrayLike,
    n_vec_envs: int,
    debug: bool = False,
):
    """Launches a series of environments in parallel and runs a model-predictive-controller to move Astrobee between
    two poses while carrying a cargo bag

    Args:
        start_pose (npt.ArrayLike): Starting pose of the Astrobee (position and XYZW quaternion), shape (7,)
        goal_pose (npt.ArrayLike): Ending pose of the Astrobee (position and XYZW quaternion), shape (7,)
        n_vec_envs (int): Number of vectorized environments to launch in parallel (>= 1)
        debug (bool, optional): Whether to launch one of the vectorized environments with the GUI active, to visualize
            some of the rollouts being evaluated. Defaults to False.
    """
    if n_vec_envs < 1:
        raise ValueError("Must have at least one environment for evaluating rollouts")
    # Set up main environment
    main_env = AstrobeeMPCEnv(use_gui=True, is_primary=True)
    # Set up vectorized environments
    env_kwargs = {"use_gui": False, "is_primary": False}
    debug_env_idx = 0
    # Enable GUI for one of the vec envs if debugging, and use this to test the nominal (non-sampled) trajs
    per_env_kwargs = {debug_env_idx: {"use_gui": debug, "nominal_rollouts": True}}
    vec_env = make_vec_env(
        AstrobeeMPCEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
        per_env_kwargs=per_env_kwargs,
    )
    main_env.reset()
    vec_env.reset()

    # Generate nominal trajectory
    dt = main_env.client.getPhysicsEngineParameters()["fixedTimeStep"]
    nominal_traj = global_planner(
        start_pose[:3],
        start_pose[3:],
        goal_pose[:3],
        goal_pose[3:],
        dt,
    )

    # Time parameters (TODO make some of these inputs?)
    cur_time = 0.0
    traj_end_time = nominal_traj.times[-1]
    max_stopping_time = 3  # seconds
    max_time = traj_end_time + max_stopping_time
    target_rollout_duration = 1  # seconds

    # Init stopping mode for when we get to the end of the trajectory
    stopping = False

    # Execute the main MPC code in a try/finally block to make sure things close out / clean up when done
    try:
        while True:
            # Determine the duration of the rollout and where on the trajectory we are interested
            remaining_traj_time = traj_end_time - cur_time
            remaining_total_time = max_time - cur_time
            stopping = remaining_traj_time <= dt
            done = remaining_total_time <= dt
            if done:
                print("Terminating due to time limit")
                break
            if stopping:
                rollout_duration = min(target_rollout_duration, remaining_total_time)
                lookahead_idx = -1
            else:  # Following the trajectory
                rollout_duration = min(target_rollout_duration, remaining_traj_time)
                # Handle the case where we are nearing the end of the trajectory
                if remaining_traj_time < target_rollout_duration:
                    lookahead_idx = -1
                else:
                    lookahead_idx = np.searchsorted(
                        nominal_traj.times, cur_time + rollout_duration
                    )
            # Clear any previously visualized trajectories before viewing the new plan
            if debug:
                vec_env.env_method("unshow_traj_plan", indices=[debug_env_idx])

            # Ensure that the vec envs start from the same point as the main simulation
            saved_file = main_env.save_state()
            vec_env.env_method("restore_state", saved_file)
            # Set the desired state of the robot at the lookahead point
            target_state = [
                nominal_traj.positions[lookahead_idx],
                nominal_traj.quaternions[lookahead_idx],
                nominal_traj.linear_velocities[lookahead_idx],
                nominal_traj.angular_velocities[lookahead_idx],
                nominal_traj.linear_accels[lookahead_idx],
                nominal_traj.angular_accels[lookahead_idx],
                rollout_duration,
            ]
            main_env.set_target_state(*target_state)
            vec_env.env_method("set_target_state", *target_state)
            # Generate sampled trajectories within each vec env
            vec_env.env_method("sample_trajectory")
            if debug:
                vec_env.env_method("show_traj_plan", 10, indices=[debug_env_idx])
            # Stepping in the vec env will follow the sampled trajectory
            # Action input in step(actions) is a dummy parameter for now, just for Gym compatibility
            observation, reward, done, info = vec_env.step(np.zeros(n_vec_envs))
            best_traj: Trajectory = vec_env.get_attr(
                "traj_plan", [int(np.argmax(reward))]
            )[0]
            # Follow the best rollout in the main environment. (Use dummy action value in step call)
            main_env.traj_plan = best_traj
            observation, reward, terminated, truncated, info = main_env.step(0)

            # TODO: use stopping criteria in terminated/truncated
            # Idea: terminated -> successfully stopped; truncated -> out of time
            # if terminated: print_green(success_message); elif truncated: print_red(failure_message); then break

            # Update our knowledge of the last acceleration commands
            main_env.last_accel_cmd = best_traj.linear_accels[-1]
            main_env.last_alpha_cmd = best_traj.angular_accels[-1]
            vec_env.set_attr("last_accel_cmd", best_traj.linear_accels[-1])
            vec_env.set_attr("last_alpha_cmd", best_traj.angular_accels[-1])

            # Update our time information
            cur_time += rollout_duration

        input("Complete. Press Enter to exit")
    finally:
        print("Closing environments")
        main_env.close()
        vec_env.close()


def _test_parallel_mpc():
    """Quick function to test that the parallel MPC is working as expected"""
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    n_vec_envs = 5
    debug = True
    parallel_mpc_main(start_pose, end_pose, n_vec_envs, debug)


if __name__ == "__main__":
    _test_parallel_mpc()
