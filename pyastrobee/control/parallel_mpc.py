"""This should be a TEMPORARY file that will get merged with the main mpc.py file

There will be a lot of crossover between the two files with the main difference being that this
will include the vectorized environments. Once we know that works well, we can update the main file
"""

import pybullet
import numpy as np
import numpy.typing as npt

import gymnasium as gym
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv  # make_vec_env


from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.environments import AstrobeeEnv, make_vec_env
from pyastrobee.trajectories.trajectory import visualize_traj, Trajectory
from pyastrobee.trajectories.planner import plan_trajectory

# TODO turn this into a class for the new version of the MPC?
# TODO add stopping at the end of the trajectory back in

# TODO: we should have 3 types of envs
# 1: The main simulation
# 2: One parallel environment which evaluates the "nominal" trajectory
# 3: Other parallel environments which evaluate deviations on the nominal value
# The main simulation should be visualized, and at least one of the parallel envs should be visualized
# We should probably just visualize the one nominal env


def parallel_mpc_main(
    start_pose: npt.ArrayLike,
    goal_pose: npt.ArrayLike,
    duration: float,
    debug: bool = False,
):
    dt = 1 / 350  # make this better
    cur_idx = 0
    n_rollout_steps = 100
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

    # Add some buffer time to the end of the trajectory for stopping
    max_stopping_time = 3  # seconds
    max_steps = nominal_traj.num_timesteps + round(max_stopping_time / dt)
    end_idx = nominal_traj.num_timesteps - 1
    cur_idx = 0
    prev_accel = 0  # TODO improve this handling
    prev_alpha = 0
    step_count = 0

    # Set up vectorized environments
    n_vec_envs = 5
    env_kwargs = {"is_primary": False, "use_gui": False}  # For vec envs
    # Enable GUI for one of the vec envs, and use this to test the nominal (non-sampled) trajs
    per_env_kwargs = {0: {"use_gui": True, "nominal_rollouts": True}}
    main_env = AstrobeeEnv(is_primary=True, use_gui=True)
    vec_env = make_vec_env(
        AstrobeeEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
        per_env_kwargs=per_env_kwargs,
    )
    main_env.reset()
    vec_env.reset()

    debug_env_idx = 0

    # Execute the main MPC code in a try/finally block to make sure things close out / clean up when done
    try:
        while True:
            if step_count >= max_steps:
                print("MAX STEPS EXCEEDED")
                break
            if debug:
                vec_env.env_method("unshow_traj_plan", indices=[debug_env_idx])
            if cur_idx == end_idx:  # TODO add stopping criteria
                break
            #
            saved_file = main_env.save_state()
            vec_env.env_method("restore_state", saved_file)
            lookahead_idx = min(cur_idx + n_rollout_steps, end_idx)
            # Set the desired state of the robot at the lookahead point
            target_state = [
                nominal_traj.positions[lookahead_idx],
                nominal_traj.quaternions[lookahead_idx],
                nominal_traj.linear_velocities[lookahead_idx],
                nominal_traj.angular_velocities[lookahead_idx],
                nominal_traj.linear_accels[lookahead_idx],
                nominal_traj.angular_accels[lookahead_idx],
            ]
            main_env.set_target_state(*target_state)
            vec_env.env_method("set_target_state", *target_state)
            # Generate sampled trajectories within each vec env
            vec_env.env_method(
                "sample_trajectory",
                min(n_rollout_steps, lookahead_idx - cur_idx + 1),  # CHECK THIS
            )
            if debug:
                vec_env.env_method("show_traj_plan", 10, indices=[debug_env_idx])

            # Stepping in the vec env will follow the sampled trajectory
            actions = np.ones(n_vec_envs)  # Dummy (TODO update?)
            observation, reward, done, info = vec_env.step(actions)
            best_traj: Trajectory = vec_env.get_attr(
                "traj_plan", [int(np.argmax(reward))]
            )[0]
            # Follow the best rollout in the main environment. (Use dummy action value in step call)
            main_env.traj_plan = best_traj
            observation, reward, terminated, truncated, info = main_env.step(0)
            # Update loop variables
            cur_idx = lookahead_idx
            step_count += n_rollout_steps
            # Update our knowledge of the last acceleration commands
            main_env.last_accel_cmd = best_traj.linear_accels[-1]
            main_env.last_alpha_cmd = best_traj.angular_accels[-1]
            vec_env.set_attr("last_accel_cmd", best_traj.linear_accels[-1])
            vec_env.set_attr("last_alpha_cmd", best_traj.angular_accels[-1])

    finally:
        main_env.close()
        vec_env.close()


def _test_parallel_mpc(debug=False):
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    duration = 5
    parallel_mpc_main(start_pose, end_pose, duration, debug)


if __name__ == "__main__":
    _test_parallel_mpc()
