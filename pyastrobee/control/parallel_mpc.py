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

from pyastrobee.core.environments import AstrobeeEnv, make_vec_env
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.trajectories.planner import plan_trajectory


def parallel_mpc_main(
    start_pose: npt.ArrayLike,
    goal_pose: npt.ArrayLike,
    duration: float,
    n_vec_envs: int,
    debug: bool = False,
):
    # Set up main environment
    main_env = AstrobeeEnv(is_primary=True, use_gui=True)
    # Set up vectorized environments
    env_kwargs = {"is_primary": False, "use_gui": False}
    debug_env_idx = 0
    # Enable GUI for one of the vec envs if debugging, and use this to test the nominal (non-sampled) trajs
    per_env_kwargs = {debug_env_idx: {"use_gui": debug, "nominal_rollouts": True}}
    vec_env = make_vec_env(
        AstrobeeEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
        per_env_kwargs=per_env_kwargs,
    )
    main_env.reset()
    vec_env.reset()

    # Generate nominal trajectory
    dt = main_env.client.getPhysicsEngineParameters()["fixedTimeStep"]
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
    step_count = 0

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
            # Ensure that the vec envs start from the same point as the main simulation
            saved_file = main_env.save_state()
            vec_env.env_method("restore_state", saved_file)
            # Set the desired state of the robot at the lookahead point
            lookahead_idx = min(cur_idx + n_rollout_steps, end_idx)
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
            # Action input in step(actions) is a dummy parameter for now, just for Gym compatibility
            observation, reward, done, info = vec_env.step(np.zeros(n_vec_envs))
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

        input("Complete. Press Enter to exit")
    finally:
        print("Closing environments")
        main_env.close()
        vec_env.close()


def _test_parallel_mpc():
    """Quick function to test that the parallel MPC is working as expected"""
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    duration = 5
    n_vec_envs = 5
    debug = True
    parallel_mpc_main(start_pose, end_pose, duration, n_vec_envs, debug)


if __name__ == "__main__":
    _test_parallel_mpc()
