"""Gym and vectorized environments for the Astrobee/ISS/Cargo setup

Note: some of this setup is specific to the MPC formulation
... TODO make a AstrobeeMPCEnv(AstrobeeEnv) where the MPC version implements the step/reset functions
in the specific way we need for MPC

TODO
- Add ability to save/restore state from a state ID (saved in memory) -- ONLY if this is useful
"""

from pathlib import Path
from typing import Optional, Any
from datetime import datetime
import time

import pybullet
import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium.core import ObsType, ActType
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv  # make_vec_env

from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.iss import load_iss
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.trajectories.planner import plan_trajectory
from pyastrobee.trajectories.rewards_and_penalties import deviation_penalty
from pyastrobee.trajectories.trajectory import Trajectory, visualize_traj
from pyastrobee.trajectories.sampling import generate_trajs
from pyastrobee.core.make_vec_env import make_vec_env  # TODO MOVE THIS INTO THIS FILE
from pyastrobee.utils.debug_visualizer import remove_debug_objects

# Idea: could have a parameter upon initialization of the env to make one the "main"
# simulation and then have the vectorized envs work for this main sim?
# Figure out: Should we create all sims though the make_vec_env or should we make a main
# simulation outside of this process??
# ^^ main simulation non-vectorized, rollout sims vectorized
# TODO figure out if the is_primary input thing I did is the best way to do this...

# .. maybe have different methods depending on if we have the main environment or not?
# class AstrobeeEnv(AstrobeeBaseEnv)
# class AstrobeeVecEnv(AstrobeeBaseEnv )


# Note that the returns from calling functions like step() and reset() in a vectorized env
# differ from that of calling these from the environment itself
# For instance: step
# Returns:
#     Tuple of:
#         np.ndarray: Observations, length n_envs (CHECK)
#         np.ndarray: Rewards, length n_envs (CHECK)
#         np.ndarray: Terminated, (boolean array), length n_envs (CHECK)
#         tuple[dict[str, Any], ...]: Info (Includes truncated info as 'TimeLimit.truncated'), length n_envs
#             (One dict per env)


class AstrobeeEnv(gym.Env):

    # Saved state class params
    SAVE_STATE_DIR = "artifacts/saved_states/"
    SAVE_STATE_PATHS = []
    # Acceleration continuity class params
    # (these can be updated in the master simulation and accessed by the rollout sims)
    # TODO FIGURE OUT IF UPDATIN THIS COULD CAUSE PARALLEL/RACE ISSUES
    # HANDLE THIS DIFFERENTLY... use set_attr?
    # LAST_ACCEL_CMD = 0.0  # init
    # LAST_ALPHA_CMD = 0.0  # init
    # ^^ These were replaced with instance variables

    def __init__(
        self,
        is_primary: bool,
        use_gui: bool,
        nominal_rollouts: bool = False,
        cleanup: bool = True,
    ):
        # TODO: make more of these parameters variables to pass in
        # e.g. initial bag/robot position, number of robots, type of bag, ...
        self.client = initialize_pybullet(use_gui)
        self.iss_ids = load_iss(client=self.client)
        self.robot = Astrobee(client=self.client)
        self.bag = CargoBag("top_handle", client=self.client)
        self.bag.reset_to_handle_pose(self.robot.ee_pose)
        self.bag.attach_to(self.robot, object_to_move="bag")
        self.dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        # TODO figure out how to handle controller parameters
        # Just fixing the gains here for now
        kp, kv, kq, kw = 20, 5, 1, 0.1  # TODO make parameters
        self.controller = ForceTorqueController(
            self.robot.id,
            self.robot.mass,
            self.robot.inertia,
            kp,
            kv,
            kq,
            kw,
            self.dt,
            client=self.client,
        )
        # Penalty multipliers for evaluating rollouts. TEMPORARY: figure these out, make parameters
        self.pos_penalty = 1
        self.orn_penalty = 1
        self.vel_penalty = 1
        self.ang_vel_penalty = 1

        # Sampling parameters (these need refinement)
        self.pos_stdev = 0.05
        self.orn_stdev = 0.05
        self.vel_stdev = 0.05
        self.ang_vel_stdev = 0.05
        self.accel_stdev = 0.05
        self.alpha_stdev = 0.05

        # Store last acceleration commands
        # Update through set_attr
        self.last_accel_cmd = 0.0  # init
        self.last_alpha_cmd = 0.0  # init

        # Keep track of any temporary debug visualizer IDs
        self.debug_viz_ids = ()

        self._is_primary = is_primary
        self._nominal_rollouts = nominal_rollouts
        self._cleanup = cleanup
        self.traj_plan = None  # Init
        self.target_pos = None  # Init
        self.target_orn = None  # Init
        self.target_vel = None  # Init
        self.target_omega = None  # Init
        # Dummy parameters for gym/stable baselines compatibility
        self.observation_space = gym.spaces.Discrete(3)  # temporary
        self.action_space = gym.spaces.Discrete(3)  # temporary
        # Step the simulation once to get the bag in the right place
        self.client.stepSimulation()
        # TODO decide if saving the initial state is useful...
        # I was thinking maybe it would be nice to have a clean slate we could go back to
        # but I don't quite know what the use case would be right now
        if self.is_primary_simulation:
            self.initial_saved_state = self.save_state()

    @property
    def is_primary_simulation(self) -> bool:
        """Whether this environment is running the primary planning/control simulation
        or is a separate (likely vectorized) environment for evaluating rollouts"""
        return self._is_primary

    # def set_target_state(self, a0, dw0, pf, qf, vf, wf, af, dwf, duration) -> None:
    #     # Note: for the primary simulation, this will be the overall goal and will
    #     # generate the nominal reference trajectory. For the vectorized sims, this will
    #     # generate the rollout trajectories
    #     self.target_pos = pf
    #     self.target_orn = qf
    #     self.target_vel = vf
    #     self.target_omega = wf
    #     # Any time we update a goal state we'll need to do a replan
    #     p0, q0, v0, w0 = self.robot.dynamics_state
    #     self.traj_plan = plan_trajectory(
    #         p0, q0, v0, w0, a0, dw0, pf, qf, vf, wf, af, dwf, duration, self.dt
    #     )

    # def set_nominal_trajectory(self, traj: Trajectory):
    #     self.traj_plan = traj

    # TODO is there a better way of doing this
    def _set_nominal_target(self, pos, orn, vel, omega, accel, alpha):
        self.target_pos = pos
        self.target_orn = orn
        self.target_vel = vel
        self.target_omega = omega
        self.target_accel = accel
        self.target_alpha = alpha

    def sample_trajectory(
        self, des_pos, des_orn, des_vel, des_omega, des_accel, des_alpha, n_steps
    ):
        self._set_nominal_target(
            des_pos, des_orn, des_vel, des_omega, des_accel, des_alpha
        )
        pos, orn, vel, omega = self.robot.dynamics_state
        n_trajs = 1
        self.traj_plan = generate_trajs(
            pos,
            orn,
            vel,
            omega,
            self.last_accel_cmd,
            self.last_alpha_cmd,
            des_pos,
            des_orn,
            des_vel,
            des_omega,
            des_accel,
            des_alpha,
            self.pos_stdev,
            self.orn_stdev,
            self.vel_stdev,
            self.ang_vel_stdev,
            self.accel_stdev,
            self.alpha_stdev,
            n_trajs,
            n_steps,
            self.dt,
            include_nominal_traj=self._nominal_rollouts,
        )[0]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        # Implementation of Gym template method reset(): See Gym for full method docstring
        # Gym states this must be the first line of the reset() method
        super().reset(seed=seed)
        # TODO: add resets for the instance variables? Class variables?
        return self._get_obs(), self._get_info()  # Initial state observation

    def _get_obs(self) -> ObsType:
        """Translates the environment's state into an observation

        Returns:
            ObsType: Observation
        """
        # This function setup was recommended in the gym documentation
        return 123  # Dummy value for now

    def _get_info(self) -> dict[str, Any]:
        """Provide auxiliary information associated with an observation

        Returns:
            dict[str, Any]: Additional observation information
        """
        # This function setup was recommended in the gym documentation
        return {"a": 1}  # Dummy value for now

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Implementation of Gym template method step(): See Gym for full method docstring
        # Note: When called from a vectorized environment, the return is different

        # This is less so a "step" function than a "rollout" function
        # TODO clear up this confusion? IDK if this is possible since gym looks for step()

        # TODO decide if the step function should sample a trajectory and then roll it out?
        # Or should the trajectory be already sampled prior to this

        # Main simulation should evaluate the number of steps in the MPC rollout length
        # Vectorized simulations should evaluate their full trajectory plan (the rollout itself)
        # ^^ HOWEVER, this logic should be handled externally to this function

        # TODO make this the default behavior in the non-MPC version of this environment
        if self.traj_plan is None:
            self.client.stepSimulation()
            reward = 0.123  # TODO FIX
        else:
            # TODO decide how to handle the stopping criteria
            self.controller.follow_traj(
                self.traj_plan, stop_at_end=False, max_stop_iters=None
            )
            # Get the state of the robot after we follow the trajectory
            pos, orn, vel, omega = self.robot.dynamics_state
            # Determine the reward based on how much we deviated from the target
            # Note that the target can be different from the last state in the traj plan
            # since that last state was sampled about the nominal target
            reward = -1 * deviation_penalty(
                pos,
                orn,
                vel,
                omega,
                self.target_pos,
                self.target_orn,
                self.target_vel,
                self.target_omega,
                self.pos_penalty,
                self.orn_penalty,
                self.vel_penalty,
                self.ang_vel_penalty,
            )
        # All returns are essentially dummy values except the reward
        observation = self._get_obs()
        terminated = False  # If at the terminal state
        truncated = False  # If stopping the sim before the terminal state
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def step_simulation(self):
        """Single pybullet simulation step"""
        self.client.stepSimulation()

    def close(self):
        # Implementation of Gym template method close(): See Gym for full method docstring
        self.client.disconnect()
        if self.is_primary_simulation and self._cleanup:
            # Delete all of the previous saved states at the end of the simulation process
            # TODO: decide if each session should have its own directory?
            for path in Path(AstrobeeEnv.SAVE_STATE_DIR).glob("*.bullet"):
                path.unlink()

    def save_state(self) -> str:
        """Saves the current simulation state to disk

        - Note: saved states are NOT overwritten because this could lead to issues with the parallel environments
          and race conditions (TODO test this out)
        - Saved states can be cleared out at the end of the simulation period

        Raises:
            PermissionError: This function should only be called from the main simulation instead of the rollout
                simulations, so we raise an error if we try to call this from the wrong simulation

        Returns:
            str: Path to the saved state file
        """
        # Ensure that any simulations strictly for evaluating rollouts cannot save their state
        if not self.is_primary_simulation:
            raise PermissionError("Only the primary simulation can save the state")

        # Autogenerate a filename/path and save to it
        filename = "state_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = AstrobeeEnv.SAVE_STATE_DIR + filename + ".bullet"
        self.client.saveBullet(filepath)
        AstrobeeEnv.SAVE_STATE_PATHS.append(filepath)
        return filepath

    def restore_state(self, filename: str) -> None:
        """Restores the simulation to a saved state file

        Args:
            filename (str): Path to a .bullet saved state within the saved state directory
        """
        filename = self._check_state_file(filename)
        self.client.restoreState(fileName=filename)

    def _check_state_file(self, filename: str) -> str:
        """Helper function: Validates that a saved state file exists

        Args:
            filename (str): Path to a .bullet saved state within the saved state directory

        Returns:
            str: Validated path
        """
        path = Path(filename)
        if path.suffix != ".bullet":
            raise ValueError(
                f"Invalid filename: {filename}.\nNot a .bullet saved state file"
            )
        if path.parent != Path(AstrobeeEnv.SAVE_STATE_DIR):
            raise ValueError(
                f"Invalid filename: {filename}.\nCheck that the filename points to within the saved state directory"
            )
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"Could not find file: {filename}")

    def show_traj_plan(self, n: Optional[int]) -> None:
        if self.traj_plan is None:
            raise ValueError("No trajectory available to visualize")
        self.debug_viz_ids = self.traj_plan.visualize(n, self.client)

    def unshow_traj_plan(self) -> None:
        if len(self.debug_viz_ids) == 0:
            return
        remove_debug_objects(self.debug_viz_ids, self.client)
        self.debug_viz_ids = ()


def test_vec_env():
    # TODO check on the difference between subproc and dummy
    n_envs = 2
    env_kwargs = {"is_primary": False, "use_gui": True}
    vec_env = make_vec_env(
        AstrobeeEnv,
        n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
    )
    # vec_env.env_method("step", indices=[0, 1])
    # need to call reset() before step() for some reason ... figure this out
    # vec_env.env_method("reset")
    ret = vec_env.reset()  # Returns an array of observations
    print("reset, returned: ", ret)
    # NOTE: step method must contain an iterable of length num_envs
    observations, rewards, dones, infos = vec_env.step(
        (
            np.zeros(2),
            np.zeros(2),
        )
    )
    time.sleep(0.2)
    print("stepped, returned: ", observations, rewards, dones, infos)
    vec_env.step_async(np.zeros(2))
    print("stepped async (no return)")
    ret = vec_env.step_wait()
    print("waited for step async, returned: ", ret)

    input("completed")


def test_main_and_vecs():
    # Test out creating a main GUI client which also handles saving the state
    # and then including some vectorized environments that can run headless
    # and restore the state saved from the main client
    n_vec_envs = 2
    env_kwargs = {"is_primary": False, "use_gui": True}  # For vec envs
    main_env = AstrobeeEnv(is_primary=True, use_gui=True)
    vec_envs = make_vec_env(
        AstrobeeEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
    )
    try:
        main_env.reset()
        vec_envs.reset()

    finally:
        main_env.close()
        vec_envs.close()


def test_attributes():
    n_vec_envs = 2
    env_kwargs = {"is_primary": False, "use_gui": True}  # For vec envs
    # main_env = AstrobeeEnv(is_primary=True, use_gui=True)
    vec_envs = make_vec_env(
        AstrobeeEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
    )
    try:
        # vec_envs.set_attr("use_gui")
        pass
    finally:
        # main_env.close()
        vec_envs.close()


def test_new_make_vec_env():
    """Test to see if the new make_vec_env function can handle different parameters upon initialization

    We should see that four vectorized environments are created, but one has the GUI enabled
    """
    n_vec_envs = 4
    # Set the default env kwargs to not use the GUI
    env_kwargs = {"is_primary": False, "use_gui": False}
    # For the rank-0 env, override the default and use the GUI
    per_env_kwargs = {0: {"use_gui": True}}
    vec_env = make_vec_env(
        AstrobeeEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
        per_env_kwargs=per_env_kwargs,
    )
    vec_env.reset()
    vec_env.step_async(np.zeros(2))
    input("completed")


def test_one_env():
    env = AstrobeeEnv(True, True)
    env.reset()
    ret = env.step(1)
    input("done")


if __name__ == "__main__":
    # _main()
    test_vec_env()
    # test_new_make_vec_env()
    # test_one_env()
