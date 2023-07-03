"""Gym and vectorized environments for the Astrobee/ISS/Cargo setup

Note: some of this setup is specific to the MPC formulation
... TODO make a AstrobeeMPCEnv(AstrobeeEnv) where the MPC version implements the step/reset functions
in the specific way we need for MPC

TODO
- Add ability to save/restore state from a state ID (saved in memory) -- ONLY if this is useful
"""

import os
import time
from pathlib import Path
from typing import Optional, Any, Callable, Dict, Type, Union
from datetime import datetime

import pybullet
import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium.core import ObsType, ActType
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.iss import load_iss
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.trajectories.planner import plan_trajectory
from pyastrobee.trajectories.rewards_and_penalties import deviation_penalty
from pyastrobee.trajectories.trajectory import Trajectory, visualize_traj
from pyastrobee.trajectories.sampling import generate_trajs
from pyastrobee.utils.debug_visualizer import remove_debug_objects


class AstrobeeEnv(gym.Env):

    # Saved state class params
    SAVE_STATE_DIR = "artifacts/saved_states/"
    SAVE_STATE_PATHS = []

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

    def set_target_state(self, pos, orn, vel, omega, accel, alpha):
        self.target_pos = pos
        self.target_orn = orn
        self.target_vel = vel
        self.target_omega = omega
        self.target_accel = accel
        self.target_alpha = alpha

    def sample_trajectory(self, n_steps):
        # note that set_target_state should be called before this
        pos, orn, vel, omega = self.robot.dynamics_state
        n_trajs = 1
        self.traj_plan = generate_trajs(
            pos,
            orn,
            vel,
            omega,
            self.last_accel_cmd,
            self.last_alpha_cmd,
            self.target_pos,
            self.target_orn,
            self.target_vel,
            self.target_omega,
            self.target_accel,
            self.target_alpha,
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


def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    per_env_kwargs: Optional[Dict[int, Dict[str, Any]]] = None,  # NEW
) -> VecEnv:
    """Modified version of make_vec_env from Stable Baselines (SB3) to allow for specifying input
    parameters on a per-environment basis

    The main changes are any lines associated with the per_env_kwargs input. Updated SB3 docstring below:

    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :param per_env_kwargs: Like env-kwargs, keyword arguments fore the env constructor, but keyed to
        allow for different initialization on a per-env basis via the env's rank. These override any
        default values set via the env_kwargs input
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    per_env_kwargs = per_env_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            this_env_kwargs = env_kwargs | per_env_kwargs.get(rank, {})

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(this_env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **this_env_kwargs)
            else:
                env = env_id(**this_env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = (
                os.path.join(monitor_dir, str(rank))
                if monitor_dir is not None
                else None
            )
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls(
        [make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs
    )
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


def _test_envs():
    """Run a quick test of the environment generation methods"""
    # Create one primary non-vectorized environment
    main_env = AstrobeeEnv(is_primary=True, use_gui=True)
    # Create a few vectorized environments
    n_vec_envs = 4
    env_kwargs = {"is_primary": False, "use_gui": False}
    # Let one vectorized environment use the GUI for debugging
    per_env_kwargs = {0: {"use_gui": True}}
    vec_envs = make_vec_env(
        AstrobeeEnv,
        n_vec_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_vec_envs > 1 else DummyVecEnv,
        per_env_kwargs=per_env_kwargs,
    )
    try:
        # Reset has to be called first
        main_env.reset()
        vec_envs.reset()
        # Call step with dummy action values, note difference in return parameters
        observation, reward, terminated, truncated, info = main_env.step(0)
        observation, reward, done, info = vec_envs.step(np.zeros(n_vec_envs))
        input("Stepped. Press Enter to finish")
    finally:
        # Terminate pybullet processes, delete any saved states
        main_env.close()
        vec_envs.close()


if __name__ == "__main__":
    _test_envs()
