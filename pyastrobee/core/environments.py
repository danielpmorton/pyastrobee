"""Gym and vectorized environments for the Astrobee/ISS/Cargo setup

NOTE:
- Class variables DO NOT get updated in vectorized environments (for instance, modifying the value in a
  non-vectorized environment will be reflected in all non-vectorized environments, but not the vectorized ones.
  You'd have to explicitly call the set_attr method for that)
"""
# TODO
# - See if there is a faster way to save/restore state in the case that we're using the rigid bag
# - Add ability to save/restore state from a state ID (saved in memory) -- ONLY if this is useful
# - Use terminated/truncated as a stopping parameter
# - Should the reset() function reset the simulation back to an initial saved state?
# - Decide if the base AstrobeeEnv should have cleanup functionality

import os
from pathlib import Path
from typing import Optional, Any, Callable, Dict, Type, Union
from datetime import datetime

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
from pyastrobee.core.iss import ISS
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.rigid_bag import RigidCargoBag
from pyastrobee.trajectories.cost_functions import state_tracking_cost
from pyastrobee.trajectories.sampling import generate_trajs
from pyastrobee.utils.debug_visualizer import remove_debug_objects


class AstrobeeEnv(gym.Env):
    """Base Astrobee environment containing the Astrobee, ISS, and a cargo bag

    Args:
        use_gui (bool): Whether or not to use the GUI as opposed to headless.
        robot_pose (npt.ArrayLike, optional): Starting position + XYZW quaternion pose of the Astrobee, shape (7,)
        bag_name (str, optional): Type of cargo bag to load. Defaults to "top_handle".
        use_deformable_bag (bool, optional): Whether to load the deformable or rigid version of the bag.
            Defaults to True (load the deformable version)
    """

    SAVE_STATE_DIR = "artifacts/saved_states/"
    SAVE_STATE_PATHS = []

    def __init__(
        self,
        use_gui: bool,
        robot_pose: npt.ArrayLike = (0, 0, 0, 0, 0, 0, 1),
        bag_name: str = "top_handle",
        use_deformable_bag: bool = True,
    ):
        self.client = initialize_pybullet(use_gui)
        self.iss = ISS(client=self.client)
        self.robot = Astrobee(robot_pose, client=self.client)
        if use_deformable_bag:
            self.bag = DeformableCargoBag(bag_name, client=self.client)
        else:
            self.bag = RigidCargoBag(bag_name, client=self.client)
        self.bag.reset_to_handle_pose(self.robot.ee_pose)
        self.bag.attach_to(self.robot, object_to_move="bag")
        self.dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        # Dummy parameters for gym/stable baselines compatibility
        self.observation_space = gym.spaces.Discrete(3)  # temporary
        self.action_space = gym.spaces.Discrete(3)  # temporary
        # Step the simulation once to get the bag in the right place
        self.client.stepSimulation()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        # Implementation of Gym template method reset(): See Gym for full method docstring
        # Gym states this must be the first line of the reset() method
        super().reset(seed=seed)
        return self._get_obs(), self._get_info()  # Initial state observation

    def _get_obs(self) -> ObsType:
        """Translates the environment's state into an observation

        Returns:
            ObsType: Observation
        """
        # This function setup was recommended in the gym documentation
        return 0  # Dummy value for now

    def _get_info(self) -> dict[str, Any]:
        """Provide auxiliary information associated with an observation

        Returns:
            dict[str, Any]: Additional observation information
        """
        # This function setup was recommended in the gym documentation
        return {}  # Dummy value for now

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Implementation of Gym template method step(): See Gym for full method docstring
        # Note: The return parameters differ slightly from step() for a vectorized environment

        # In this base environment, we will just step the pybullet simulation and return a dummy value for reward
        # The MPC environment can add more specific MPC/control functionality here
        self.client.stepSimulation()
        reward = 0
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

    def save_state(self) -> str:
        """Saves the current simulation state to disk

        - Note: saved states are not currently overwritten (could lead to issues with parallel environments?). But,
          these can be cleared out at the end of the simulation period

        Returns:
            str: Path to the saved state file
        """
        # Autogenerate a unique filename/path and save to it
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


class AstrobeeMPCEnv(AstrobeeEnv):
    """Astrobee environment for MPC: Contains additional controller parameters and functions associated with MPC, on
    top of the base Astrobee environment capability

    Args:
        use_gui (bool): Whether or not to use the GUI as opposed to headless.
        is_primary (bool): Whether or not this environment is the main simulation (True) or if it is one of the
            vectorized environments for evaluating a rollout (False)
        robot_pose (npt.ArrayLike, optional): Starting position + XYZW quaternion pose of the Astrobee, shape (7,)
        bag_name (str, optional): Type of cargo bag to load. Defaults to "top_handle".
        use_deformable_bag (bool, optional): Whether to load the deformable or rigid version of the bag.
            Defaults to True (load the deformable version)
        nominal_rollouts (bool, optional): If True, will roll-out a trajectory based on the nominal target.
            If False, will sample a trajectory about the nominal target. Defaults to False.
        cleanup (bool, optional): Whether or not to delete all saved states when the simulation ends. Defaults to True.
    """

    def __init__(
        self,
        use_gui: bool,
        is_primary: bool,
        robot_pose: npt.ArrayLike = (0, 0, 0, 0, 0, 0, 1),
        bag_name: str = "top_handle",
        use_deformable_bag: bool = True,
        nominal_rollouts: bool = False,
        cleanup: bool = True,
    ):
        super().__init__(use_gui, robot_pose, bag_name, use_deformable_bag)
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
        self.target_duration = None  # Init
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

    def set_target_state(
        self,
        pos: npt.ArrayLike,
        orn: npt.ArrayLike,
        vel: npt.ArrayLike,
        omega: npt.ArrayLike,
        accel: npt.ArrayLike,
        alpha: npt.ArrayLike,
        duration: float,
    ) -> None:
        """Set the target dynamics state for planning/sampling trajectories and determining penalties

        Args:
            pos (npt.ArrayLike): Desired position, shape (3,)
            orn (npt.ArrayLike): Desired XYZW quaternion orientation, shape (4,)
            vel (npt.ArrayLike): Desired linear velocity, shape (3,)
            omega (npt.ArrayLike): Desired angular velocity, shape (3,)
            accel (npt.ArrayLike): Desired linear acceleration, shape (3,)
            alpha (npt.ArrayLike): Desired angular acceleration, shape (3,)
            duration (float): Amount of time to pass before achieving this desired state
        """
        self.target_pos = pos
        self.target_orn = orn
        self.target_vel = vel
        self.target_omega = omega
        self.target_accel = accel
        self.target_alpha = alpha
        self.target_duration = duration

    def sample_trajectory(self) -> None:
        """Samples a trajectory about the nominal target.

        - If the nominal_rollouts parameter is True for this environment, the final state of the trajectory will be
          exactly the nominal value (no noise added when sampling)
        - This should just be called in the vectorized rollout environments (not the main environment) since the main
          environment will use the best trajectory from the rollout envs
        """
        if self.is_primary_simulation:
            raise ValueError(
                "Trajectory sampling should only occur in one of parallel environments for evaluation purposes"
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
            self.target_duration,
            self.dt,
            include_nominal_traj=self._nominal_rollouts,
        )[0]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Note: For MPC, this is less so a "step" than a "rollout" function. The trajectory should be sampled
        #       before calling this function

        if self.traj_plan is None:
            raise ValueError("Trajectory has not been planned")
        # TODO decide how to handle the stopping criteria
        self.controller.follow_traj(
            self.traj_plan, stop_at_end=False, max_stop_iters=None
        )
        # Get the state of the robot after we follow the trajectory
        pos, orn, vel, omega = self.robot.dynamics_state
        # Determine the reward based on how much we deviated from the target
        reward = -1 * state_tracking_cost(
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

    def close(self):
        self.client.disconnect()
        if self.is_primary_simulation and self._cleanup:
            # Delete all of the previous saved states at the end of the simulation process
            # TODO: decide if each session should have its own directory?
            for path in Path(AstrobeeMPCEnv.SAVE_STATE_DIR).glob("*.bullet"):
                path.unlink()

    def save_state(self) -> str:
        # Ensure that any simulations strictly for evaluating rollouts cannot save their state
        if not self.is_primary_simulation:
            raise PermissionError("Only the primary simulation can save the state")
        return super().save_state()

    def show_traj_plan(self, n: Optional[int]) -> None:
        """Displays the planned trajectory on the current pybullet client GUI (if enabled)

        Args:
            n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
                Defaults to None (plot all frames)
        """
        if self.traj_plan is None:
            raise ValueError("No trajectory available to visualize")
        self.debug_viz_ids = self.traj_plan.visualize(n, self.client)

    def unshow_traj_plan(self) -> None:
        """Removes a displayed trajectory from the pybullet client GUI"""
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
    main_env = AstrobeeEnv(use_gui=True)
    # Create a few vectorized environments
    n_vec_envs = 4
    env_kwargs = {"use_gui": False}
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
