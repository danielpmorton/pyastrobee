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
from enum import Enum

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
from pyastrobee.core.abstract_bag import CargoBag
from pyastrobee.core.deformable_bag import DeformableCargoBag
from pyastrobee.core.constraint_bag import ConstraintCargoBag
from pyastrobee.control.cost_functions import safe_set_cost
from pyastrobee.trajectories.sampling import generate_trajs
from pyastrobee.utils.debug_visualizer import remove_debug_objects
from pyastrobee.utils.boxes import check_box_containment, visualize_3D_box
from pyastrobee.config.iss_safe_boxes import FULL_SAFE_SET
from pyastrobee.utils.quaternions import quaternion_dist
from pyastrobee.control.cost_functions import robot_and_bag_termination_criteria
from pyastrobee.config.astrobee_motion import MAX_FORCE_MAGNITUDE, MAX_TORQUE_MAGNITUDE
from pyastrobee.trajectories.trajectory import Trajectory, ArmTrajectory
from pyastrobee.utils.transformations import invert_transform_mat
from pyastrobee.trajectories.planner import local_planner
from pyastrobee.trajectories.trajectory import concatenate_trajs


class AstrobeeEnv(gym.Env):
    """Base Astrobee environment containing the Astrobee, ISS, and a cargo bag

    Args:
        use_gui (bool): Whether or not to use the GUI as opposed to headless.
        robot_pose (npt.ArrayLike, optional): Starting position + XYZW quaternion pose of the Astrobee, shape (7,)
        bag_name (str, optional): Type of cargo bag to load. Defaults to "top_handle".
        bag_mass (float): Mass of the cargo bag, in kg. Defaults to 10
        bag_type (type[CargoBag]): Class of cargo bag to use in the environment. Defaults to DeformableCargoBag
        load_full_iss (bool, optional): Whether to load the ISS (expensive, not necessarily required for rollouts) or
            just work with the safe set information. Defaults to True (load the ISS)
    """

    SAVE_STATE_DIR = "artifacts/saved_states/"
    SAVE_STATE_PATHS = []

    def __init__(
        self,
        use_gui: bool,
        robot_pose: npt.ArrayLike = (0, 0, 0, 0, 0, 0, 1),
        bag_name: str = "top_handle",
        bag_mass: float = 10,
        bag_type: type[CargoBag] = DeformableCargoBag,
        load_full_iss: bool = True,
    ):
        self.client = initialize_pybullet(use_gui)
        self.safe_set = FULL_SAFE_SET
        if load_full_iss:
            self.iss = ISS(client=self.client)
        elif use_gui:
            for box in self.safe_set.values():
                visualize_3D_box(box, rgba=(1, 0, 0, 0.3))
        self.robot = Astrobee(robot_pose, client=self.client)
        self.bag = bag_type(bag_name, bag_mass, client=self.client)
        self.bag.reset_to_handle_pose(self.robot.ee_pose)
        self.bag.attach_to(self.robot, object_to_move="bag")
        self.dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        # Dummy parameters for gym/stable baselines compatibility
        # TODO make custom gym.spaces.space.Space subclasses for these?
        self.observation_space = gym.spaces.Discrete(3)  # temporary, unused
        self.action_space = gym.spaces.Discrete(3)  # temporary, unused
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
        return None, None  # Dummy value for now

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

    def send_client_command(self, *args, **kwargs) -> Any:
        """Send a command to the environment's pybullet client

        For instance, we can use pybullet.getBasePositionAndOrientation with this as
        send_client_command("getBasePositionAndOrientation", body_id)

        Returns:
            Any: The return from the Pybullet command
        """
        attr = getattr(self.client, args[0])
        if isinstance(attr, Callable):
            return attr(*args[1:], **kwargs)
        return attr


class AstrobeeMPCEnv(AstrobeeEnv):
    """Astrobee environment for MPC: Contains additional controller parameters and functions associated with MPC, on
    top of the base Astrobee environment capability

    Args:
        use_gui (bool): Whether or not to use the GUI as opposed to headless.
        is_primary (bool): Whether or not this environment is the main simulation (True) or if it is one of the
            vectorized environments for evaluating a rollout (False)
        robot_pose (npt.ArrayLike, optional): Starting position + XYZW quaternion pose of the Astrobee, shape (7,)
        bag_name (str, optional): Type of cargo bag to load. Defaults to "top_handle".
        bag_mass (float): Mass of the cargo bag, in kg. Defaults to 10
        bag_type (type[CargoBag]): Class of cargo bag to use in the environment. Defaults to DeformableCargoBag
        load_full_iss (bool, optional): Whether to load the ISS (expensive, not necessarily required for rollouts) or
            just work with the safe set information. Defaults to True (load the ISS)
        nominal_rollouts (bool, optional): If True, will roll-out a trajectory based on the nominal target.
            If False, will sample a trajectory about the nominal target. Defaults to False.
        cleanup (bool, optional): Whether or not to delete all saved states when the simulation ends. Defaults to True.
    """

    class FlightStates(Enum):
        STARTING = "starting"
        NOMINAL = "nominal"
        # SLOWING = "slowing"
        STOPPING = "stopping"

    def __init__(
        self,
        use_gui: bool,
        is_primary: bool,
        robot_pose: npt.ArrayLike = (0, 0, 0, 0, 0, 0, 1),
        bag_name: str = "top_handle",
        bag_mass: float = 10,
        bag_type: type[CargoBag] = DeformableCargoBag,
        load_full_iss: bool = True,
        nominal_rollouts: bool = False,
        cleanup: bool = True,
    ):
        super().__init__(
            use_gui, robot_pose, bag_name, bag_mass, bag_type, load_full_iss
        )
        # TODO figure out how to handle controller parameters
        # Just fixing the gains here for now
        # TODO should these be functions of the bag mass???
        kp, kv, kq, kw = 20, 5, 5, 0.1  # TODO make parameters
        p = self.bag.position - self.robot.position
        self.controller = ForceTorqueController(
            self.robot.id,
            self.robot.mass + bag_mass,
            # self.robot.inertia,
            self.robot.inertia
            + bag_mass
            * (
                np.dot(p, p) * np.eye(3) - np.outer(p, p)
            ),  # TODO parallel axis theorem for bag?? test this
            kp,
            kv,
            kq,
            kw,
            self.dt,
            max_force=MAX_FORCE_MAGNITUDE,
            max_torque=MAX_TORQUE_MAGNITUDE,
            client=self.client,
        )

        # Sampling parameters (TODO these need refinement)
        self.pos_stdev = 0.1
        self.orn_stdev = 0.1
        self.vel_stdev = 0.1
        self.ang_vel_stdev = 0.1
        self.accel_stdev = 0.1
        self.alpha_stdev = 0.1

        # Store last acceleration commands
        # Update through set_attr
        self.last_accel_cmd = 0.0  # init
        self.last_alpha_cmd = 0.0  # init

        # Frequency at which we query our "stay away from the walls" cost function
        self.safe_set_eval_freq = 10  # Hz

        # Keep track of any temporary debug visualizer IDs
        self.debug_viz_ids = ()

        # Keep track of whether we're stopping or in a nominal flight mode
        self.flight_state = self.FlightStates.NOMINAL  # init

        # Store where we want the Astrobee to be at the end of the MPC run to determine if we are done
        self.goal_pose = None  # init

        # HACK - improve how this is handled
        self.arm_traj_plan = None

        self._is_primary_env = is_primary
        self._is_debugging_env = not is_primary and use_gui
        self._nominal_rollouts = nominal_rollouts
        self._cleanup = cleanup
        self.traj_plan = None  # Init
        self.target_pos = None  # Init
        self.target_orn = None  # Init
        self.target_vel = None  # Init
        self.target_omega = None  # Init
        self.target_duration = None  # Init

    @property
    def is_primary_simulation(self) -> bool:
        """Whether this environment is running the primary planning/control simulation
        or is a separate (likely vectorized) environment for evaluating rollouts"""
        return self._is_primary_env

    @property
    def is_debugging_simulation(self) -> bool:
        """Whether this is an environment launched in debug mode"""
        return self._is_debugging_env

    def set_arm_traj(self, traj: ArmTrajectory):  # TODO IMPROVE THIS
        self.arm_traj_plan = traj

    def set_flight_state(self, state: Union[str, FlightStates]):
        """Set the current flight state: for instance, whether we are in nominal operating mode, stopping, ...

        Args:
            state (Union[str, FlightStates]): A flight state or its string representation
                (i.e. "nominal", "stopping", ...)
        """
        # TODO add check that it is valid
        # TODO should we store the state as the string or the Enum????
        if isinstance(state, str):
            self.flight_state = self.FlightStates(state)
        elif isinstance(state, self.FlightStates):
            self.flight_state = state
        else:
            raise ValueError("Flight state not recognized")

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
        if self.flight_state == self.FlightStates.NOMINAL:
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
        elif self.flight_state == self.FlightStates.STOPPING:
            stopping_time_stdev = 1  # TODO refine this and move it'
            # Note: ensure that this is a positive value post-sampling
            stopping_time = np.maximum(
                np.random.normal(self.target_duration, stopping_time_stdev), 0
            )
            traj = local_planner(
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
                stopping_time,
                self.dt,
            )
            n_timesteps = round(self.target_duration / self.dt)
            # If the stopping time is greater than the rollout time (target duration), clip it
            # If the stopping time is less than the rollout time, extend it
            if stopping_time >= self.target_duration:
                self.traj_plan = traj.get_segment(0, n_timesteps)
            else:  # Less than
                # Create a trajectory at the stopped position for the remaining timesteps
                remaining_timesteps = n_timesteps - traj.num_timesteps
                stop_traj = Trajectory(
                    self.target_pos * np.ones((remaining_timesteps, 1)),
                    self.target_orn * np.ones((remaining_timesteps, 1)),
                    np.zeros((remaining_timesteps, 3)),
                    np.zeros((remaining_timesteps, 3)),
                    np.zeros((remaining_timesteps, 3)),
                    np.zeros((remaining_timesteps, 3)),
                    np.arange(remaining_timesteps) * self.dt,
                )
                self.traj_plan = concatenate_trajs(traj, stop_traj)

        self.sampled_end_state = (
            self.traj_plan.positions[-1],
            self.traj_plan.quaternions[-1],
            self.traj_plan.linear_velocities[-1],
            self.traj_plan.angular_velocities[-1],
        )

    def _get_obs(self) -> ObsType:
        if self.is_primary_simulation:
            return self.robot.full_state, self.bag.dynamics_state
        else:
            return None, None

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Note: For MPC, this is less so a "step" than a "rollout" function. The trajectory should be sampled
        #       before calling this function
        # TODO use the action parameter to pass in a trajectory to follow?

        terminated = False  # init (If at the terminal state)
        truncated = False  # init (If stopping the sim before the terminal state)

        if self.traj_plan is None:
            raise ValueError("Trajectory has not been planned")
        # Follow the trajectory. NOTE: This is effectively the same as the follow_traj() function in the controller,
        # but accessing the loop directly allows us to do more with the data at each step
        # TODO decide how to handle the stopping criteria

        if not self.traj_plan.num_timesteps == self.arm_traj_plan.num_timesteps:
            raise ValueError("Mismatched time info between base and arm trajs")

        # If this is the primary simulation, we just follow the best trajectory we have
        # Rewrd for the primary simulation doesn't mean anything, so no computation needed
        inertia_update_freq = 5
        steps_per_inertia_update = round(
            1 / (self.traj_plan.timestep * inertia_update_freq)
        )
        if self.is_primary_simulation:
            for i in range(self.traj_plan.num_timesteps):
                pos, orn, lin_vel, ang_vel = self.controller.get_current_state()
                self.controller.step(
                    pos,
                    lin_vel,
                    orn,
                    ang_vel,
                    self.traj_plan.positions[i, :],
                    self.traj_plan.linear_velocities[i, :],
                    self.traj_plan.linear_accels[i, :],
                    self.traj_plan.quaternions[i, :],
                    self.traj_plan.angular_velocities[i, :],
                    self.traj_plan.angular_accels[i, :],
                )
                self.robot.set_joint_angles(
                    self.arm_traj_plan.angles[i, :], self.arm_traj_plan.joint_ids
                )
                # TODO THIS KINDA SUCKS
                if i % steps_per_inertia_update == 0:
                    T_R2W = self.robot.tmat
                    T_B2W = self.bag.tmat
                    T_B2R = invert_transform_mat(T_R2W) @ T_B2W
                    p = T_B2R[:3, 3]
                    self.controller.inertia = self.robot.inertia + self.bag.mass * (
                        np.dot(p, p) * np.eye(3) - np.outer(p, p)
                    )

            reward = 0
        else:
            # We are in a rollout environment
            # So, follow the trajectory, but also keep track of a bunch of things so that we can compute the reward
            robot_safe_set_cost = 0  # init
            bag_safe_set_cost = 0  # init
            stabilization_cost = 0
            tracking_cost = 0
            bag_vel_cost = 0
            steps_per_safe_set_eval = round(
                1 / (self.traj_plan.timestep * self.safe_set_eval_freq)
            )
            # TODO IMPROVE THIS
            # the thought here was that if we're stopping (or slowing) we care more about the overall positioning
            # rather than just staying in the middle of the modules
            safe_set_weight = (
                1 if self.flight_state == self.FlightStates.NOMINAL else 0.1
            )
            for i in range(self.traj_plan.num_timesteps):
                # Note: the traj log gets updated whenever we access the current state
                pos, orn, lin_vel, ang_vel = self.controller.get_current_state()
                self.controller.step(
                    pos,
                    lin_vel,
                    orn,
                    ang_vel,
                    self.traj_plan.positions[i, :],
                    self.traj_plan.linear_velocities[i, :],
                    self.traj_plan.linear_accels[i, :],
                    self.traj_plan.quaternions[i, :],
                    self.traj_plan.angular_velocities[i, :],
                    self.traj_plan.angular_accels[i, :],
                )
                self.robot.set_joint_angles(
                    self.arm_traj_plan.angles[i, :], self.arm_traj_plan.joint_ids
                )
                # TODO THIS KINDA SUCKS
                if i % steps_per_inertia_update == 0:
                    T_R2W = self.robot.tmat
                    T_B2W = self.bag.tmat
                    T_B2R = invert_transform_mat(T_R2W) @ T_B2W
                    p = T_B2R[:3, 3]
                    self.controller.inertia = self.robot.inertia + self.bag.mass * (
                        np.dot(p, p) * np.eye(3) - np.outer(p, p)
                    )
                # *** COST FUNCTION ***
                # Perform collision checking on every timestep
                robot_bb = self.robot.bounding_box
                bag_bb = self.bag.bounding_box
                robot_is_safe = check_box_containment(robot_bb, self.safe_set.values())
                bag_is_safe = check_box_containment(bag_bb, self.safe_set.values())
                # If either the robot or bag collided, stop the simulation and return an effectively infinite cost
                # (Very large but not infinity to maintain sorting order in the edge case that all rollouts collide)
                if not robot_is_safe:
                    robot_safe_set_cost += 10000
                    truncated = True
                    # break
                if not bag_is_safe:
                    bag_safe_set_cost += 10000
                    truncated = True
                    # break
                # These "stay away from the walls" costs are somewhat expensive to compute and don't necessarily need
                # to be done every timestep. TODO just use the local description of the safe set, not the full thing
                if i % steps_per_safe_set_eval == 0:
                    robot_safe_set_cost += safe_set_weight * safe_set_cost(
                        robot_bb[0], self.safe_set.values()
                    )
                    robot_safe_set_cost += safe_set_weight * safe_set_cost(
                        robot_bb[1], self.safe_set.values()
                    )
                    bag_safe_set_cost += safe_set_weight * safe_set_cost(
                        bag_bb[0], self.safe_set.values()
                    )
                    bag_safe_set_cost += safe_set_weight * safe_set_cost(
                        bag_bb[1], self.safe_set.values()
                    )
                # Penalizing bag velocities perpendicular to the robot's velocity during tracking
                if i % steps_per_bag_vel_eval == 0:
                    if self.flight_state == self.FlightStates.NOMINAL:
                        # TODO move the bag dynamics eval
                        bag_pos, bag_orn, bag_vel, bag_ang_vel = self.bag.dynamics_state
                        bag_vel_cost += 30 * np.linalg.norm(bag_vel) - np.dot(
                            lin_vel / np.linalg.norm(lin_vel), bag_vel
                        )
                    else:
                        bag_vel_cost += 0

            # End-of-rollout additional cost function evaluations
            # 1) Stabilize the motion of the bag with respect to the robot
            # 2) Position the robot so it's stopped at the goal pose
            # Both of these are only relevant when we're at the end of the nominal trajectory
            # TODO tune all of the scaling factors on the costs
            if self.flight_state == self.FlightStates.STOPPING:
                bag_pos, bag_orn, bag_vel, bag_ang_vel = self.bag.dynamics_state
                angular_term = np.linalg.norm(ang_vel - bag_ang_vel)
                r_r2b = bag_pos - pos  # Vector from robot to bag
                linear_term = np.linalg.norm(
                    lin_vel - bag_vel + np.cross(ang_vel, r_r2b)
                )
                stabilization_cost += 500 * (linear_term + angular_term)

                # TODO make this a separate function?
                # Adding back in a tracking cost component
                # If we are stopping then we know that the target state is the goal
                pos_error = np.linalg.norm(pos - self.target_pos)
                orn_error = quaternion_dist(orn, self.target_orn)
                vel_error = np.linalg.norm(lin_vel - self.target_vel)
                ang_vel_error = np.linalg.norm(ang_vel - self.target_omega)
                if self.is_debugging_simulation:
                    print("Position error: ", pos_error)
                    print("Orn error: ", orn_error)
                    print("Vel error: ", vel_error)
                    print("Ang vel error: ", ang_vel_error)
                tracking_cost += (
                    200 * pos_error
                    + 100 * orn_error
                    + 200 * vel_error
                    + 100 * ang_vel_error
                )
            else:
                stabilization_cost = 0
                tracking_cost = 0

            if self.is_debugging_simulation:
                print("Robot safe set cost: ", robot_safe_set_cost)
                print("Bag safe set cost: ", bag_safe_set_cost)
                print("Stabilization cost: ", stabilization_cost)
                print("Tracking cost: ", tracking_cost)
                print("Bag velocity cost: ", bag_vel_cost)
            reward = -1 * (
                robot_safe_set_cost
                + bag_safe_set_cost
                + stabilization_cost
                + tracking_cost
                + bag_vel_cost
            )

        # Observe the robot/bag state in the main env, dummy value if in rollout env
        observation = self._get_obs()

        # Evaluate if we have stabilized the robot and the bag at the end of the trajectory
        # (main env only since that's what we care about and we don't want to waste compute)
        if (
            self.flight_state == self.FlightStates.STOPPING
            and self.is_primary_simulation
            and robot_and_bag_termination_criteria(
                observation[0], observation[1], self.goal_pose
            )
        ):
            terminated = True

        # TODO: If we change the observation function to return the state of the robot and the bag,
        # we can determine the "terminated" parameter!
        # But note that we should only really do the observation in the main env

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

    def get_robot_state(self) -> tuple[np.ndarray, ...]:
        """Returns the full state information for the Astrobee in the environment (Base pos/orn/vels, joint angles/vels)

        Returns:
            tuple[np.ndarray, ...]:
                np.ndarray: Position, shape (3,)
                np.ndarray: Orientation (XYZW quaternion), shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
                np.ndarray: Joint positions, shape (NUM_JOINTS,)
                np.ndarray: Joint velocities, shape (NUM_JOINTS,)
        """
        return self.robot.full_state

    def get_bag_state(self) -> tuple[np.ndarray, ...]:
        """Returns the dynamics state information for the bag in the environment (pos/orn/vels)

        Returns:
            tuple[np.ndarray, ...]:
                np.ndarray: Position, shape (3,)
                np.ndarray: XYZW quaternion orientation, shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        return self.bag.dynamics_state

    def reset_robot_state(self, state: tuple[np.ndarray, ...]) -> None:
        """Fully resets the state of the Astrobee in the environment

        Args:
            state (tuple[np.ndarray, ...]): Full Astrobee state information containing:
                np.ndarray: Position, shape (3,)
                np.ndarray: Orientation (XYZW quaternion), shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
                np.ndarray: Joint positions, shape (NUM_JOINTS,)
                np.ndarray: Joint velocities, shape (NUM_JOINTS,)
        """
        assert len(state) == 6
        assert len(state[0]) == 3
        assert len(state[1]) == 4
        assert len(state[2]) == 3
        assert len(state[3]) == 3
        assert len(state[4]) == Astrobee.NUM_JOINTS
        assert len(state[5]) == Astrobee.NUM_JOINTS
        self.robot.reset_full_state(*state)

    def reset_bag_state(self, state: tuple[np.ndarray, ...]) -> None:
        """Resets the dynamics of the bag in the environment

        Args:
            state (tuple[np.ndarray, ...]): Dynamics info of the bag, containing:
                np.ndarray: Position, shape (3,)
                np.ndarray: Orientation (XYZW quaternion), shape (4,)
                np.ndarray: Linear velocity, shape (3,)
                np.ndarray: Angular velocity, shape (3,)
        """
        assert len(state) == 4
        assert len(state[0]) == 3
        assert len(state[1]) == 4
        assert len(state[2]) == 3
        assert len(state[3]) == 3
        self.bag.reset_dynamics(*state)

    def show_traj_plan(self, n: Optional[int]) -> None:
        """Displays the planned trajectory on the current pybullet client GUI (if enabled)

        Args:
            n (Optional[int]): Number of frames to plot, if plotting all of the frames is not desired.
                Defaults to None (plot all frames)
        """
        if self.traj_plan is None:
            raise ValueError("No trajectory available to visualize")
        self.debug_viz_ids = self.traj_plan.visualize(n, client=self.client)

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
