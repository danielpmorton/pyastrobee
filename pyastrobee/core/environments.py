"""Gym and vectorized environments for the Astrobee/ISS/Cargo setup"""


import pybullet
import numpy as np
import numpy.typing as npt

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv

from pyastrobee.control.force_controller_new import ForcePIDController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.core.iss import load_iss
from pyastrobee.core.cargo_bag import CargoBag
from pyastrobee.trajectories.planner import plan_trajectory


class AstrobeeEnv(gym.Env):
    def __init__(self, use_gui: bool = True):
        # TODO: make more of these parameters variables to pass in
        # e.g. initial bag/robot position, number of robots, type of bag, ...
        self.client_id = initialize_pybullet(use_gui)
        self.iss_ids = load_iss()
        self.robot = Astrobee()
        self.bag = CargoBag("top_handle")
        self.bag.attach_to(self.robot, object_to_move="bag")
        self.dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
        # TODO figure out how to handle controller parameters
        # Just fixing the gains here for now
        kp, kv, kq, kw = 20, 5, 1, 0.1  # TODO make parameters
        self.controller = ForcePIDController(
            self.robot.id, self.robot.mass, self.robot.inertia, kp, kv, kq, kw, self.dt
        )
        self.traj_plan = None  # Init
        self._traj_idx = 0  # Init
        self.target_pos = None  # Init
        self.target_orn = None  # Init
        self.target_vel = None  # Init
        self.target_omega = None  # Init
        # Dummy parameters for gym/stable baselines compatibility
        self.observation_space = None
        self.action_space = None

    def set_target_state(self, des_pos, des_orn, des_vel, des_omega) -> None:
        self.target_pos = des_pos
        self.target_orn = des_orn
        self.target_vel = des_vel
        self.target_omega = des_omega
        # Any time we update a goal state we'll need to do a replan
        pos, orn, vel, omega = self.robot.dynamics_state
        # TODO figure out if any of these values should be nonzero
        a0 = np.zeros(3)
        dw0 = np.zeros(3)
        af = np.zeros(3)
        dwf = np.zeros(3)
        duration = 10  # TODO UPDATE THIS!! MAKE HEURISTIC ESTIMATE
        self.traj_plan = plan_trajectory(
            pos,
            orn,
            vel,
            omega,
            a0,
            dw0,
            des_pos,
            des_orn,
            des_vel,
            des_omega,
            af,
            dwf,
            duration,
            self.dt,
        )

    def reset(self):
        # Decide if this should take in a saved state ID
        # Check with the stable baselines / gym env classes to see if this is possible
        pass

    def step(self):
        # The controller calls stepSimulation.. Determine if there is a better way to handle this
        # if there are more processes that should step
        # Use noisy localization??
        # TODO check on this "end of trajectory" indexing...
        if self.traj_plan is None or self._traj_idx == self.traj_plan.num_timesteps:
            pybullet.stepSimulation()
        else:
            pos, orn, vel, omega = self.robot.dynamics_state
            self.controller.step(
                pos,
                vel,
                orn,
                omega,
                self.traj_plan.positions[self._traj_idx],
                self.traj_plan.linear_velocities[self._traj_idx],
                self.traj_plan.linear_accels[self._traj_idx],
                self.traj_plan.quaternions[self._traj_idx],
                self.traj_plan.angular_velocities[self._traj_idx],
                self.traj_plan.angular_accels[self._traj_idx],
            )


def test_vec_env():
    # TODO check on the difference between subproc and dummy
    n_envs = 2
    env_kwargs = {"use_gui": False}
    vec_env = make_vec_env(
        AstrobeeEnv,
        n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
    )
    input("completed")


def _main():
    env = AstrobeeEnv()
    while True:
        env.step()


if __name__ == "__main__":
    # _main()
    test_vec_env()
