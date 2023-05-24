"""Model predictive control"""

import time
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

from pyastrobee.elements.astrobee import Astrobee
from pyastrobee.elements.cargo_bag import CargoBag
from pyastrobee.elements.iss import load_iss
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.quaternions import quaternion_angular_error
from pyastrobee.control.polynomial_trajectories import (
    polynomial_trajectory,
    polynomial_traj_with_velocity_bcs,
)
from pyastrobee.utils.math_utils import spherical_vonmises_sampling
from pyastrobee.control.trajectory import Trajectory


def reset_state(
    state_id: int,
    bag_id: int,
    bag_pos: npt.ArrayLike,
    bag_orn: npt.ArrayLike,
    bag_lin_vel: npt.ArrayLike,
    bag_ang_vel: npt.ArrayLike,
) -> None:
    """Resets the state of the simulation (including the state of the cargo bag)

    Args:
        state_id (int): Pybullet ID for the saveState object
        bag_id (int): Pybullet ID for the cargo bag
        bag_pos (npt.ArrayLike): Position of the cargo bag, shape (3,)
        bag_orn (npt.ArrayLike): Orientation of the cargo bag (XYZW quaternion), shape (4,)
        bag_lin_vel (npt.ArrayLike): Linear velocity of the cargo bag, shape (3,)
        bag_ang_vel (npt.ArrayLike): Angular velocity of the cargo bag, shape (3,)
    """
    pybullet.restoreState(stateId=state_id)
    pybullet.resetBasePositionAndOrientation(bag_id, bag_pos, bag_orn)
    pybullet.resetBaseVelocity(bag_id, bag_lin_vel, bag_ang_vel)


def save_state(
    bag: CargoBag,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Save the state of the simulation, including the state of the bag

    - Normally, you could just call saveState, but this does not save the state of deformable objects
    - saveState will save info about the Astrobee and the environment, but we need to manually save information
      about the current dyanmics of the bag

    Args:
        bag (CargoBag): The cargo bag in the current simulation

    Returns:
        tuple of:
            int: The Pybullet ID of the saveState object
            np.ndarray: Position of the cargo bag, shape (3,)
            np.ndarray: Orientation of the cargo bag (XYZW quaternion), shape (4,)
            np.ndarray: Linear velocity of the cargo bag, shape (3,)
            np.ndarray: Angular velocity of the cargo bag, shape (3,)
    """
    state_id = pybullet.saveState()
    bag_pos, bag_orn, bag_vel, bag_ang_vel = bag.dynamics_state
    return state_id, bag_pos, bag_orn, bag_vel, bag_ang_vel


# see simple_init
def init(robot_pose, use_gui: bool = True):
    client = initialize_pybullet(use_gui)
    load_iss()
    robot = Astrobee(robot_pose)
    bag = CargoBag("top_handle_bag", robot)
    return client


def deviation_penalty(
    cur_pos: npt.ArrayLike,
    cur_orn: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    cur_ang_vel: npt.ArrayLike,
    des_pos: npt.ArrayLike,
    des_orn: npt.ArrayLike,
    des_vel: npt.ArrayLike,
    des_ang_vel: npt.ArrayLike,
    l_pos: float,
    l_orn: float,
    l_vel: float,
    l_ang_vel: float,
) -> float:
    pos_err = np.subtract(cur_pos, des_pos)
    orn_err = quaternion_angular_error(cur_orn, des_orn)
    vel_err = np.subtract(cur_vel, des_vel)
    ang_vel_err = np.subtract(cur_ang_vel, des_ang_vel)
    # NOTE Would an L1 norm be better here?
    return (
        l_pos * np.linalg.norm(pos_err)
        + l_orn * np.linalg.norm(orn_err)
        + l_vel * np.linalg.norm(vel_err)
        + l_ang_vel * np.linalg.norm(ang_vel_err)
    )


def generate_trajs(
    cur_pos: npt.ArrayLike,
    cur_orn: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    cur_ang_vel: npt.ArrayLike,
    nominal_target_pos: npt.ArrayLike,
    nominal_target_orn: npt.ArrayLike,
    nominal_target_vel: npt.ArrayLike,
    nominal_target_ang_vel: npt.ArrayLike,
    pos_sampling_stdev: float,
    orn_sampling_stdev: float,
    vel_sampling_stdev: float,
    ang_vel_sampling_stdev: float,
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

    # Sample endpoints for the candidate trajectories about the nominal targets
    sampled_positions = np.random.multivariate_normal(
        nominal_target_pos, pos_sampling_stdev**2 * np.eye(3), n_trajs
    )
    sampled_quats = spherical_vonmises_sampling(
        nominal_target_orn, 1 / (orn_sampling_stdev**2), n_trajs
    )
    sampled_vels = np.random.multivariate_normal(
        nominal_target_vel, vel_sampling_stdev**2 * np.eye(3), n_trajs
    )
    sampled_ang_vels = np.random.multivariate_normal(
        nominal_target_ang_vel, ang_vel_sampling_stdev**2 * np.eye(3), n_trajs
    )

    # Generate trajectories from the sampled endpoints
    trajs = []
    for i in range(n_trajs):
        trajs.append(
            polynomial_traj_with_velocity_bcs(
                cur_pos,
                cur_orn,
                cur_vel,
                cur_ang_vel,
                sampled_positions[i],
                sampled_quats[i],
                sampled_vels[i],
                sampled_ang_vels[i],
                n_steps * dt,
                dt,
            )
        )

    return trajs


def mpc_main():
    # call init to start up the sim
    # loop over steps:
    # break the loop at a stopping criteria
    # (for us, stopping criteria is probably if the bag is "stopped" at the desired
    # location and the astrobee is also "stopped", so we can disconnect the bag from astrobee)
    # TODO check on what perfect model rollouts means
    # TODO check on the (step % MPC_STEPS)?
    #
    # ah i see
    # ok main loop is the overall control loop like what I've made before
    # The inner MPC if statement / loop runs the sim a bunch of times to figure out what the
    # best action to take is, then spits out the best move
    # the main loop then continues from this point knowing the best move, and executes it
    #
    # NOTE rika uses a "visualize traj deviation" function which might be useful to implement
    # We should consider defining the trajectory of the bag rather than the astrobee though)
    #
    # the loop continues until the stopping condition which breaks

    # Based on how I set up the generate trajs function, we'll need one baseline trajectory to use as a reference
    # for sampling purposes
    # I am assuming this is a trajectory for the bag
    # But perhaps this should be a trajectory for the gripper? Or the gripper mapped back to the robot base?
    pass


# See perfect_model_rollout
# TODO is there ever an imperfect rollout? I suppoose there could be... implement this later
# TODO figure out how to use the sim parameter effectively here so we can get some parallel envs going
def rollout():
    # This seems to be effectively the same thing as stepping through the main control loop
    # (but, with a limited horizon)
    # The implementation for this should look pretty similar to some of the step() functions
    # I've written in some of the other controllers
    pass


def visualize_deviation():
    # It might be nice to modify the visuzlize_frame function to work here
    # Could visualize a frame normally for the desired pose
    # and then maybe a different-colored frame for the current frame
    # TODO check if we can add a dotted line? That would be easiest
    pass


if __name__ == "__main__":
    pass
