"""Test to see if our state space models track the ground truth error with minimal linearization error

Right now it seems that using the "true" inertia tensor (the one based on the actual configuration of the robot,
including the arm positioning) does reduce the tracking error for the angular velocity

However, the angular velocity error does still seem a little high... Maybe this is acceptable given good values for
the other parameters
"""


import pybullet
import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.trajectories.planner import plan_trajectory
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.utils.plotting import num_subplots_to_shape


def test_state_space(use_sim_inertial_props: bool = False):
    """Evaluate a trajectory and see how well the state-space model matches the true state derivatives

    Args:
        use_sim_inertial_props (bool, optional): Whether to recompute the mass/inertia properties based on the
            current Astrobee state in simulation. Defaults to False (use NASA's constant values)
    """
    np.random.seed(0)
    client = initialize_pybullet()
    robot = Astrobee(client=client)
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]

    if use_sim_inertial_props:
        robot.recompute_inertial_properties()

    kp, kv, kq, kw = 20, 5, 1, 0.1
    controller = ForceTorqueController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt
    )
    pos, orn, vel, omega = robot.dynamics_state
    traj = plan_trajectory(
        pos,
        orn,
        vel,
        omega,
        np.zeros(3),
        np.zeros(3),
        [1, 1, 1],
        random_quaternion(),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        5,
        dt,
    )
    state_dim = 13
    control_dim = 6
    x_log = np.zeros((traj.num_timesteps, state_dim))
    u_log = np.zeros((traj.num_timesteps, control_dim))
    x_dot_log = np.zeros((traj.num_timesteps, state_dim))
    # TODO: try try making the controls just based on the acceleration, rather than the CL error?

    for i in range(traj.num_timesteps):
        x = robot.state_vector
        x_log[i] = x
        pos = x[:3]
        vel = x[3:6]
        orn = x[6:10]
        omega = x[10:]
        A, B = robot.state_space_matrices
        F = controller.get_force(
            pos,
            vel,
            traj.positions[i],
            traj.linear_velocities[i],
            traj.linear_accels[i],
        )
        T = controller.get_torque(
            orn,
            omega,
            traj.quaternions[i],
            traj.angular_velocities[i],
            traj.angular_accels[i],
        )
        client.applyExternalForce(robot.id, -1, F, list(pos), pybullet.WORLD_FRAME)
        client.applyExternalTorque(robot.id, -1, list(T), pybullet.WORLD_FRAME)
        client.stepSimulation()
        u = np.concatenate((F, T))
        u_log[i] = u
        x_dot_log[i] = A @ x + B @ u

    client.disconnect()
    eval_state_space(x_log, x_dot_log, dt)


def eval_state_space(x_log, x_dot_log, dt):
    # Compare if the state space model for the derivative actually matches up with what we'd expect
    # TODO: clean this up a bit and improve the plot partitioning

    x_dot_true = np.gradient(x_log, dt, axis=0)
    times = np.arange(x_log.shape[0]) * dt
    state_dim = 13
    subplot_shape = num_subplots_to_shape(state_dim)
    # fmt: off
    labels = ["dx", "dy", "dz", "dvx", "dvy", "dvz", "dqx", "dqy", "dqz", "dqw", "dwx", "dwy", "dwz"]
    # fmt: on
    for i in range(state_dim):
        plt.subplot(*subplot_shape, i + 1)
        plt.plot(times, x_dot_log[:, i], "b", label="ss")
        plt.plot(times, x_dot_true[:, i], "r", label="true")
        plt.title(labels[i])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_state_space(use_sim_inertial_props=True)
