"""Test to see if our linearized dynamics models track the ground truth state with minimal error

Right now it seems that using the "true" inertia tensor (the one based on the actual configuration of the robot,
including the arm positioning) does reduce the tracking error for the angular velocity

However, the angular velocity error does still seem a little high... Maybe this is acceptable given good values for
the other parameters
"""
import time
import pybullet
import numpy as np
import matplotlib.pyplot as plt

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.trajectories.planner import local_planner
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet


def test_state_space(use_sim_inertial_props: bool = False):
    """Evaluate a trajectory and see how well the state-space model matches the true state derivatives

    Args:
        use_sim_inertial_props (bool, optional): Whether to recompute the mass/inertia properties based on the
            current Astrobee state in simulation. Defaults to False (use NASA's constant values)
    """
    # TODO: Does making the control open-loop (just based on the traj acceleration values) significantly
    # change this comparison?

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
    start_time = time.time()
    traj = local_planner(
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
    print("Local planner time: ", time.time() - start_time)
    state_dim = 13
    control_dim = 6
    x_log = np.zeros((traj.num_timesteps, state_dim))
    u_log = np.zeros((traj.num_timesteps, control_dim))
    dx_model_log = np.zeros((traj.num_timesteps, state_dim))
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
        dx_model_log[i] = A @ x + B @ u

    client.disconnect()

    dx_true = np.gradient(x_log, dt, axis=0)
    plot_dynamics_comparison(dx_true, dx_model_log, dt)


def _plot(
    axes: np.ndarray[plt.Axes],
    true_data: np.ndarray,
    ss_data: np.ndarray,
    titles: list[str],
    x_axis: np.ndarray,
    *args,
    **kwargs,
):
    """Plotting helper function"""
    for i, ax in enumerate(axes):
        ax.plot(x_axis, true_data[:, i], "k--", label="True", *args, **kwargs)
        ax.plot(x_axis, ss_data[:, i], "r", label="S.S.", *args, **kwargs)
        ax.set_title(titles[i])


def plot_dynamics_comparison(dx_true: np.ndarray, dx_linearized: np.ndarray, dt: float):
    """Plot a comparison of the true state derivatives to what we predict based on the linearized dynamics model

    Args:
        dx_true (np.ndarray): True derivative of the robot state data, shape (n_timesteps, state_dim)
        dx_linearized (np.ndarray): Derivatives predicted by the linearized dynamics, shape (n_timesteps, state_dim)
        dt (float): Timestep
    """
    times = np.arange(dx_true.shape[0]) * dt
    fig = plt.figure()
    subfigs = fig.subfigures(4, 1)
    pos_fig = subfigs[0].subplots(1, 3)
    orn_fig = subfigs[1].subplots(1, 4)
    vel_fig = subfigs[2].subplots(1, 3)
    omega_fig = subfigs[3].subplots(1, 3)

    # fmt: off
    _plot(pos_fig, dx_true[:, :3], dx_linearized[:, :3], ["dx", "dy", "dz"], times)
    _plot(orn_fig, dx_true[:, 3:7], dx_linearized[:, 3:7], ["dqx", "dqy", "dqz", "dqw"], times)
    _plot(vel_fig, dx_true[:, 7:10], dx_linearized[:, 7:10], ["dvx", "dvy", "dvz"], times)
    _plot(omega_fig, dx_true[:, 10:], dx_linearized[:, 10:], ["dwx", "dwy", "dwz"], times)
    # fmt: on

    # Put the legend in roughly the correct location outside the plots
    pos_fig[-1].legend(bbox_to_anchor=(1.1, 0.5), loc="center left")
    plt.show()


if __name__ == "__main__":
    test_state_space(use_sim_inertial_props=True)
