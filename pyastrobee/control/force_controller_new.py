"""

TODO integrate this with the PID class I made
TODO update the way mass and inertia are handled (attributes of Astrobee?)
TODO see if we can bring back the matrix forms of these gains
TODO add stopping tolerances as inputs?
TODO unify variable naming
"""
import time
from typing import Optional

import pybullet
import numpy as np
import numpy.typing as npt

from pyastrobee.control.trajectory import Trajectory, stopping_criteria
from pyastrobee.utils.quaternions import quaternion_angular_diff


class ForcePIDController:
    def __init__(self, robot_id, mass, inertia, kv, kp, kw, kq, dt, log: Trajectory):
        self.id = robot_id
        self.mass = mass
        self.inertia = inertia
        self.kv = kv
        self.kp = kp
        self.kw = kw
        self.kq = kq
        self.dt = dt
        self.log = log

    def get_force(
        self,
        cur_pos: npt.ArrayLike,
        cur_vel: npt.ArrayLike,
        des_pos: npt.ArrayLike,
        des_vel: npt.ArrayLike,
        des_accel: npt.ArrayLike,
    ) -> np.ndarray:
        M = self.mass * np.eye(3)
        pos_err = np.subtract(cur_pos, des_pos)
        vel_err = np.subtract(cur_vel, des_vel)
        return M @ np.asarray(des_accel) - self.kv * vel_err - self.kp * pos_err

    def get_torque(
        self,
        cur_q: npt.ArrayLike,
        cur_w: npt.ArrayLike,
        des_q: npt.ArrayLike,
        des_w: npt.ArrayLike,
        des_a: npt.ArrayLike,
    ) -> np.ndarray:
        ang_err = quaternion_angular_diff(des_q, cur_q)
        ang_vel_err = cur_w - des_w
        return self.inertia @ des_a - self.kw * ang_vel_err - self.kq * ang_err

    def follow_traj(self, traj: Trajectory):
        for i in range(traj.num_timesteps):
            pos, orn, lin_vel, ang_vel = self.get_current_state()
            self.step(
                pos,
                lin_vel,
                orn,
                ang_vel,
                traj.positions[i, :],
                traj.linear_velocities[i, :],
                traj.linear_accels[i, :],
                traj.quaternions[i, :],
                traj.angular_velocities[i, :],
                traj.angular_accels[i, :],
            )
        self.stop(traj.positions[-1, :], traj.quaternions[-1, :])

    def stop(self, des_pos, des_quat):
        des_vel = np.zeros(3)
        des_accel = np.zeros(3)
        des_omega = np.zeros(3)
        des_alpha = np.zeros(3)
        while True:
            pos, orn, lin_vel, ang_vel = self.get_current_state()
            if stopping_criteria(pos, orn, lin_vel, ang_vel, des_pos, des_quat):
                return
            self.step(
                pos,
                lin_vel,
                orn,
                ang_vel,
                des_pos,
                des_vel,
                des_accel,
                des_quat,
                des_omega,
                des_alpha,
            )

    def step(
        self,
        pos,
        vel,
        orn,
        omega,
        des_pos,
        des_vel,
        des_accel,
        des_orn,
        des_omega,
        des_alpha,
    ):
        force = self.get_force(pos, vel, des_pos, des_vel, des_accel)
        torque = self.get_torque(orn, omega, des_orn, des_omega, des_alpha)
        # TODO: explain -1? And does pos need to be a list?
        pybullet.applyExternalForce(self.id, -1, force, list(pos), pybullet.WORLD_FRAME)
        pybullet.applyExternalTorque(self.id, -1, list(torque), pybullet.WORLD_FRAME)
        pybullet.stepSimulation()
        time.sleep(self.dt)

    def get_current_state(self):
        pos, quat = pybullet.getBasePositionAndOrientation(self.id)
        lin_vel, ang_vel = pybullet.getBaseVelocity(self.id)
        if self.log is not None:
            self.log.log_state(pos, quat, lin_vel, ang_vel, self.dt)
        return pos, quat, lin_vel, ang_vel


if __name__ == "__main__":
    # MOVE THIS. Just want to see what the code would look like
    from pyastrobee.utils.bullet_utils import create_box
    from pyastrobee.utils.quaternions import random_quaternion
    from pyastrobee.control.polynomial_trajectories import polynomial_trajectory
    from pyastrobee.control.trajectory import visualize_traj, compare_trajs

    def box_inertia(m, l, w, h):
        return (
            (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])
        )

    tracker = Trajectory()
    pybullet.connect(pybullet.GUI)
    np.random.seed(0)
    pose_1 = [0, 0, 0, 0, 0, 0, 1]
    pose_2 = [1, 2, 3, *random_quaternion()]
    mass = 10
    sidelengths = [0.25, 0.25, 0.25]
    box = create_box(pose_1[:3], pose_1[3:], mass, sidelengths, True)
    max_time = 10
    dt = 0.01
    traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
    visualize_traj(traj, 20)
    # mass * dt seems to give a general trend of how the required gains change depending on mass/time
    # However, it seems like this shouldn't depend on dt? Perhaps it's an artifact of doing discrete simulation steps
    kp = 1000 * mass * dt
    kv = 100 * mass * dt
    kq = 10 * mass * dt
    kw = 1 * mass * dt
    base_idx = -1  # Base link index of the robot
    inertia = box_inertia(mass, *sidelengths)
    controller = ForcePIDController(box, mass, inertia, kv, kp, kw, kq, dt, tracker)
    controller.follow_traj(traj)
    pybullet.disconnect()
    compare_trajs(traj, tracker)
