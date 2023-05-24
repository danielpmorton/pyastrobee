# This file is a bit of a mess because I started off trying this and then realized it would be
# easier to just work with polynomial basis functions for the trajectory components

# However, this might be of use for the future if we decide to use convex optimization for trajectory generation
# (the constraints and overall setup will be different, but the structure will be similar)

# Ideas
# Add a binary search to figure out what duration will lead to satisfying the max vel/accel constraints?
# Or, there should actually be an analytical solution for the maximums of the associated functions for velocity/accel
# so we can just choose the duration such that this is satisfied

# BUG
# The interpolation used here doesn't really make sense with Euler angles due to the wraparound at 2*pi
# Polynomial SLERP will be a better option but IDK if we can form that in a convex manner

# Also note that right now, this might be a fully constrained linear system, so there should just be a simple
# analytical solution via a linear matrix system

import cvxpy as cp
import numpy as np
import pybullet
import matplotlib.pyplot as plt
from pyastrobee.utils.rotations import (
    quat_to_fixed_xyz,
    fixed_xyz_to_rmat,
    fixed_xyz_to_quat,
)
from pyastrobee.utils.transformations import make_transform_mat
from pyastrobee.utils.poses import batched_pos_quats_to_tmats
from pyastrobee.trajectories.trajectory import visualize_traj


class TrajFinder:
    def __init__(self, start_pose, end_pose, tf, dt):
        self.a = cp.Variable(4)
        self.b = cp.Variable(4)
        self.c = cp.Variable(4)
        self.d = cp.Variable(4)
        self.e = cp.Variable(4)
        self.f = cp.Variable(4)
        # self.T = cp.Variable(1)
        self.tf = tf  # Can't optimize this at the same time
        self.dt = dt
        self.times = np.arange(0, tf + dt, dt)
        self.n_timesteps = len(self.times)
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.roll_0, self.pitch_0, self.yaw_0 = quat_to_fixed_xyz(self.start_pose[3:])
        self.roll_f, self.pitch_f, self.yaw_f = quat_to_fixed_xyz(self.end_pose[3:])

    def tau(self, t):
        n = 1 if np.ndim(t) == 0 else len(t)
        return np.row_stack([np.ones(n), t, t**2, t**3])

    def dtau(self, t):
        n = 1 if np.ndim(t) == 0 else len(t)
        return np.row_stack([np.zeros(n), np.ones(n), 2 * t, 3 * t**2])

    def ddtau(self, t):
        n = 1 if np.ndim(t) == 0 else len(t)
        return np.row_stack([np.zeros((2, n)), 2 * np.ones(n), 6 * t])

    def dddtau(self, t):
        n = 1 if np.ndim(t) == 0 else len(t)
        return np.row_stack([np.zeros((3, n)), 6 * np.ones(n)])

    def x(self, t, use_opt=False):
        if use_opt:
            return self.a.value @ self.tau(t)
        return self.a.T @ self.tau(t)

    def y(self, t, use_opt=False):
        if use_opt:
            return self.b.value @ self.tau(t)
        return self.b.T @ self.tau(t)

    def z(self, t, use_opt=False):
        if use_opt:
            return self.c.value @ self.tau(t)
        return self.c.T @ self.tau(t)

    def roll(self, t, use_opt=False):
        if use_opt:
            return self.d.value @ self.tau(t)
        return self.d.T @ self.tau(t)

    def pitch(self, t, use_opt=False):
        if use_opt:
            return self.e.value @ self.tau(t)
        return self.e.T @ self.tau(t)

    def yaw(self, t, use_opt=False):
        if use_opt:
            return self.f.value @ self.tau(t)
        return self.f.T @ self.tau(t)

    def v_x(self, t, use_opt=False):
        if use_opt:
            return self.a.value @ self.dtau(t)
        return self.a.T @ self.dtau(t)

    def v_y(self, t, use_opt=False):
        if use_opt:
            return self.b.value @ self.dtau(t)
        return self.b.T @ self.dtau(t)

    def v_z(self, t, use_opt=False):
        if use_opt:
            return self.c.value @ self.dtau(t)
        return self.c.T @ self.dtau(t)

    def v_roll(self, t, use_opt=False):
        if use_opt:
            return self.d.value @ self.dtau(t)
        return self.d.T @ self.dtau(t)

    def v_pitch(self, t, use_opt=False):
        if use_opt:
            return self.e.value @ self.dtau(t)
        return self.e.T @ self.dtau(t)

    def v_yaw(self, t, use_opt=False):
        if use_opt:
            return self.f.value @ self.dtau(t)
        return self.f.T @ self.dtau(t)

    def a_x(self, t, use_opt=False):
        if use_opt:
            return self.a.value @ self.dtau(t)
        return self.a.T @ self.ddtau(t)

    def a_y(self, t, use_opt=False):
        if use_opt:
            return self.b.value @ self.dtau(t)
        return self.b.T @ self.ddtau(t)

    def a_z(self, t, use_opt=False):
        if use_opt:
            return self.c.value @ self.dtau(t)
        return self.c.T @ self.ddtau(t)

    def a_roll(self, t, use_opt=False):
        if use_opt:
            return self.d.value @ self.dtau(t)
        return self.d.T @ self.ddtau(t)

    def a_pitch(self, t, use_opt=False):
        if use_opt:
            return self.e.value @ self.dtau(t)
        return self.e.T @ self.ddtau(t)

    def a_yaw(self, t, use_opt=False):
        if use_opt:
            return self.f.value @ self.dtau(t)
        return self.f.T @ self.ddtau(t)

    def jerk(self):
        # g = cp.vstack([self.a, self.b, self.c, self.d, self.e, self.f])
        jmat = self.dddtau(self.times)
        return cp.sum(
            self.a.T @ jmat
            + self.b.T @ jmat
            + self.c.T @ jmat
            + self.d.T @ jmat
            + self.e.T @ jmat
            + self.f.T @ jmat
        )

    @property
    def constraints(self):
        return [
            self.x(0) == self.start_pose[0],
            self.y(0) == self.start_pose[1],
            self.z(0) == self.start_pose[2],
            self.roll(0) == self.roll_0,
            self.pitch(0) == self.pitch_0,
            self.yaw(0) == self.yaw_0,
            self.v_x(0) == 0,
            self.v_y(0) == 0,
            self.v_z(0) == 0,
            self.v_roll(0) == 0,
            self.v_pitch(0) == 0,
            self.v_yaw(0) == 0,
            # THE FINAL-TIME CONSTRAINTS ARE NONCONVEX if the final time is a variable as well
            self.x(self.tf) == self.end_pose[0],
            self.y(self.tf) == self.end_pose[1],
            self.z(self.tf) == self.end_pose[2],
            self.roll(self.tf) == self.roll_f,
            self.pitch(self.tf) == self.pitch_f,
            self.yaw(self.tf) == self.yaw_f,
            self.v_x(self.tf) == 0,
            self.v_y(self.tf) == 0,
            self.v_z(self.tf) == 0,
            self.v_roll(self.tf) == 0,
            self.v_pitch(self.tf) == 0,
            self.v_yaw(self.tf) == 0,
            # TODO need to add constraints linking position and velocity
            # For instance stating that x(t+1) = x(t) + v(t) * dt + (1/2)
            # I think this is necessary to give non-trivial answers
            # Actually hmm this might overconstrain the problem if it's based on polynomials
            # I think right now we are fully constrained
            # Otherwise, we could remove the polynomial structure and then enforce the manual integration
            # But, it is easy to exploit the derivative structure of the polys
        ]

    @property
    def objective(self):
        return cp.Minimize(self.jerk())
        # return cp.Minimize(0)

    def solve(self):
        prob = cp.Problem(self.objective, self.constraints)
        prob.solve()
        self.a_opt = self.a.value
        self.b_opt = self.b.value
        self.c_opt = self.c.value
        self.d_opt = self.d.value
        self.e_opt = self.e.value
        self.f_opt = self.f.value
        print("Done")

    def traj_from_polys(self):
        xs = self.x(self.times, True)
        ys = self.y(self.times, True)
        zs = self.z(self.times, True)
        rolls = self.roll(self.times, True)
        pitches = self.pitch(self.times, True)
        yaws = self.yaw(self.times, True)
        pqs = np.zeros((self.n_timesteps, 7))
        for i in range(self.n_timesteps):
            q = fixed_xyz_to_quat([rolls[i], pitches[i], yaws[i]])
            pqs[i, :] = np.array([xs[i], ys[i], zs[i], *q])
        return pqs

    def visualize_polys(self):
        xs = self.x(self.times, True)
        ys = self.y(self.times, True)
        zs = self.z(self.times, True)
        rolls = self.roll(self.times, True)
        pitches = self.pitch(self.times, True)
        yaws = self.yaw(self.times, True)
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.plot(self.times, xs)
        plt.title("X")
        plt.subplot(2, 3, 2)
        plt.plot(self.times, ys)
        plt.title("Y")
        plt.subplot(2, 3, 3)
        plt.plot(self.times, xs)
        plt.title("Z")
        plt.subplot(2, 3, 4)
        plt.plot(self.times, xs)
        plt.title("Roll")
        plt.subplot(2, 3, 5)
        plt.plot(self.times, xs)
        plt.title("Pitch")
        plt.subplot(2, 3, 6)
        plt.plot(self.times, xs)
        plt.title("Yaw")
        plt.show()


# def traj_between(pose1, pose2):
#     x0, y0, z0, qx0, qy0, qz0, qw0 = pose1
#     xf, yf, zf, qxf, qyf, qzf, qwf = pose2
#     roll0, pitch0, yaw0 = quat_to_fixed_xyz(pose1[3:])
#     rollf, pitchf, yawf = quat_to_fixed_xyz(pose2[3:])

if __name__ == "__main__":
    solver = TrajFinder([0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0.707, 0, 0.707], 5, 0.1)
    solver.solve()
    traj = solver.traj_from_polys()
    solver.visualize_polys()
    pybullet.connect(pybullet.GUI)
    visualize_traj(traj)
    input()
