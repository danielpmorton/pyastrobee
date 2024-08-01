"""Script to show the Astrobee tracking a trajectory plan through the interior of the ISS

This was made so that we could record an face-forward visualization of this motion. It's a bit
hacky, but it worked well enough to get the video we needed.
"""
# TODO switch out all of the planning stuff with the unified function once this is done

import time

import pybullet
import numpy as np

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.trajectories.face_forward import face_forward_traj
from pyastrobee.utils.boxes import find_containing_box
from pyastrobee.config.iss_safe_boxes import ROBOT_SAFE_SET, compute_iss_graph
from pyastrobee.utils.algos import dfs
from pyastrobee.trajectories.splines import spline_trajectory_with_retiming
from pyastrobee.utils.rotations import rmat_to_quat, Rz
from pyastrobee.utils.debug_visualizer import visualize_path, get_viz_camera_params
from pyastrobee.trajectories.trajectory import stopping_criteria
from pyastrobee.core.iss import ISS
from pyastrobee.trajectories.planner import global_planner


def plan_path(p0, pf, T, n_timesteps):
    names = []
    boxes = []
    for name, box in ROBOT_SAFE_SET.items():
        names.append(name)
        boxes.append(box)
    start_id = find_containing_box(p0, boxes)
    end_id = find_containing_box(pf, boxes)
    graph = compute_iss_graph()
    path = dfs(graph, names[start_id], names[end_id])
    box_path = [ROBOT_SAFE_SET[p] for p in path]  # TODO improve this
    init_durations = T / len(box_path) * np.ones(len(box_path))
    curve, cost = spline_trajectory_with_retiming(
        p0,
        pf,
        0,
        T,
        8,
        box_path,
        init_durations,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )
    return curve(np.linspace(0, T, n_timesteps))


def face_forward_test(record_video: bool = False):
    p0 = ROBOT_SAFE_SET["jpm"].center
    # pf = ROBOT_SAFE_SET["cupola"].center
    # There seems to be something weird going on with my relationship between quaternions
    # and headings since it does an odd spin if we move to the cupola... (TODO debug)

    pf = ROBOT_SAFE_SET["cupola"].center
    T = 30
    q0 = rmat_to_quat(Rz(-np.pi))
    pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
    pybullet.configureDebugVisualizer(rgbBackground=(0, 0, 0))
    iss = ISS()
    # dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    dt = 1 / 10
    robot = Astrobee([*p0, *q0])
    robot.store_arm(force=True)
    kp, kv, kq, kw = 20, 10, 5, 5
    controller = ForceTorqueController(
        robot.id, robot.mass, robot.inertia, kp, kv, kq, kw, dt
    )
    n_timesteps = round(T / dt)
    traj = global_planner(p0, (0, 0, 0, 1), pf, (0, 0, 0, 1), dt)
    if record_video:
        log_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4, "artifacts/iss_traj_new.mp4"
        )
    visualize_path(traj.positions, 30)
    ff_traj = face_forward_traj(traj.positions, q0, np.zeros(3), np.zeros(3), dt)

    # follow_traj_with_sleep(traj, controller, True, 0)
    follow_traj_forced(ff_traj, robot)
    if record_video:
        pybullet.stopStateLogging(log_id)


def follow_traj_forced(traj, robot):
    for i in range(traj.num_timesteps):
        pybullet.resetBasePositionAndOrientation(
            robot.id, traj.positions[i], traj.quaternions[i]
        )
        pybullet.resetDebugVisualizerCamera(*get_viz_camera_params(robot.tmat))
        time.sleep(1 / 120)


def follow_traj_with_sleep(traj, controller, stop_at_end, sleep_dt):
    for i in range(traj.num_timesteps):
        pos, orn, lin_vel, ang_vel = controller.get_current_state()
        controller.step(
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
        time.sleep(sleep_dt)
    if stop_at_end:
        max_iters = None
        des_pos = traj.positions[-1]
        des_quat = traj.quaternions[-1]
        des_vel = np.zeros(3)
        des_accel = np.zeros(3)
        des_omega = np.zeros(3)
        des_alpha = np.zeros(3)
        iters = 0
        while True:
            pos, orn, lin_vel, ang_vel = controller.get_current_state()
            if stopping_criteria(
                pos,
                orn,
                lin_vel,
                ang_vel,
                des_pos,
                des_quat,
                controller.pos_tol,
                controller.orn_tol,
                controller.vel_tol,
                controller.ang_vel_tol,
            ):
                return
            if max_iters is not None and iters >= max_iters:
                print("Maximum iterations reached, stopping unsuccessful")
                return
            controller.step(
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
            iters += 1
            time.sleep(sleep_dt)


if __name__ == "__main__":
    face_forward_test(record_video=False)
