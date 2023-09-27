"""Example of just tracking the standard trajectory we're working with in the MPC file, without MPC

This naive method will result in collisions with the sides of the ISS, particularly within the narrow corridor
"""

from pathlib import Path
from datetime import datetime

from pyastrobee.core.environments import AstrobeeEnv
from pyastrobee.trajectories.planner import global_planner
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.config.astrobee_motion import MAX_FORCE_MAGNITUDE, MAX_TORQUE_MAGNITUDE

# Enable/disable this flag depending on if we want to record or not
RECORD_VIDEO = True
VIDEO_LOCATION = (
    f"artifacts/{Path(__file__).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
)
NODE_2_VIEW = (1.40, -69.60, -19.00, (0.55, 0.00, -0.39))
JPM_VIEW = (1.00, 64.40, -12.20, (6.44, -0.39, 0.07))


def main():
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    env = AstrobeeEnv(True, start_pose)
    dt = env.client.getPhysicsEngineParameters()["fixedTimeStep"]
    traj = global_planner(
        start_pose[:3],
        start_pose[3:],
        end_pose[:3],
        end_pose[3:],
        dt,
    )
    kp, kv, kq, kw = 20, 5, 1, 0.1
    controller = ForceTorqueController(
        env.robot.id,
        env.robot.mass,
        env.robot.inertia,
        kp,
        kv,
        kq,
        kw,
        dt,
        max_force=MAX_FORCE_MAGNITUDE,
        max_torque=MAX_TORQUE_MAGNITUDE,
        client=env.client,
    )
    if RECORD_VIDEO:
        env.client.resetDebugVisualizerCamera(*NODE_2_VIEW)
        log_id = env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4, VIDEO_LOCATION
        )
    camera_moved = False
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
        if RECORD_VIDEO and not camera_moved and pos[0] >= 2.5:
            env.client.resetDebugVisualizerCamera(*JPM_VIEW)
            camera_moved = True
    try:
        while True:
            pos, orn, lin_vel, ang_vel = controller.get_current_state()
            controller.step(
                pos,
                lin_vel,
                orn,
                ang_vel,
                traj.positions[-1, :],
                (0, 0, 0),
                (0, 0, 0),
                traj.quaternions[-1, :],
                (0, 0, 0),
                (0, 0, 0),
            )
    finally:
        if RECORD_VIDEO:
            env.client.stopStateLogging(log_id)
        env.client.disconnect()


if __name__ == "__main__":
    main()
