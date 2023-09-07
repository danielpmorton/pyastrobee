"""Example of just tracking the standard trajectory we're working with in the MPC file, without MPC

This naive method will result in collisions with the sides of the ISS, particularly within the narrow corridor
"""

from pyastrobee.core.environments import AstrobeeEnv
from pyastrobee.trajectories.planner import global_planner
from pyastrobee.control.force_torque_control import ForceTorqueController

# Enable/disable this flag depending on if we want to record or not
RECORD_VIDEO = True
VIDEO_LOCATION = "artifacts/naive_approach.mp4"


def main():
    start_pose = [0, 0, 0, 0, 0, 0, 1]
    end_pose = [6, 0, 0.2, 0, 0, 0, 1]  # Easy-to-reach location in JPM
    env = AstrobeeEnv(True, start_pose)
    dt = env.client.getPhysicsEngineParameters()["fixedTimeStep"]
    nominal_traj = global_planner(
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
        client=env.client,
    )
    if RECORD_VIDEO:
        log_id = env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4, VIDEO_LOCATION
        )
    nominal_traj.visualize(20, client=env.client)
    controller.follow_traj(nominal_traj)
    if RECORD_VIDEO:
        env.client.stopStateLogging(log_id)
    env.client.disconnect()


if __name__ == "__main__":
    main()
