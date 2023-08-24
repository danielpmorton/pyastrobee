"""Example of planning a trajectory through the ISS and enforcing all constraints"""

import numpy as np

from pyastrobee.core.iss import ISS
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.iss_safe_boxes import ALL_BOXES
from pyastrobee.utils.debug_visualizer import visualize_path, animate_path
from pyastrobee.trajectories.planner import global_planner
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.trajectories.trajectory import plot_traj_constraints, plot_traj
from pyastrobee.config.astrobee_motion import (
    LINEAR_SPEED_LIMIT,
    ANGULAR_SPEED_LIMIT,
    LINEAR_ACCEL_LIMIT,
    ANGULAR_ACCEL_LIMIT,
)


def main():
    """Generate trajectories between the same two points using Tobia's method and mine"""

    np.random.seed(0)
    p0 = ALL_BOXES["jpm"].center
    pf = ALL_BOXES["cupola"].center
    q0 = np.array([0, 0, 0, 1])
    qf = random_quaternion()
    dt = 0.1  # Just for testing purposes
    traj = global_planner(p0, q0, pf, qf, dt)

    print("Trajectory planned. Showing plots")
    print("Close the plots to bring up the Pybullet visualization")
    plot_traj(traj, show=False)
    # TODO it would be nice for the constraint plotting to take in the full list of boxes
    plot_traj_constraints(
        traj,
        None,
        LINEAR_SPEED_LIMIT,
        LINEAR_ACCEL_LIMIT,
        ANGULAR_SPEED_LIMIT,
        ANGULAR_ACCEL_LIMIT,
        show=True,
    )
    input("Press Enter to visualize the trajectory in Pybullet")
    client = initialize_pybullet()
    iss = ISS()
    iss.show_safe_set()
    visualize_path(traj.positions, 50, color=(0, 0, 1))
    traj.visualize(20)
    input("Press Enter to animate the path")
    animate_path(traj.positions, 5, 500)
    input("Press Enter to finish")
    client.disconnect()


if __name__ == "__main__":
    main()
