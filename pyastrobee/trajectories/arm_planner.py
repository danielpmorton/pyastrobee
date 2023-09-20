"""Planning (relatively simple) trajectories for the arm during the manipulation motion"""

# Note: Some of the methods here are separated out into separate functions so we can test them individually

import numpy as np
from pyastrobee.trajectories.polynomials import fifth_order_poly
from pyastrobee.trajectories.trajectory import Trajectory, ArmTrajectory


def plan_arm_traj(base_traj: Trajectory) -> ArmTrajectory:
    """Plans a motion for the arm so that we are dragging the bag behind the robot for most of the trajectory, but we
    start and end from the grasping position

    This will only use the shoulder joint -- other joints are less relevant for the motion of the bag/robot system

    This is essentially comprised of three parts:
    1) Moving the arm from an initial grasp position to the "dragging behind" position
    2) Maintaining the "dragging behind" position for most of the trajectory
    3) Moving the arm to the grasp position again at the end of the trajectory

    Args:
        base_traj (Trajectory): Trajectory of the Astrobee base

    Returns:
        ArmTrajectory: Positional trajectory for the arm (shoulder joint only)
    """
    times, transition_idxs = _get_arm_transition_times(base_traj)
    return _plan_arm_traj(times, *transition_idxs)


def _get_arm_transition_times(base_traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Determine the transition times for the bag manipulation arm motions

    This will assume that we start in the "grasp" position, transition to "drag" for most of the trajectory,
    and then return to the "grasp" position when we're at the end

    These transition times will depend on how fast we're moving with the base - in essence, we want these
    arm motions to complete once the base travels a certain distance - so that the motion of the base and the
    arm work together and don't lead to excessive disturbances

    Args:
        base_traj (Trajectory): Trajectory for the Astrobee base

    Returns:
        tuple[np.ndarray, np.ndarray]:
            np.ndarray: Time information for the full trajectory
            np.ndarray: Transition times for each motion: "begin_drag_motion", "end_drag_motion",
                "begin_grasp_motion", and "end_grasp_motion"

    """
    dists_from_start = np.linalg.norm(
        base_traj.positions - base_traj.positions[0], axis=1
    )
    dists_from_end = np.linalg.norm(
        base_traj.positions - base_traj.positions[-1], axis=1
    )
    # A little over 1 meter seems to work well for the deployment distance (TODO probably needs tuning)
    # For reference, the 0.23 number is how much the x position of the arm changes when it moves back
    dx_arm = 0.23162640743995172 * 5
    # Note: originally I used searchsorted to find these indices, but these distances are not guaranteed to be sorted
    # if the trajectory has a lot of curvature
    drag_idx = np.flatnonzero(dists_from_start >= dx_arm)[0]
    grasp_idx = np.flatnonzero(dists_from_end <= dx_arm)[0]
    return (
        base_traj.times,
        base_traj.times[np.array([0, drag_idx, grasp_idx, -1])],
    )


def _plan_arm_traj(
    times: np.ndarray,
    drag_start_time: float,
    drag_end_time: float,
    grasp_start_time: float,
    grasp_end_time: float,
) -> ArmTrajectory:
    """Helper function for plan_arm_traj to compute the trajectory based on times transition indices

    Args:
        times (np.ndarray): Trajectory time information, shape (n_timesteps,)
        drag_start_time (float): Time to start the arm dragging motion (usually 0)
        drag_end_time (float): Time to end the arm dragging motion
        grasp_start_time (float): Time to start the arm grasping motion
        grasp_end_time (float): Time to end the arm grasping motion (usually the final time)

    Returns:
        ArmTrajectory: Positional trajectory for the arm (shoulder joint only)
    """
    # Shoulder joint parameters for Astrobee
    grasp_angle = 0
    drag_angle = -1.57079
    shoulder_index = 1

    drag_start_id = np.searchsorted(times, drag_start_time)
    drag_end_id = np.searchsorted(times, drag_end_time)
    grasp_start_id = np.searchsorted(times, grasp_start_time)
    grasp_end_id = np.searchsorted(times, grasp_end_time)
    # Edge case: in case the grasp end time is greater than the last time
    grasp_end_id = min(len(times) - 1, grasp_end_id)

    # Generate polynomials to define the motions within their respective time periods
    drag_poly = fifth_order_poly(
        drag_start_time, drag_end_time, grasp_angle, drag_angle, 0, 0, 0, 0
    )
    grasp_poly = fifth_order_poly(
        grasp_start_time, grasp_end_time, drag_angle, grasp_angle, 0, 0, 0, 0
    )
    arm_motion = np.ones_like(times) * drag_angle
    arm_motion[drag_start_id : drag_end_id + 1] = drag_poly(
        times[drag_start_id : drag_end_id + 1]
    )
    arm_motion[grasp_start_id : grasp_end_id + 1] = grasp_poly(
        times[grasp_start_id : grasp_end_id + 1]
    )
    # key_times = _get_arm_transition_times(base_traj)
    # Edge case for short trajectories: not enough time to fully move arm there and back again
    overlap = drag_end_id > grasp_start_id
    if overlap:
        # Blend the two motions in the overlapping region
        overlap_poly = (
            (drag_poly - drag_angle) + (grasp_poly - drag_angle)
        ) + drag_angle
        arm_motion[grasp_start_id : drag_end_id + 1] = np.clip(
            overlap_poly(times[grasp_start_id : drag_end_id + 1]),
            drag_angle,
            grasp_angle,
        )
    key_times = {
        "begin_drag_motion": drag_start_time,
        "end_drag_motion": drag_end_time,
        "begin_grasp_motion": grasp_start_time,
        "end_grasp_motion": grasp_end_time,
    }
    return ArmTrajectory(arm_motion, [shoulder_index], times, key_times)


def _main():
    # Quick test/example to make sure that the polynomials blend properly at the transition times
    times = np.linspace(0, 10, 1000)
    traj = _plan_arm_traj(times, 0, 3, 7, 10)
    traj.plot()


if __name__ == "__main__":
    _main()
