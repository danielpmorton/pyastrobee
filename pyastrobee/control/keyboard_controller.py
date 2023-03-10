"""Class/script for controlling the Astrobee via keyboard presses

Usage:
- Positional control via WASDQE
    - w: +x
    - s: -x
    - a: +y
    - d: -y
    - e: +z
    - q: -z
- Angular control via IJKLUO
    - l: + roll
    - j: - roll
    - i: + pitch
    - k: - pitch
    - u: + yaw
    - o: - yaw
- Arm control via arrow keys
    - Up: + proximal angle
    - Down: - proximal angle
    - Right: + distal angle
    - Left: - distal angle
- Toggle opening/closing the gripper with G
- Record a waypoint with P
- Coarse control of position/orientation with capital letters (via Shift)
- Coarse control of the arm positioning via C
- Press Space to switch reference frames (global <=> robot)
- Press Esc to exit

Motions are only recorded on a button release to prevent over-queueing actions

NOTE
- Should this incorporate any of the motion planning things? Or just change the constraint directly?
- World frame control may not be very useful. Should we toggle between gripper frame and robot frame?

TODO
- Unify the coarse control toggling between the position and the arm control
- Add force control, velocity control, and a toggle between methods
- Add back finer control of the gripper rather than just an open/close toggle?
- Add back sshkeyboard? (This likely won't work with threading?)
- Velocity / force control
- Angle snapping?
- Make it possible to control the debug viz camera independently? (This may be unnecessary)
- Allow for switching between certain fixed camera positions 
"""

import time

import numpy as np
import pybullet
from pynput import keyboard

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.poses import (
    pos_euler_xyz_to_pos_quat,
    add_local_pose_delta,
    add_global_pose_delta,
)
from pyastrobee.utils.iss_utils import load_iss
from pyastrobee.vision.debug_visualizer import get_viz_camera_params
from pyastrobee.utils.python_utils import print_green, print_red


class KeyboardController:
    """Class to encompass the keyboard listening and corresponding robot actions

    TODO add linear and angular speeds to the inputs?

    Args:
        robot (Astrobee): The Astrobee to control
    """

    def __init__(
        self,
        robot: Astrobee,
    ):
        # Disable keyboard shortcuts in pybullet so it doesn't do anything weird while we're controlling it
        if pybullet.isConnected():
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)

        self.robot = robot
        self.gripper_is_open = robot.gripper_position >= 90

        # self.linear_speed (todo?)
        # self.angular_speed
        self.dx = 0.05  # m
        self.dy = 0.05  # m
        self.dz = 0.05  # m
        self.droll = 0.05  # rad
        self.dpitch = 0.05  # rad
        self.dyaw = 0.05  # rad
        self.djoint = 0.05  # rad

        # Configuration for non-position/angular control keys
        self.frame_switch_key = keyboard.Key.space
        self.exit_key = keyboard.Key.esc
        self.gripper_key = keyboard.KeyCode.from_char("g")
        self.waypoint_key = keyboard.KeyCode.from_char("p")
        self.coarse_control_key = keyboard.KeyCode.from_char("c")
        self.arm_keys = {
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.right,
            keyboard.Key.left,
        }

        # Coarse control multiplier
        self.mult = 5

        # States
        self.in_robot_frame = False
        self.in_coarse_mode = False

        # Initialize listener, update when listener started
        self.listener: keyboard.Listener = None

        # Having this in position/Euler can be more interpretable than position/quaternion
        # Also, it is easier to scale Euler angles than quaternions for coarse control
        # (multiplying a quaternion by a constant won't change the rotation it represents)
        self.pos_euler_deltas_lowercase = {
            "w": np.array([self.dx, 0, 0, 0, 0, 0]),
            "s": np.array([-self.dx, 0, 0, 0, 0, 0]),
            "a": np.array([0, self.dy, 0, 0, 0, 0]),
            "d": np.array([0, -self.dy, 0, 0, 0, 0]),
            "e": np.array([0, 0, self.dz, 0, 0, 0]),
            "q": np.array([0, 0, -self.dz, 0, 0, 0]),
            "l": np.array([0, 0, 0, self.droll, 0, 0]),
            "j": np.array([0, 0, 0, -self.droll, 0, 0]),
            "i": np.array([0, 0, 0, 0, self.dpitch, 0]),
            "k": np.array([0, 0, 0, 0, -self.dpitch, 0]),
            "u": np.array([0, 0, 0, 0, 0, self.dyaw]),
            "o": np.array([0, 0, 0, 0, 0, -self.dyaw]),
        }
        # Change the keys to be in pynput Key format, and add coarse control for uppercase letters
        self.pose_deltas = {}
        for (key, delta) in self.pos_euler_deltas_lowercase.items():
            self.pose_deltas[
                keyboard.KeyCode.from_char(key)
            ] = pos_euler_xyz_to_pos_quat(delta)
            self.pose_deltas[
                keyboard.KeyCode.from_char(key.upper())
            ] = pos_euler_xyz_to_pos_quat(self.mult * delta)

    def _print_commands(self):
        """Prints the keyboard/control action mapping information"""
        print(
            "Commands:\n"
            + "w/s: +/- x\n"
            + "a/d: +/- y\n"
            + "e/q: +/- z\n"
            + "l/j: +/- roll\n"
            + "i/k: +/- pitch\n"
            + "u/o: +/- yaw\n"
            + "up/down: +/- arm proximal joint angle\n"
            + "right/left: +/- arm distal joint angle\n"
            + "c: Toggle coarse control for the arm\n"
            + "g: Open/close the gripper\n"
            + "p: Record a pose\n"
            + "Space: Switch reference frames\n"
            + "Esc: Stop listening"
        )

    def start_listening(self):
        """Starts the keyboard listener in a new thread"""
        print_green("Now listening")
        self._print_commands()
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    @property
    def is_listening(self) -> bool:
        """Whether or not the pynput Listener has been initialized"""
        return self.listener is not None

    def on_press(self, key: keyboard.Key):
        """Callback for a keypress

        Args:
            key (Key): Key that was pressed

        Returns:
            bool: False when the "stop listening" key (Esc) is pressed
        """
        if key == keyboard.Key.esc:
            print_red("\nLISTENER STOPPED")
            print("Simulation will remain active until killed")
            return False  # Returning false from a callback stops pynput

    def on_release(self, key: keyboard.Key):
        """Callback for when a key is released

        Args:
            key (Key): Key that was pressed
        """
        # If the key is associated with a motion,
        if key in self.pose_deltas:
            init_pose = self.robot.pose
            # Get the pose delta that key is associated with
            pose_delta = self.pose_deltas[key]
            if self.in_robot_frame:
                new_pose = add_local_pose_delta(init_pose, pose_delta)
                pybullet.changeConstraint(
                    self.robot.constraint_id, new_pose[:3], new_pose[3:]
                )
            else:
                new_pose = add_global_pose_delta(init_pose, pose_delta)
                pybullet.changeConstraint(
                    self.robot.constraint_id, new_pose[:3], new_pose[3:]
                )
        elif key == self.frame_switch_key:
            # Toggle the reference frame and update our knowledge of the state
            self.in_robot_frame = not self.in_robot_frame
            print(
                f"\nFRAME SWITCHED. Now in {'robot' if self.in_robot_frame else 'world'} frame"
            )
        elif key == self.gripper_key:
            # Toggle the gripper and update our knowledge of the state
            if self.gripper_is_open:
                self.robot.close_gripper()
                self.gripper_is_open = False
            else:
                self.robot.open_gripper()
                self.gripper_is_open = True
            print(f"\nGRIPPER {'OPENED' if self.gripper_is_open else 'CLOSED'}")
        elif key == self.waypoint_key:
            # Record a waypoint and print the info to the terminal
            print(f"\nPose recorded: {self.robot.pose}")
            print(f"Gripper position: {self.robot.gripper_position}")
            print(f"Arm joints: {self.robot.arm_joint_angles}")
        elif key == self.coarse_control_key:
            # Toggle coarse control mode
            self.in_coarse_mode = not self.in_coarse_mode
            print(
                f"\nCoarse control {'ACTIVATED' if self.in_coarse_mode else 'DEACTIVATED'}"
            )
        elif key in self.arm_keys:
            cur_prox, cur_dist = self.robot.arm_joint_angles
            prox_idx, dist_idx = self.robot.ARM_JOINT_IDXS
            delta = self.djoint * (self.mult if self.in_coarse_mode else 1)
            if key == keyboard.Key.up:
                # Raise the proximal joint
                self.robot.set_joint_angles(
                    min(cur_prox + delta, Astrobee.JOINT_POS_LIMITS[prox_idx, 1]),
                    prox_idx,
                )
            elif key == keyboard.Key.down:
                # Lower the proximal joint
                self.robot.set_joint_angles(
                    max(cur_prox - delta, Astrobee.JOINT_POS_LIMITS[prox_idx, 0]),
                    prox_idx,
                )
            elif key == keyboard.Key.right:
                # Raise the distal joint
                self.robot.set_joint_angles(
                    min(cur_dist + delta, Astrobee.JOINT_POS_LIMITS[dist_idx, 1]),
                    dist_idx,
                )
            elif key == keyboard.Key.left:
                # Lower the distal joint
                self.robot.set_joint_angles(
                    max(cur_dist - delta, Astrobee.JOINT_POS_LIMITS[dist_idx, 0]),
                    dist_idx,
                )

    def step(self):
        """Updates one step of the simulation"""
        pybullet.stepSimulation()
        # Update the camera view so we maintain our same perspective on the robot as it moves
        pybullet.resetDebugVisualizerCamera(*get_viz_camera_params(self.robot.tmat))
        time.sleep(1 / 120)  # TODO make this a parameter?

    def run(self):
        """Runs the simulation loop with the keyboard listener active"""
        if not self.is_listening:
            self.start_listening()
        try:
            while True:
                self.step()
        finally:
            self.listener.stop()


if __name__ == "__main__":
    pybullet.connect(pybullet.GUI)
    # pybullet.resetDebugVisualizerCamera(1.6, 206, -26.2, [0, 0, 0]) # Camera in node 1
    # Loading the ISS and then the astrobee at the origin is totally fine right now (collision free, inside node 1)
    load_iss()
    bee = Astrobee()
    controller = KeyboardController(bee)
    controller.run()
