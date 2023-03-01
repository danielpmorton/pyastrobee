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
- Toggle opening/closing the gripper with G
- Coarse control with capital letters
- Press Space to switch reference frames (global <=> robot)
- Press Esc to exit

Motions are only recorded on a button release to prevent over-queueing actions

NOTE
- Should this incorporate any of the motion planning things? Or just change the constraint directly?
- World frame control may not be very useful. Should we toggle between gripper frame and robot frame?

TODO
- Add raise/lower arm control
- Multiprocessing!!! Change the hacky way I'm currently stepping through the simulation
- Add force control, velocity control, and a toggle between methods
- Change documentation back to capital letters for coarse control
- Add back finer control of the gripper rather than just an open/close toggle?
- Add back sshkeyboard?
- Velocity / force control
- Angle snapping?
"""

import time
import pprint

import numpy as np
import pybullet
from pynput import keyboard

from pyastrobee.control.astrobee import Astrobee
from pyastrobee.utils.poses import (
    pos_euler_xyz_to_pos_quat,
    add_local_pose_delta,
    add_global_pose_delta,
)


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

        # Configuration for non-position/angular control keys
        self.frame_switch_key = keyboard.Key.space
        self.exit_key = keyboard.Key.esc
        # Allow for either lowercase or capital G to control the gripper
        self.gripper_keys = {
            keyboard.KeyCode.from_char("g"),
            keyboard.KeyCode.from_char("G"),
        }

        # Coarse control multiplier
        self.mult = 5

        # Current state of the frame being controlled
        self.in_robot_frame = False

        self.commands = {
            "w/s": "+/- x",
            "a/d": "+/- y",
            "e/q": "+/- z",
            "l/j": "+/- roll",
            "i/k": "+/- pitch",
            "u/o": "+/- yaw",
            "Capital letters": "Coarse control",
            "Space": "Switch reference frames",
            "Esc": "Stop listening",
        }

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

    def start_listening(self):
        """Starts the keyboard listener"""
        print("Now listening. Commands:")
        pprint.pprint(self.commands)
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            listener.join()

    def on_press(self, key: keyboard.Key):
        """Callback for a keypress

        Args:
            key (Key): Key that was pressed

        Returns:
            bool: False when the "stop listening" key (Esc) is pressed
        """
        if key == keyboard.Key.esc:
            return False  # Stop the listener

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
            # TODO CHANGE THE WAY THIS LOOP IS HANDLED
            # (Will probably need some multiprocessing)
            for _ in range(10):
                pybullet.stepSimulation()
                time.sleep(1 / 120)
        elif key == self.frame_switch_key:
            # Toggle the reference frame and update our knowledge of the state
            self.in_robot_frame = not self.in_robot_frame
            print(
                f"FRAME SWITCHED. Now in {'robot' if self.in_robot_frame else 'world'} frame"
            )
        elif key in self.gripper_keys:  # g or G
            # Toggle the gripper and update our knowledge of the state
            if self.gripper_is_open:
                self.robot.close_gripper()
                self.gripper_is_open = False
            else:
                self.robot.open_gripper()
                self.gripper_is_open = True
            print(f"GRIPPER {'OPENED' if self.gripper_is_open else 'CLOSED'}")


if __name__ == "__main__":
    pybullet.connect(pybullet.GUI)
    bee = Astrobee()
    controller = KeyboardController(bee)
    # This start_listening() command is blocking, so unless we kill the listener,
    # nothing after this will run
    controller.start_listening()
    # while pybullet.isConnected():
    #     pybullet.stepSimulation()
    #     time.sleep(1 / 120)  # UPDATE THIS
