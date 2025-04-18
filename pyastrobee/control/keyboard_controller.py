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
"""

import time

import numpy as np
import pybullet
from pynput import keyboard

from pyastrobee.core.astrobee import Astrobee
from pyastrobee.utils.poses import (
    pos_euler_xyz_to_pos_quat,
    add_local_pose_delta,
    add_global_pose_delta,
)
from pyastrobee.core.iss import ISS
from pyastrobee.utils.debug_visualizer import get_viz_camera_params
from pyastrobee.utils.python_utils import print_green, print_red


class KeyboardController:
    """Class to encompass the keyboard listening and corresponding robot actions

    Args:
        robot (Astrobee): The Astrobee to control
        pov (bool, optional): Whether to use a "third person point of view" perspective that follows the
            robot around. Defaults to True.
    """

    def __init__(self, robot: Astrobee, pov: bool = True):
        if not pybullet.isConnected():
            raise ConnectionError(
                "Connect to pybullet before starting the keyboard controller"
            )
        # Disable keyboard shortcuts in pybullet so it doesn't do anything weird while we're controlling it
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)

        self.robot = robot
        self.pov = pov
        self.gripper_is_open = robot.gripper_position >= 90
        # Copying over the original constraint from the Astrobee class (and the position controller)
        # TODO remove this when we switch over to velocity control for this controller
        self.constraint_id = pybullet.createConstraint(
            self.robot.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0)
        )

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

        # Timestep
        self.dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]

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
        for key, delta in self.pos_euler_deltas_lowercase.items():
            self.pose_deltas[keyboard.KeyCode.from_char(key)] = (
                pos_euler_xyz_to_pos_quat(delta)
            )
            self.pose_deltas[keyboard.KeyCode.from_char(key.upper())] = (
                pos_euler_xyz_to_pos_quat(self.mult * delta)
            )

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
            + "Capital letters: Coarse control of position/orientation\n"
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
                    self.constraint_id, new_pose[:3], new_pose[3:]
                )
            else:
                new_pose = add_global_pose_delta(init_pose, pose_delta)
                pybullet.changeConstraint(
                    self.constraint_id, new_pose[:3], new_pose[3:]
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
        if self.pov:
            # Update the camera view so we maintain our same perspective on the robot as it moves
            pybullet.resetDebugVisualizerCamera(*get_viz_camera_params(self.robot.tmat))
        time.sleep(self.dt)

    def run(self):
        """Runs the simulation loop with the keyboard listener active"""
        if not self.is_listening:
            self.start_listening()
        try:
            while True:
                self.step()
        finally:
            self.listener.stop()


def _main():
    pybullet.connect(pybullet.GUI)
    # Turn off additional GUI windows
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
    iss = ISS(debug=False)
    robot = Astrobee()
    robot.store_arm(force=True)
    controller = KeyboardController(robot, pov=True)
    controller.run()


if __name__ == "__main__":
    _main()
