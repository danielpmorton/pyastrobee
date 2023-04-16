"""Translation of gnc/ctl/src/ctl.cc to python

WORK IN PROGRESS

https://github.com/nasa/astrobee/blob/master/gnc/ctl/src/ctl.cc
"""

import numpy as np
from scipy import linalg

from pyastrobee.utils.math_utils import normalize, safe_divide
from pyastrobee.utils.rotations import rotate_point, quat_to_rmat, quaternion_dist


class ControlOutput:
    def __init__(self):
        self.body_force_cmd = np.zeros(3)
        self.body_torque_cmd = np.zeros(3)
        self.body_accel_cmd = np.zeros(3)
        self.body_alpha_cmd = np.zeros(3)
        self.pos_err = np.zeros(3)
        self.pos_err_int = np.zeros(3)
        self.att_err = np.zeros(3)
        self.att_err_int = np.zeros(3)
        self.traj_pos = np.zeros(3)
        self.traj_quat = np.array([0, 0, 0, 1])
        self.traj_vel = np.zeros(3)
        self.traj_accel = np.zeros(3)
        self.traj_omega = np.zeros(3)
        self.traj_alpha = np.zeros(3)
        self.att_err_mag = 0.0
        self.traj_erros_pos = 0.0
        self.traj_error_att = 0.0
        self.traj_error_vel = 0.0
        self.traj_error_omega = 0.0
        self.ctl_status = 0


class ControlCommand:
    def __init__(self):
        self.P_B_ISS_ISS = np.zeros(3)
        self.quat_ISS2B = np.array([0, 0, 0, 1])
        self.V_B_ISS_ISS = np.zeros(3)
        self.A_B_ISS_ISS = np.zeros(3)
        self.omega_B_ISS_ISS = np.zeros(3)
        self.alpha_B_ISS_ISS = np.zeros(3)
        self.mode = 0


class ControlState:
    def __init__(self):
        self.est_P_B_ISS_ISS = np.zeros(3)
        self.est_quat_ISS2B = np.array([0, 0, 0, 1])
        self.est_V_B_ISS_ISS = np.zeros(3)
        self.est_omega_B_ISS_B = np.zeros(3)
        self.inertia = np.zeros((3, 3))
        self.att_kp = np.zeros(3)
        self.att_ki = np.zeros(3)
        self.omega_kd = np.zeros(3)
        self.pos_kp = np.zeros(3)
        self.pos_ki = np.zeros(3)
        self.vel_kd = np.zeros(3)
        self.mass = 0.0
        self.est_confidence = 0


class Control:
    def __init__(self):
        # See Control::Initialize and Control::ReadParams

        self.out = ControlOutput()

    def step(self, dt, state, cmd):
        self.forward_trajectory(dt, state, cmd)
        self.update_mode(state, cmd)
        self.update_ctl_status(state)
        self.find_pos_error(state, cmd)
        self.find_body_force_cmd(state)
        self.find_att_error(state)
        self.find_body_alpha_torque_cmd(state)
        self.update_previous(state)

    def omega_matrix(self, input):
        return np.array(
            [
                [0, input[2], -input[1], input[0]],
                [-input[2], 0, input[0], input[1]],
                [input[1], -input[0], 0, input[2]],
                [-input[0], -input[1], -input[2], 0],
            ]
        )

    def forward_trajectory(self, dt, state: ControlState, cmd: ControlCommand):
        # Forward integrate
        self.out.traj_pos = (
            cmd.P_B_ISS_ISS + cmd.V_B_ISS_ISS * dt + 0.5 * cmd.A_B_ISS_ISS * dt**2
        )
        self.out.traj_vel = cmd.V_B_ISS_ISS + cmd.A_B_ISS_ISS * dt
        self.out.traj_accel = cmd.A_B_ISS_ISS
        self.out.traj_alpha = cmd.alpha_B_ISS_ISS
        self.out.traj_omega = cmd.omega_B_ISS_ISS + cmd.alpha_B_ISS_ISS * dt

        omega_omega = self.omega_matrix(cmd.omega_B_ISS_ISS)
        omega_alpha = self.omega_matrix(cmd.alpha_B_ISS_ISS)

        a = 0.5 * dt * (0.5 * dt * omega_alpha + omega_omega)
        a = linalg.expm(a)
        a += (
            (1 / 48) * dt**3 * (omega_alpha * omega_omega - omega_omega * omega_alpha)
        )
        self.out.traj_quat = normalize(a * cmd.quat_ISS2B)
        self.out.traj_erros_pos = np.linalg.norm(
            self.out.traj_pos - state.est_P_B_ISS_ISS
        )
        self.out.traj_error_vel = np.linalg.norm(
            self.out.traj_vel - state.est_V_B_ISS_ISS
        )
        self.out.traj_error_omega = np.linalg.norm(
            self.out.traj_omega - state.est_omega_B_ISS_B
        )
        self.out.traj_error_att = self.quat_error(
            self.out.traj_quat, state.est_quat_ISS2B
        )

    def find_att_error(self, state):
        pass

    def find_body_alpha_torque_cmd(self, state):
        pass

    def find_body_force_cmd(self, state):
        pass

    def saturate_vector(self, v, limit):
        pass

    def discrete_time_integrator(
        self, input, accumulator, ctl_status, upper_limit, lower_limit
    ):
        pass

    def find_pos_error(self, state, cmd):
        pass

    def update_ctl_status(self, state):
        pass

    def update_previous(self, state):
        pass

    def quat_error(self, cmd, actual):
        pass

    def butterworth_filter(self, input, delay):
        pass

    def filter_threshold(self, vec, threshold, prev):
        pass

    def update_mode(self, state, cmd):
        pass
