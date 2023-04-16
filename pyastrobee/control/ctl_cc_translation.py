"""Translation of gnc/ctl/src/ctl.cc to python

WORK IN PROGRESS

https://github.com/nasa/astrobee/blob/master/gnc/ctl/src/ctl.cc
"""

import numpy as np
from scipy import linalg

from pyastrobee.utils.math_utils import normalize, safe_divide
from pyastrobee.utils.rotations import rotate_point, quat_to_rmat, quaternion_dist
from pyastrobee.utils.quaternion import conjugate


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
        self.mode = ""
        self.prev_att = None

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

    def find_att_error(self, state: ControlState):
        q_cmd = self.out.traj_quat
        if self.mode == "stopped":
            q_cmd = self.prev_att
        q_out = conjugate(state.est_quat_ISS2B) * q_cmd
        q_out = normalize(q_out)
        self.out.att_err_mag = 2 * np.arccos(q_out[3])  # qw
        self.out.att_err = q_out[:3]  # qx, qy, qz
        Ki_rot = safe_divide(state.att_ki, state.omega_kd)
        in_ = self.out.att_err * Ki_rot
        self.out.att_err_int = self.discrete_time_integrator(
            in_,
            self.rotational_integrator,
            self.out.ctl_status,
            tun_ctl_att_sat_upper,
            tun_ctl_att_sat_lower,
        )

    def find_body_alpha_torque_cmd(self, state: ControlState):
        cmd_omega = self.out.traj_omega
        cmd_alpha = self.out.traj_alpha
        if self.mode == "stopped":
            cmd_omega = np.zeros(3)
            cmd_alpha = np.zeros(3)
        rate_error = -1 * state.est_omega_B_ISS_B
        if self.out.ctl_status > 1:
            Kp_rot = safe_divide(state.att_kp, state.omega_kd)
            rate_error = (
                cmd_omega
                + self.out.att_err_int
                + (Kp_rot * self.out.att_err)
                - state.est_omega_B_ISS_B
            )
        Kd_rot = state.omega_kd * np.diag(state.inertia)
        rate_error = rate_error * Kd_rot
        self.out.body_alpha_cmd = np.linalg.inverse(state.inertia) * rate_error
        if self.out.ctl_status == 0:
            self.out.body_torque_cmd = np.zeros(3)
        else:
            cmd_alpha = turn_alpha_gain * cmd_alpha
            self.out.body_torque_cmd = (
                state.inertia * cmd_alpha
                + rate_error
                - np.cross(
                    state.inertia * state.est_omega_B_ISS_B, state.est_omega_B_ISS_B
                )
            )

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
