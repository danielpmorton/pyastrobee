"""Translation of gnc/ctl/src/ctl.cc to python

This is a bit messy and is NOT meant to be run directly
Rather, it's a starting point for our own controller logic

See: https://github.com/nasa/astrobee/blob/master/gnc/ctl/src/ctl.cc


NOTE
- Their safe divide defaults to setting the result as 0 if div/0
"""

import numpy as np
from scipy import linalg

from pyastrobee.utils.math_utils import normalize, safe_divide
from pyastrobee.utils.rotations import rotate_point, quat_to_rmat
from pyastrobee.utils.quaternion import conjugate, quaternion_dist


# Stores:
# Forces, torques, accel, angular accel
# Errors in position + attitude
# Trajectory information (position, orientation, velocities, accelerations)
# Errors in the trajectory (How does this differ from the pos error?)
# Integrator info?
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


# Position, orientation, velocity, acceleration, angular vel/accel
# And a "mode"
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
    # Keep track of:
    # The actual output of the controller
    # Mode
    # Previous values of the filter
    # Multiple timesteps of determining if we should switch to "stop" mode?
    # Previous position/attitude
    # Integrators (linear and angular)
    def __init__(self):
        self.out = ControlOutput()
        self.mode = ""
        self.stopped_mode = False
        self.prev_filter_vel = np.zeros(3)
        self.prev_filter_omega = np.zeros(3)
        self.prev_mode_cmd = np.zeros(4)
        self.prev_att = np.zeros(4)
        self.prev_position = np.zeros(3)
        self.linear_integrator = np.zeros(3)
        self.rotational_integrator = np.zeros(3)

    # Main control sequence!
    def step(self, dt, state, cmd):
        self.forward_trajectory(dt, state, cmd)
        self.update_mode(state, cmd)
        self.update_ctl_status(state)
        self.find_pos_error(state, cmd)
        self.find_body_force_cmd(state)
        self.find_att_error(state)
        self.find_body_alpha_torque_cmd(state)
        self.update_previous(state)

    # I think this is the matrix relating quaternions and angular velocity
    def omega_matrix(self, input):
        return np.array(
            [
                [0, input[2], -input[1], input[0]],
                [-input[2], 0, input[0], input[1]],
                [input[1], -input[0], 0, input[2]],
                [-input[0], -input[1], -input[2], 0],
            ]
        )

    # They seem to get a sense of where they are along the trajectory based on the
    # commanded values of the position/vel/accel
    # But, it seems that this info is not actually used???
    # Actually never mind, the traj_vel, accel, alpha, omega are used
    # But, the errors do not seem to be used
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

    # finds the attitude error between the current attitude estimate of the astrobee
    # and the commanded quaternion from the trajectory
    # And, updates the integrator information
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

    # Finds the torque command based on the values from the trajectory and the
    # current estimate of the attitude, and the integrator values
    # There is some interesting math going on here so it would be nice to have a source
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
        self.out.body_alpha_cmd = np.linalg.inv(state.inertia) * rate_error
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

    # Find the force to apply to the robot based on the targeted velocity/accel
    # The parts relating to the orientation seem to be doing a conversion from
    # the world frame to the body frame
    def find_body_force_cmd(self, state: ControlState):
        if self.out.ctl_status == 0:
            self.out.body_force_cmd = np.zeros(3)
            self.out.body_accel_cmd = np.zeros(3)
            return
        target_vel = self.out.traj_vel
        target_accel = self.out.traj_accel
        if self.mode == "stopped":
            target_vel = np.zeros(3)
            target_accel = np.zeros(3)
        Kp_lin = safe_divide(state.pos_kp, state.vel_kd)  # This is interesting
        v = -1 * state.est_V_B_ISS_ISS
        if self.out.ctl_status > 1:
            v += target_vel + Kp_lin * self.out.pos_err + self.out.pos_err_int
        a = quat_to_rmat(state.est_quat_ISS2B) @ v
        a = a * state.vel_kd
        b = quat_to_rmat(state.est_quat_ISS2B) @ (target_accel * tun_accel_gain)
        self.out.body_force_cmd = self.saturate_vector(
            state.mass * (a + b), tun_ctl_linear_force_limit
        )
        self.out.body_accel_cmd = self.out.body_force_cmd / state.mass

    def saturate_vector(self, v, limit):
        mag = np.linalg.norm(v)
        if mag < limit:
            return v
        else:
            return (limit / mag) * v

    # Just seems to add an input to the accumulator but scaled by 1/62.5
    # Maybe the 62.5 is a sampling rate? unclear
    def discrete_time_integrator(
        self, input, accumulator, ctl_status, upper_limit, lower_limit
    ):
        # TODO: accumulator is passed by reference.... so make a class attribute?
        output = np.zeros(3)
        if ctl_status <= 1:
            accumulator = np.zeros(3)
            return output
        accumulator += input / 62.5  # ???
        output = accumulator
        for i in range(3):
            if output[i] > upper_limit:
                output[i] = upper_limit
            elif output[i] < lower_limit:
                output[i] = lower_limit
        return output

    # Finds the error between the current estimated position and the target from the trajectory
    # Check on the gain division
    def find_pos_error(self, state: ControlState, cmd: ControlCommand):
        target = self.out.traj_pos
        if self.mode == "stopped":
            target = self.prev_position
        self.out.pos_err = target - state.est_P_B_ISS_ISS
        Ki_lin = safe_divide(state.pos_ki, state.vel_kd)
        self.out.pos_err_int = self.discrete_time_integrator(
            Ki_lin * self.out.pos_err,
            self.linear_integrator,
            self.out.ctl_status,
            tun_ctl_pos_sat_upper,
            tun_ctl_pos_sat_lower,
        )

    # This pretty much seems to be a helper function to define when they enter "stop" mode
    # Some of the math here is a little weird (why square the error?) but we can redefine
    # the "stopping" logic on our own
    def update_ctl_status(self, state: ControlState):
        pos_err = np.linalg.norm(self.prev_position - state.est_P_B_ISS_ISS) ** 2
        quat_err = self.quat_error(state.est_quat_ISS2B, self.prev_att)
        if (
            pos_err > tun_ctl_stopped_pos_thresh
            or abs(quat_err) > tun_ctl_stopped_quat_thresh
            and mode_cmd == constants.ctl_stopping_mode
        ):
            self.out.ctl_status = "stopping"
        elif self.stopped_mode:
            self.out.ctl_status == "stopped"
        else:
            self.out.ctl_status = mode_cmd

    # This updates their trackers on the previous position/attitude
    # (presumably, for a derivative error calculation)
    def update_previous(self, state: ControlState):
        if not stopped:
            self.prev_position = state.est_P_B_ISS_ISS
            self.prev_att = state.est_quat_ISS2B

    # TODO Add this to quaternion.py?
    def quat_error(self, cmd, actual):
        out = normalize(conjugate(actual) * cmd)
        return np.arccos(out[3]) * 2  # qw

    # Seems to be helping filter the position/attitude for determining if they've stopped
    def butterworth_filter(self, input, delay, sum_out):
        # WTF
        gain_1 = 0.0031317642291927056
        gain_2 = -0.993736471541614597
        tmp_out = input * gain_1
        previous_gain = delay * gain_2
        tmp_out = tmp_out - previous_gain
        output = tmp_out + delay
        # THIS IS A POINTER - needs to be handled differently
        sum_out += output**2
        return tmp_out, sum_out

    # See butterworth
    def filter_threshold(self, vec, threshold, prev):
        # PREVIOUS IS PASSED BY REF - handle this
        sum_ = 0.0
        for i in range(3):
            prev[i], sum_ = self.butterworth_filter(vec[i], prev[i], sum_)
        return sum_ < threshold

    # Defines transitions between idle/stopping/stopped (and nominal, presumably)
    def update_mode(self, state: ControlState, cmd: ControlCommand):
        if state.est_confidence != constants.ase_status_converged:
            mode_cmd = constants.ctl_idle_mode
        else:
            mode_cmd = cmd.mode
        for i in range(3, -1, -1):
            prev_mode_cmd[i + 1] = prev_mode_cmd[i]
        prev_mode_cmd[0] = mode_cmd
        vel_below_threshold = self.filter_threshold(
            state.est_V_B_ISS_ISS, tun_ctl_stopping_vel_thresh, prev_filter_vel
        )
        omega_below_threshold = self.filter_threshold(
            state.est_omega_B_ISS_B, tun_ctl_stopping_omega_thresh, prev_filter_omega
        )
        stopped_mode = False
        # ?????
        if vel_below_threshold and omega_below_threshold:
            if (
                prev_mode_cmd[4] == constants.ctl_stopping_mode
                and prev_mode_cmd[4] == prev_mode_cmd[3]
                and prev_mode_cmd[3] == prev_mode_cmd[2]
                and prev_mode_cmd[2] == prev_mode_cmd[1]
            ):
                stopped_mode = True
