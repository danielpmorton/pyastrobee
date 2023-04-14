"""Translation of gnc/ctl/src/ctl.cc to python

WORK IN PROGRESS

https://github.com/nasa/astrobee/blob/master/gnc/ctl/src/ctl.cc
"""

from pyastrobee.utils.math_utils import normalize, safe_divide
from pyastrobee.utils.rotations import rotate_point, quat_to_rmat, quaternion_dist


class ControlOutput:
    def __init__(self):
        pass

    pass


class Control:
    def __init__(self):
        # See Control::Initialize and Control::ReadParams
        pass

    def step(self, dt, state, cmd):
        pass

    def omega_matrix(self, input):
        pass

    def forward_trajectory(self, dt, state, cmd):
        pass

    def find_att_error(self, state):
        pass

    def fing_body_alpha_torque_cmd(self, state):
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
