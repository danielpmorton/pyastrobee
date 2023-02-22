"""Astrobee constants associated with motion planning and control

Sourced from:
- astrobee/gds_configs/AllOperatingLimitsConfig.json
- A Brief Guide to Astrobee
"""


import numpy as np

# We'll set our speed/accel limits as specified by the "iss_difficult" flight mode
# TODO decide if we want to handle different flight modes
LINEAR_SPEED_LIMIT = 0.4  # m/s
LINEAR_ACCEL_LIMIT = 0.02  # m/s^2
ANGULAR_SPEED_LIMIT = 0.5236  # rad/s
ANGULAR_ACCEL_LIMIT = 0.2500  # rad/s^2

# For forces/torques, assume we can operate at up to the max motor speed in RPM
# See the Astrobee guide for more values if this is not the case
MAX_FORCE = np.array([0.849, 0.406, 0.486])  # Max force applied in xyz axes from fans
MAX_TORQUE = MAX_FORCE / 10  # Max torques about xyz axes
# NOTE: The torque values are an approximation. See guide for reasoning
