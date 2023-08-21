"""Astrobee constants associated with motion planning and control

TODO make flight mode an input somewhere, rather than a hardcoded variable here?

Sourced from:
- astrobee/gds_configs/AllOperatingLimitsConfig.json
- A Brief Guide to Astrobee
"""

from enum import Enum
import numpy as np


class FlightModes(Enum):
    # Standard flight modes specified in the NASA codebase
    QUIET = 0
    DIFFICULT = 1
    NOMINAL = 2
    LOMO = 3  # (not entirely sure what this mode means. locomotion?)
    # Custom flight modes
    MAXED_OUT = -1


# TODO: Update this parameter if testing a different mode
# The maxed out flight mode is good for some of our planing methods, which can be a bit conservative
# when enforcing these speed limits
FLIGHT_MODE = FlightModes.MAXED_OUT

if FLIGHT_MODE == FlightModes.QUIET:
    LINEAR_SPEED_LIMIT = 0.02  # m/s
    LINEAR_ACCEL_LIMIT = 0.002  # m/s^2
    ANGULAR_SPEED_LIMIT = 0.0174  # rad/s
    ANGULAR_ACCEL_LIMIT = 0.0174  # rad/s^2
elif FLIGHT_MODE == FlightModes.DIFFICULT:
    LINEAR_SPEED_LIMIT = 0.4  # m/s
    LINEAR_ACCEL_LIMIT = 0.02  # m/s^2
    ANGULAR_SPEED_LIMIT = 0.5236  # rad/s
    ANGULAR_ACCEL_LIMIT = 0.2500  # rad/s^2
elif FLIGHT_MODE == FlightModes.NOMINAL:
    LINEAR_SPEED_LIMIT = 0.2  # m/s
    LINEAR_ACCEL_LIMIT = 0.0175  # m/s^2
    ANGULAR_SPEED_LIMIT = 0.1745  # rad/s
    ANGULAR_ACCEL_LIMIT = 0.1745  # rad/s^2
elif FLIGHT_MODE == FlightModes.LOMO:
    LINEAR_SPEED_LIMIT = 0.15  # m/s
    LINEAR_ACCEL_LIMIT = 0.0175  # m/s^2
    ANGULAR_SPEED_LIMIT = 0.0873  # rad/s
    ANGULAR_ACCEL_LIMIT = 0.1745  # rad/s^2
elif FLIGHT_MODE == FlightModes.MAXED_OUT:
    # Use the absolute maximum values that our actuators can provide
    # Acceleration limits are calculated using standard rigid body dynamics (F=ma, t=Ia)
    # with the Astrobee's mass and inertia properties
    # Speed limits are left the same as the ISS_DIFFICULT profile
    LINEAR_SPEED_LIMIT = 0.4  # m/s
    LINEAR_ACCEL_LIMIT = 0.042  # m/s^2
    ANGULAR_SPEED_LIMIT = 0.5236  # rad/s
    ANGULAR_ACCEL_LIMIT = 0.2840  # rad/s^2
else:
    raise NotImplementedError("Flight mode has not been implemented")


# For forces/torques, assume we can operate at up to the max motor speed in RPM
# See the Astrobee guide for more values if this is not the case
MAX_FORCE = np.array([0.849, 0.406, 0.486])  # Max force applied in xyz axes from fans
MAX_TORQUE = MAX_FORCE / 10  # Max torques about xyz axes
# NOTE: The torque values are an approximation. See guide for reasoning
