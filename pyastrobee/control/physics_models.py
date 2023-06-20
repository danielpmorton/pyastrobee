"""Physics models for enhancing simulation/control accuracy

In general, these are based on the Gazebo plugins which can be found at:
https://github.com/nasa/astrobee/tree/master/simulation/src
"""

import numpy as np
import numpy.typing as npt

from pyastrobee.core.astrobee import Astrobee  # Circular import?
from pyastrobee.config.astrobee_geom import CROSS_SECTION_AREA


def drag_force_model(
    vel: npt.ArrayLike, area: float = CROSS_SECTION_AREA
) -> np.ndarray:
    """Determines the drag force vector acting on the Astrobee for a given velocity

    - This does not currently account for any drag due to the rotation of the Astrobee, or anything
      related to the position/motion of the arm
    - See the drag plugin in nasa/astrobee/simulation for more details
    - If calculating drag for another moving object (like a cargo bag), use a different area

    Args:
        vel (npt.ArrayLike): Current velocity vector of the Astrobee, shape (3,)
        area (float, optional): Area of the body (e.g. the Astrobee) opposing the flow.
            Defaults to CROSS_SECTION_AREA, the default value that NASA uses

    Returns:
        np.ndarray: Drag force vector opposing the current velocity, shape (3,)
    """
    drag_coeff = 1.05
    density = 1.225
    return -0.5 * drag_coeff * area * density * vel * np.linalg.norm(vel)


# document this! get the meaning of "cdp"
def area_to_cdp_model(area):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    # These values observed from plotting the data
    mu = 0.00057
    sig = 0.0045
    offset = 0.0058
    scale = 0.084775 - offset
    return offset + scale * gaussian(area, mu, sig)


# random idea... integrate this into the force controller?
# or, make a separate fan controller that actually uses the fan model?
# See https://github.com/nasa/astrobee/blob/master/simulation/src/gazebo_model_plugin_pmc/gazebo_model_plugin_pmc.cc
# and https://github.com/nasa/astrobee/blob/master/gnc/pmc/include/pmc/pmc_sim.h
# and https://github.com/nasa/astrobee/blob/master/gnc/pmc/src/pmc_sim.cc
def fan_force_model(robot: Astrobee):
    cur_pose = robot.pose
    cur_vel = robot.velocity
    cur_omega = robot.angular_velocity
    robot_center = cur_pose[:3]
    nozzle_offsets = NotImplemented
    nozzle_moment_arm = NotImplemented
    nozzle_orientation = NotImplemented

    # Might need to take into account the curret

    # I dunno what this exact model should be
    # some simple ones would probably be a quadratic or linear fit
    # (so we get the nominal max force at 0 velocity, and 0 force at max velocity)
    # We should probably ask them for clarification
    pass
