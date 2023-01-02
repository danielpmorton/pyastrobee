"""Script to load Astrobee URDFs and view them in a Pybullet GUI window
"""

import os
import time
from typing import Union, Optional
import pybullet as p
import pybullet_data

from src.scripts.set_env_vars import set_env_vars
from src.utils.pybullet_wrapper import initialize_pybullet, load_urdf


def main():
    set_env_vars()
    client = initialize_pybullet()
    # TODO: need to figure out how to load the xacro parameters
    astrobee_urdf_loc = (
        f"{os.environ.get('ASTROBEE_DESCRIPTION')}/urdf/model.urdf.xacro"
    )
    # iss_urdf_loc = f"{os.environ.get('ASTROBEE_MEDIA')}/astrobee_iss/urdf/model.urdf"

    load_urdf(astrobee_urdf_loc)

    # This can probably be removed
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
