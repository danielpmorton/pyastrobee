"""Script to load Astrobee URDFs and view them in a Pybullet GUI window
"""

import os

from src.scripts.set_env_vars import set_env_vars


def main():
    set_env_vars()
    # TODO: need to figure out how to load the xacro parameters
    astrobee_urdf_loc = (
        f"{os.environ.get('ASTROBEE_DESCRIPTION')}/urdf/model.urdf.xacro"
    )
    iss_urdf_loc = f"{os.environ.get('ASTROBEE_MEDIA')}/astrobee_iss/urdf/model.urdf"


if __name__ == "__main__":
    main()
