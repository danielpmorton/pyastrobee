"""Script to add the locations of external resources to the environment variables

Most likely, the function will be called from another script to load the info
"""

import os


def set_env_vars() -> None:
    """Sets the environment variables for the locations of the following:

    - ASTROBEE_WS
    - ASTROBEE_SRC
    - ASTROBEE_DESCRIPTION
    - ASTROBEE_MEDIA
    - RPBI_WS
    - RPBI_SRC
    """
    # Astrobee locations
    os.environ["ASTROBEE_WS"] = "$HOME/astrobee"
    os.environ["ASTROBEE_SRC"] = f"{os.environ.get('ASTROBEE_WS')}/src"
    # Note that "description" will refer to the description/description folder
    os.environ[
        "ASTROBEE_DESCRIPTION"
    ] = f"{os.environ.get('ASTROBEE_SRC')}/description/description"
    os.environ["ASTROBEE_MEDIA"] = f"{os.environ.get('ASTROBEE_SRC')}/description/media"

    # ROS-Pybullet locations
    os.environ["RPBI_WS"] = "$HOME/rpbi_workspace"
    os.environ["RPBI_SRC"] = f"{os.environ.get('RPBI_WS')}/src/ros_pybullet_interface"


if __name__ == "__main__":
    set_env_vars()
