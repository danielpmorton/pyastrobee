"""Script to load Astrobee URDFs and view them in a Pybullet GUI window
"""

import time
import pybullet as p

from src.scripts.set_env_vars import set_env_vars
from src.utils.pybullet_wrapper import initialize_pybullet, load_urdf


def main():
    set_env_vars()
    initialize_pybullet()
    astrobee_urdf = "src/resources/astrobee.urdf"
    iss_urdf = "src/resources/iss.urdf"
    # Debugging URDF for collision visualization
    astrobee_debug_urdf = "src/resources/astrobee_collision.urdf"

    load_urdf(iss_urdf, fixed=True)
    # load_urdf(astrobee_urdf, pos=(-5.0, 0.0, 5))
    # load_urdf(astrobee_urdf)
    load_urdf(astrobee_debug_urdf, pos=(-5.0, 0.0, 5))

    time.sleep(10)
    for _ in range(100000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
