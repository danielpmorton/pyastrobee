"""Script to load Astrobee URDFs and view them in a Pybullet GUI window

TODO
- Make the ISS collision debugging OBJ file import as white rather than black
- Get bullet to import the right colors for the astrobee? May need to specify in URDF
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
    iss_debug_urdf = "src/resources/iss_collision.urdf"

    # Some starting positions for the astrobee within the ISS
    cupola = (-5.0, 0.0, 5)  # Don't use this one for now until the mesh is fixed
    us_lab = (3.0, 0.0, 5)

    # Loading the standard ISS and Astrobee
    load_urdf(iss_urdf, fixed=True)
    load_urdf(astrobee_urdf, pos=us_lab)

    # Loading the "debugging" versions to visualize the collision info
    # load_urdf(iss_debug_urdf, fixed=True)
    # load_urdf(astrobee_debug_urdf, pos=us_lab)

    time.sleep(10)
    for _ in range(100000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
