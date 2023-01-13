"""Script to load Astrobee URDFs and view them in a Pybullet GUI window

TODO
- Make the ISS collision debugging OBJ file import as white rather than black
- Get bullet to import the right colors for the astrobee? May need to specify in URDF
"""

import time
import pybullet

from astrobee_pybullet.utils.pybullet_wrapper import (
    initialize_pybullet,
    load_urdf,
    set_gravity,
)


def main():
    initialize_pybullet(use_deformability=True)
    astrobee_urdf = "astrobee_pybullet/resources/urdf/astrobee.urdf"
    iss_urdf = "astrobee_pybullet/resources/urdf/iss.urdf"
    # Debugging URDF for collision visualization
    astrobee_debug_urdf = "astrobee_pybullet/resources/urdf/astrobee_collision.urdf"
    iss_debug_urdf = "astrobee_pybullet/resources/urdf/iss_collision.urdf"

    bag_urdf = "astrobee_pybullet/resources/urdf/cargo_bag.urdf"
    bag_debug_urdf = "astrobee_pybullet/resources/urdf/cargo_bag_rigid.urdf"

    # Some starting positions for the astrobee within the ISS
    cupola = (-5.0, 0.0, 5.0)  # Don't use this one for now until the mesh is fixed
    us_lab = (3.0, 0.0, 5.0)
    outside = (0.0, -5.0, 5.0)  # A random place outside the ISS
    origin = (0.0, 0.0, 0.0)

    # Loading the standard ISS and Astrobee
    load_urdf(iss_urdf, fixed=True)
    load_urdf(astrobee_urdf, pos=us_lab)
    # load_urdf(bag_urdf, pos=origin)
    load_urdf(bag_debug_urdf, pos=outside)
    # load_urdf("cube.urdf", pos=origin)

    # p.loadSoftBody("astrobee_pybullet/resources/meshes/cargo_bag.obj")
    # p.loadSoftBody("cube.obj")

    # Loading the "debugging" versions to visualize the collision info
    # load_urdf(iss_debug_urdf, fixed=True)
    # load_urdf(astrobee_debug_urdf, pos=us_lab)

    # For some reason I had to call this after the loadSoftBody because the soft body would somehow
    # still be affected by gravity even if everything else wasn't?
    set_gravity("iss")

    # time.sleep(10)
    for _ in range(100000):
        pybullet.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
